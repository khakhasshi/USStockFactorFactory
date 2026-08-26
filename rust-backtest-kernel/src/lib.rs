//! FactorFactory's dependency-free Rust event kernel.
//!
//! The ABI is deliberately columnar. Python owns all buffers and Rust never
//! retains a pointer after `ff_run_v1` returns. This keeps the event ledger
//! deterministic and avoids Python-object creation in the hot loop.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_char;

const ABI_VERSION: u32 = 2;
const OK: i32 = 0;
const ERR_ABI: i32 = 1;
const ERR_CAPACITY: i32 = 2;
const ERR_INPUT: i32 = 3;

#[repr(C)]
pub struct FFColumnsV1 {
    pub abi_version: u32,
    pub row_count: usize,
    pub session_count: usize,
    pub session_offsets: *const usize,
    pub day: *const i32,
    pub symbol: *const i32,
    pub univ_rank: *const i32,
    pub factor: *const f64,
    pub raw_open: *const f64,
    pub raw_close: *const f64,
    pub volume: *const f64,
    pub adv20_prev: *const f64,
    pub atr_pct: *const f64,
    pub vol20_prev: *const f64,
    pub adjustment_factor: *const f64,
    pub can_buy: *const u8,
    pub can_sell: *const u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FFConfigV1 {
    pub abi_version: u32,
    pub market: u8, // 1=US, 2=A-share
    pub mode: u8,   // 1=long-only, 2=long-short
    pub direction: i8,
    pub account_type: u8, // 1=cash, 2=margin
    pub impact_model: u8, // 1=fixed, 2=linear, 3=square-root
    pub universe_n: i32,
    pub rebalance_every: i32,
    pub max_positions: i32,
    pub initial_capital: f64,
    pub top_fraction: f64,
    pub slippage_bps: f64,
    pub spread_bps: f64,
    pub max_volume_participation: f64,
    pub cash_buffer_fraction: f64,
    pub max_gross_leverage: f64,
    pub max_position_weight: f64,
    pub min_trade_notional: f64,
    pub rebalance_buffer_pct: f64,
    pub long_gross_target: f64,
    pub short_gross_target: f64,
    pub borrow_cost_bps_annual: f64,
    pub margin_interest_bps_annual: f64,
    pub impact_coefficient_bps: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FFTradeV1 {
    pub fill_seq: u64,
    pub order_seq: u64,
    pub signal_day: i32,
    pub trade_day: i32,
    pub symbol: i32,
    pub side: i8, // 1=buy, -1=sell
    pub requested_quantity: f64,
    pub filled_quantity: f64,
    pub reference_price: f64,
    pub fill_price: f64,
    pub commission: f64,
    pub stamp_duty: f64,
    pub transfer_fee: f64,
    pub total_fees: f64,
    pub cash_before: f64,
    pub cash_after: f64,
    pub position_before: f64,
    pub position_after: f64,
    pub participation: f64,
    pub slippage_cost: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FFDailyV1 {
    pub day: i32,
    pub close_nlv: f64,
    pub net_nav: f64,
    pub daily_return: f64,
    pub cash: f64,
    pub long_market_value: f64,
    pub short_market_value: f64,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub turnover: f64,
    pub borrow_fee: f64,
    pub margin_interest: f64,
    pub fills: u64,
    pub orders_created: u64,
    pub positions: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FFSummaryV1 {
    pub abi_version: u32,
    pub status: i32,
    pub trade_count: usize,
    pub daily_count: usize,
    pub order_count: u64,
    pub rejected_orders: u64,
    pub partial_orders: u64,
    pub final_nlv: f64,
    pub cumulative_fees: f64,
    pub cumulative_slippage: f64,
    pub cumulative_borrow: f64,
    pub cumulative_margin_interest: f64,
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // Reserved by ABI v1 for inverse-volatility/ATR sizing.
struct Row {
    index: usize,
    day: i32,
    symbol: i32,
    rank: i32,
    factor: f64,
    open: f64,
    close: f64,
    volume: f64,
    adv: f64,
    atr_pct: f64,
    vol20: f64,
    adjustment: f64,
    can_buy: bool,
    can_sell: bool,
}

#[derive(Clone)]
struct Order {
    seq: u64,
    signal_day: i32,
    execute_day: i32,
    symbol: i32,
    target: f64,
}

#[derive(Clone, Copy, Default)]
struct PositionState {
    quantity: f64,
    settled_quantity: f64,
}

#[inline]
fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[inline]
fn round_to(value: f64, digits: i32) -> f64 {
    let scale = 10_f64.powi(digits);
    (value * scale).round() / scale
}

#[inline]
fn money(value: f64) -> f64 {
    // Python's authority uses Decimal(str(value), ROUND_HALF_UP). Values in
    // this engine are non-negative and statement prices are frozen at six
    // decimals, so an epsilon below one micro-cent removes only binary-float
    // representation noise at exact half-cent boundaries.
    (value.max(0.0) * 100.0 + 0.5 + 1e-9).floor() / 100.0
}

#[inline]
fn ashare_stamp_rate(day: i32) -> f64 {
    // Days since 1970-01-01: 2023-08-28 = 19597.
    if day >= 19_597 { 0.0005 } else { 0.001 }
}

#[inline]
fn ashare_transfer_rate(day: i32) -> f64 {
    // 2022-04-29 = 19111.
    if day >= 19_111 { 0.00001 } else { 0.00002 }
}

fn fees(cfg: FFConfigV1, side: i8, quantity: f64, price: f64, day: i32) -> (f64, f64, f64, f64) {
    let notional = quantity * price;
    if cfg.market == 2 {
        let commission = money(notional * 0.0002);
        let transfer = money(notional * ashare_transfer_rate(day));
        let stamp = if side < 0 {
            money(notional * ashare_stamp_rate(day))
        } else {
            0.0
        };
        let total = money(commission + transfer + stamp);
        (commission, stamp, transfer, total)
    } else {
        let commission = money((quantity * 0.005).max(1.0).min(notional * 0.01));
        (commission, 0.0, 0.0, commission)
    }
}

struct Engine<'a> {
    cfg: FFConfigV1,
    trades: &'a mut [FFTradeV1],
    daily: &'a mut [FFDailyV1],
    trade_count: usize,
    daily_count: usize,
    cash: f64,
    positions: BTreeMap<i32, PositionState>,
    last_close: HashMap<i32, f64>,
    last_adjustment: HashMap<i32, f64>,
    pending: Vec<Order>,
    order_seq: u64,
    fill_seq: u64,
    rejected: u64,
    partial: u64,
    previous_close_nlv: f64,
    cumulative_fees: f64,
    cumulative_slippage: f64,
    cumulative_borrow: f64,
    cumulative_margin: f64,
}

impl<'a> Engine<'a> {
    fn lot(&self) -> f64 {
        if self.cfg.market == 2 { 100.0 } else { 1.0 }
    }

    fn price(&self, symbol: i32, market: &HashMap<i32, Row>, open: bool) -> f64 {
        if let Some(row) = market.get(&symbol) {
            let value = if open { row.open } else { row.close };
            if finite_positive(value) {
                return value;
            }
        }
        *self.last_close.get(&symbol).unwrap_or(&0.0)
    }

    fn nlv(&self, market: &HashMap<i32, Row>, open: bool) -> (f64, f64, f64) {
        let mut long_value = 0.0;
        let mut short_value = 0.0;
        for (&symbol, state) in &self.positions {
            let value = state.quantity * self.price(symbol, market, open);
            if value >= 0.0 {
                long_value += value;
            } else {
                short_value += value;
            }
        }
        (
            self.cash + long_value + short_value,
            long_value,
            short_value,
        )
    }

    fn gross_leverage(&self, market: &HashMap<i32, Row>) -> f64 {
        let (nlv, long_value, short_value) = self.nlv(market, true);
        if nlv <= 1e-12 {
            f64::INFINITY
        } else {
            (long_value + short_value.abs()) / nlv
        }
    }

    fn projected_leverage(
        &self,
        symbol: i32,
        signed: f64,
        fill_price: f64,
        day: i32,
        market: &HashMap<i32, Row>,
    ) -> f64 {
        let current = self
            .positions
            .get(&symbol)
            .map(|p| p.quantity)
            .unwrap_or(0.0);
        let valuation = self.price(symbol, market, true);
        let (nlv, long_value, short_value) = self.nlv(market, true);
        let gross = long_value + short_value.abs();
        let after = current + signed;
        let gross_after = gross - current.abs() * valuation + after.abs() * valuation;
        let (_, _, _, fee) = fees(
            self.cfg,
            if signed > 0.0 { 1 } else { -1 },
            signed.abs(),
            fill_price,
            day,
        );
        let nlv_after = nlv - signed * (fill_price - valuation) - fee;
        if nlv_after <= 1e-12 {
            f64::INFINITY
        } else {
            gross_after / nlv_after
        }
    }

    fn leverage_cap(
        &self,
        symbol: i32,
        requested: f64,
        side: i8,
        fill_price: f64,
        day: i32,
        market: &HashMap<i32, Row>,
    ) -> f64 {
        let signed_direction = side as f64;
        let current = self
            .positions
            .get(&symbol)
            .map(|p| p.quantity)
            .unwrap_or(0.0);
        let full_signed = signed_direction * requested;
        let full = self.projected_leverage(symbol, full_signed, fill_price, day, market);
        if (current + full_signed).abs() <= current.abs() + 1e-10
            || full <= self.cfg.max_gross_leverage + 1e-10
        {
            return requested;
        }
        let reducing = if current * signed_direction < 0.0 {
            requested.min(current.abs())
        } else {
            0.0
        };
        if reducing <= 0.0 && self.gross_leverage(market) > self.cfg.max_gross_leverage + 1e-10 {
            return 0.0;
        }
        let mut low = reducing;
        let mut high = requested;
        for _ in 0..48 {
            let middle = (low + high) / 2.0;
            if self.projected_leverage(symbol, signed_direction * middle, fill_price, day, market)
                <= self.cfg.max_gross_leverage + 1e-10
            {
                low = middle;
            } else {
                high = middle;
            }
        }
        let allowed =
            reducing + ((low - reducing).max(0.0) / self.lot() + 1e-12).floor() * self.lot();
        round_to(allowed.min(requested), 6)
    }

    fn apply_adjustments_and_settle(&mut self, market: &HashMap<i32, Row>) {
        if self.cfg.market == 2 {
            for state in self.positions.values_mut() {
                if state.quantity > 0.0 {
                    state.settled_quantity = state.quantity.abs();
                }
            }
        }
        let symbols: Vec<i32> = self.positions.keys().copied().collect();
        for symbol in symbols {
            let Some(row) = market.get(&symbol) else {
                continue;
            };
            if !finite_positive(row.adjustment) {
                continue;
            }
            let Some(&previous) = self.last_adjustment.get(&symbol) else {
                continue;
            };
            if !finite_positive(previous) {
                continue;
            }
            let ratio = row.adjustment / previous;
            if (ratio - 1.0).abs() <= 1e-8 {
                continue;
            }
            if let Some(state) = self.positions.get_mut(&symbol) {
                state.quantity = round_to(state.quantity * ratio, 6);
                state.settled_quantity = round_to(state.settled_quantity * ratio, 6);
            }
            for order in &mut self.pending {
                if order.symbol == symbol {
                    order.target = round_to(order.target * ratio, 6);
                }
            }
        }
    }

    fn max_affordable_buy(&self, requested: f64, price: f64, day: i32) -> f64 {
        if self.cfg.account_type != 1 {
            return requested;
        }
        let lot = self.lot();
        let reserve = self.previous_close_nlv.max(0.0) * self.cfg.cash_buffer_fraction;
        let available = (self.cash - reserve).max(0.0);
        let mut quantity = (requested.min(available / price.max(1e-12)) / lot).floor() * lot;
        while quantity > 0.0 {
            let (_, _, _, fee) = fees(self.cfg, 1, quantity, price, day);
            if quantity * price + fee <= available + 1e-9 {
                return quantity;
            }
            quantity -= lot;
        }
        0.0
    }

    fn execute(&mut self, order: &Order, day: i32, market: &HashMap<i32, Row>) -> f64 {
        let current = self
            .positions
            .get(&order.symbol)
            .map(|p| p.quantity)
            .unwrap_or(0.0);
        let requested_signed = round_to(order.target - current, 6);
        if requested_signed.abs() < 1e-8 {
            return 0.0;
        }
        let requested = requested_signed.abs();
        let Some(row) = market.get(&order.symbol) else {
            self.rejected += 1;
            return requested;
        };
        if !finite_positive(row.open) {
            self.rejected += 1;
            return requested;
        }
        let side: i8 = if requested_signed > 0.0 { 1 } else { -1 };
        if (side > 0 && !row.can_buy) || (side < 0 && !row.can_sell) {
            self.rejected += 1;
            return requested;
        }
        let volume = if finite_positive(row.adv) {
            row.adv
        } else {
            row.volume.max(0.0)
        };
        if volume <= 0.0 {
            self.rejected += 1;
            return requested;
        }
        let mut filled = requested
            .min((volume * self.cfg.max_volume_participation / self.lot()).floor() * self.lot());
        let reference = round_to(row.open, 6);
        let initial_participation = filled / volume;
        let impact_bps = self.cfg.slippage_bps
            + self.cfg.spread_bps / 2.0
            + match self.cfg.impact_model {
                2 => self.cfg.impact_coefficient_bps * initial_participation,
                3 => self.cfg.impact_coefficient_bps * initial_participation.max(0.0).sqrt(),
                _ => 0.0,
            };
        let fill_price = round_to(reference * (1.0 + side as f64 * impact_bps / 10_000.0), 6);
        if side > 0 {
            filled = self.max_affordable_buy(filled, fill_price, day);
        }
        if self.cfg.market == 2 && side < 0 && current > 0.0 {
            filled = filled.min(
                self.positions
                    .get(&order.symbol)
                    .map(|p| p.settled_quantity)
                    .unwrap_or(requested),
            );
        }
        filled = self.leverage_cap(order.symbol, filled, side, fill_price, day, market);
        filled = round_to(filled, 6);
        if filled <= 0.0 {
            self.rejected += 1;
            return requested;
        }
        let (commission, stamp, transfer, total_fee) =
            fees(self.cfg, side, filled, fill_price, day);
        let cash_before = self.cash;
        let signed = side as f64 * filled;
        self.cash -= signed * fill_price + total_fee;
        let position_after = round_to(current + signed, 6);
        if position_after.abs() < 1e-8 {
            self.positions.remove(&order.symbol);
        } else {
            let settled = if self.cfg.market == 2 && position_after > 0.0 {
                let prior = self
                    .positions
                    .get(&order.symbol)
                    .map(|p| p.settled_quantity)
                    .unwrap_or(0.0);
                if side < 0 {
                    (prior - filled).max(0.0)
                } else {
                    prior
                }
            } else {
                position_after.abs()
            };
            self.positions.insert(
                order.symbol,
                PositionState {
                    quantity: position_after,
                    settled_quantity: settled,
                },
            );
        }
        self.fill_seq += 1;
        let unfilled = (requested - filled).max(0.0);
        if unfilled > 1e-8 {
            self.partial += 1;
        }
        self.cumulative_fees += total_fee;
        let slippage_cost = filled * (fill_price - reference).abs();
        self.cumulative_slippage += slippage_cost;
        if self.trade_count < self.trades.len() {
            self.trades[self.trade_count] = FFTradeV1 {
                fill_seq: self.fill_seq,
                order_seq: order.seq,
                signal_day: order.signal_day,
                trade_day: day,
                symbol: order.symbol,
                side,
                requested_quantity: round_to(requested, 6),
                filled_quantity: filled,
                reference_price: reference,
                fill_price,
                commission,
                stamp_duty: stamp,
                transfer_fee: transfer,
                total_fees: total_fee,
                cash_before: round_to(cash_before, 6),
                cash_after: round_to(self.cash, 6),
                position_before: round_to(current, 6),
                position_after,
                participation: round_to(filled / volume, 8),
                slippage_cost: round_to(slippage_cost, 6),
            };
            self.trade_count += 1;
        }
        unfilled
    }

    fn enforce_open_leverage(
        &mut self,
        day: i32,
        next_day: Option<i32>,
        market: &HashMap<i32, Row>,
    ) {
        let (nlv, long_value, short_value) = self.nlv(market, true);
        let gross = long_value + short_value.abs();
        let leverage = if nlv > 1e-12 {
            gross / nlv
        } else {
            f64::INFINITY
        };
        let tolerance = 1e-8_f64.max(self.cfg.max_gross_leverage * 1e-6);
        if leverage <= self.cfg.max_gross_leverage + tolerance {
            return;
        }
        let (long_target, short_target) = self.effective_targets();
        let operating = long_target + short_target;
        let recovery_band = 0.01_f64.max(self.cfg.cash_buffer_fraction);
        let recovery_target = (if operating > 0.0 {
            operating
        } else {
            self.cfg.max_gross_leverage
        })
        .min(self.cfg.max_gross_leverage * (1.0 - recovery_band));
        let scale = if gross > 1e-12 {
            (nlv.max(0.0) * recovery_target / gross).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut holdings: Vec<(i32, f64)> = self
            .positions
            .iter()
            .map(|(&symbol, state)| (symbol, state.quantity))
            .collect();
        holdings.sort_by(|a, b| {
            let av = a.1.abs() * self.price(a.0, market, true);
            let bv = b.1.abs() * self.price(b.0, market, true);
            bv.partial_cmp(&av).unwrap_or(Ordering::Equal)
        });
        for (symbol, current) in holdings {
            let target_abs = (current.abs() * scale / self.lot() + 1e-12).floor() * self.lot();
            let target = current.signum() * target_abs;
            if (target - current).abs() < 1e-8 {
                continue;
            }
            self.order_seq += 1;
            let order = Order {
                seq: self.order_seq,
                signal_day: day,
                execute_day: day,
                symbol,
                target,
            };
            let unfilled = self.execute(&order, day, market);
            if unfilled > 1e-8 {
                if let Some(execute_day) = next_day {
                    self.pending.push(Order {
                        execute_day,
                        ..order
                    });
                }
            }
        }
    }

    fn effective_targets(&self) -> (f64, f64) {
        let configured = self.cfg.long_gross_target + self.cfg.short_gross_target;
        if configured <= 1e-12 {
            return (0.0, 0.0);
        }
        let scale = (self.cfg.max_gross_leverage * (1.0 - self.cfg.cash_buffer_fraction)
            / configured)
            .clamp(0.0, 1.0);
        (
            self.cfg.long_gross_target * scale,
            self.cfg.short_gross_target * scale,
        )
    }

    fn targets(&self, rows: &[Row], nlv: f64) -> BTreeMap<i32, f64> {
        let mut eligible: Vec<&Row> = rows
            .iter()
            .filter(|r| {
                r.rank <= self.cfg.universe_n && r.factor.is_finite() && finite_positive(r.close)
            })
            .collect();
        eligible.sort_by(|a, b| {
            let av = a.factor * self.cfg.direction as f64;
            let bv = b.factor * self.cfg.direction as f64;
            bv.partial_cmp(&av)
                .unwrap_or(Ordering::Equal)
                .then(a.index.cmp(&b.index))
        });
        if eligible.is_empty() {
            return BTreeMap::new();
        }
        let mut count = ((eligible.len() as f64 * self.cfg.top_fraction) as usize).max(1);
        let leg_cap = if self.cfg.mode == 2 {
            (self.cfg.max_positions / 2).max(1)
        } else {
            self.cfg.max_positions
        } as usize;
        count = count.min(leg_cap);
        let (long_gross, short_gross) = self.effective_targets();
        let lot = self.lot();
        let cap = nlv * self.cfg.max_position_weight;
        let mut target = BTreeMap::new();
        let add_leg = |slice: &[&Row], gross: f64, sign: f64, target: &mut BTreeMap<i32, f64>| {
            if slice.is_empty() || nlv <= 0.0 || gross <= 0.0 {
                return;
            }
            let per_name = (nlv * gross / slice.len() as f64).min(cap);
            for row in slice {
                let quantity = (per_name / row.close / lot).floor() * lot;
                if quantity > 0.0 {
                    target.insert(row.symbol, sign * quantity);
                }
            }
        };
        add_leg(&eligible[..count], long_gross, 1.0, &mut target);
        if self.cfg.mode == 2 {
            add_leg(
                &eligible[eligible.len() - count..],
                short_gross,
                -1.0,
                &mut target,
            );
        }
        target
    }

    fn create_orders(
        &mut self,
        signal_day: i32,
        execute_day: i32,
        rows: &[Row],
        close_nlv: f64,
    ) -> u64 {
        let targets = self.targets(rows, close_nlv);
        let row_map: HashMap<i32, &Row> = rows.iter().map(|r| (r.symbol, r)).collect();
        let mut symbols: Vec<i32> = self
            .positions
            .keys()
            .chain(targets.keys())
            .copied()
            .collect();
        symbols.sort_unstable();
        symbols.dedup();
        let mut created = 0;
        for symbol in symbols {
            let target = *targets.get(&symbol).unwrap_or(&0.0);
            let current = self
                .positions
                .get(&symbol)
                .map(|p| p.quantity)
                .unwrap_or(0.0);
            if (target - current).abs() < 1e-8 {
                continue;
            }
            let reference = row_map
                .get(&symbol)
                .map(|r| r.close)
                .unwrap_or_else(|| *self.last_close.get(&symbol).unwrap_or(&0.0));
            let trade_notional = (target - current).abs() * reference;
            let target_notional = target.abs() * reference;
            if trade_notional < self.cfg.min_trade_notional {
                continue;
            }
            if target_notional > 0.0
                && trade_notional / target_notional < self.cfg.rebalance_buffer_pct
            {
                continue;
            }
            self.order_seq += 1;
            self.pending.push(Order {
                seq: self.order_seq,
                signal_day,
                execute_day,
                symbol,
                target,
            });
            created += 1;
        }
        created
    }

    fn run_session(&mut self, rows: &[Row], next_day: Option<i32>, session_index: usize) {
        let day = rows[0].day;
        let market: HashMap<i32, Row> = rows.iter().map(|r| (r.symbol, *r)).collect();
        self.apply_adjustments_and_settle(&market);
        let fill_start = self.fill_seq;
        let notional_before: f64 = self.trades[..self.trade_count]
            .iter()
            .map(|t| t.filled_quantity * t.fill_price)
            .sum();
        let mut due: Vec<Order> = self
            .pending
            .iter()
            .filter(|o| o.execute_day == day)
            .cloned()
            .collect();
        self.pending.retain(|o| o.execute_day != day);
        if !due.is_empty() {
            let mut projected: BTreeMap<i32, f64> = self
                .positions
                .iter()
                .map(|(&s, p)| (s, p.quantity))
                .collect();
            for order in &due {
                projected.insert(order.symbol, order.target);
            }
            let (open_nlv, _, _) = self.nlv(&market, true);
            let projected_gross: f64 = projected
                .iter()
                .map(|(&s, &q)| q.abs() * self.price(s, &market, true))
                .sum();
            let (lg, sg) = self.effective_targets();
            let allowed = open_nlv.max(0.0)
                * self.cfg.max_gross_leverage.min(if lg + sg > 0.0 {
                    lg + sg
                } else {
                    self.cfg.max_gross_leverage
                });
            if projected_gross > allowed + 1e-8 && projected_gross > 0.0 {
                let scale = allowed / projected_gross;
                let lot = self.lot();
                for order in &mut due {
                    let sign = if order.target >= 0.0 { 1.0 } else { -1.0 };
                    order.target = sign * (order.target.abs() * scale / lot).floor() * lot;
                }
            }
        }
        due.sort_by(|a, b| {
            let ac = self
                .positions
                .get(&a.symbol)
                .map(|p| p.quantity)
                .unwrap_or(0.0);
            let bc = self
                .positions
                .get(&b.symbol)
                .map(|p| p.quantity)
                .unwrap_or(0.0);
            let ai = a.target.abs() > ac.abs() + 1e-8;
            let bi = b.target.abs() > bc.abs() + 1e-8;
            ai.cmp(&bi).then(a.seq.cmp(&b.seq))
        });
        for order in &due {
            self.execute(order, day, &market);
        }
        self.enforce_open_leverage(day, next_day, &market);

        let (before_financing, long_value, short_value) = self.nlv(&market, false);
        let borrow = if self.cfg.mode == 2 && short_value < 0.0 {
            short_value.abs() * self.cfg.borrow_cost_bps_annual / 10_000.0 / 252.0
        } else {
            0.0
        };
        if borrow > 0.0 {
            self.cash -= borrow;
            self.cumulative_borrow += borrow;
        }
        let margin = if self.cash < 0.0 && self.cfg.account_type == 2 {
            self.cash.abs() * self.cfg.margin_interest_bps_annual / 10_000.0 / 252.0
        } else {
            0.0
        };
        if margin > 0.0 {
            self.cash -= margin;
            self.cumulative_margin += margin;
        }
        let close_nlv = before_financing - borrow - margin;
        let daily_return = if self.previous_close_nlv > 0.0 {
            close_nlv / self.previous_close_nlv - 1.0
        } else {
            0.0
        };
        let traded_notional: f64 = self.trades[..self.trade_count]
            .iter()
            .map(|t| t.filled_quantity * t.fill_price)
            .sum::<f64>()
            - notional_before;
        let gross = if close_nlv > 0.0 {
            (long_value + short_value.abs()) / close_nlv
        } else {
            0.0
        };
        let net = if close_nlv != 0.0 {
            (long_value + short_value) / close_nlv
        } else {
            0.0
        };
        let created = if session_index % self.cfg.rebalance_every as usize == 0
            && next_day.is_some()
            && close_nlv > 0.0
        {
            self.create_orders(day, next_day.unwrap(), rows, close_nlv)
        } else {
            0
        };
        for row in rows {
            if finite_positive(row.close) {
                self.last_close.insert(row.symbol, row.close);
            }
            if finite_positive(row.adjustment) {
                self.last_adjustment.insert(row.symbol, row.adjustment);
            }
        }
        if self.daily_count < self.daily.len() {
            self.daily[self.daily_count] = FFDailyV1 {
                day,
                close_nlv: round_to(close_nlv, 6),
                net_nav: round_to(close_nlv / self.cfg.initial_capital, 8),
                daily_return: round_to(daily_return, 8),
                cash: round_to(self.cash, 6),
                long_market_value: round_to(long_value, 6),
                short_market_value: round_to(short_value, 6),
                gross_exposure: round_to(gross, 6),
                net_exposure: round_to(net, 6),
                turnover: round_to(
                    if self.previous_close_nlv > 0.0 {
                        traded_notional / self.previous_close_nlv
                    } else {
                        0.0
                    },
                    6,
                ),
                borrow_fee: round_to(borrow, 6),
                margin_interest: round_to(margin, 6),
                fills: self.fill_seq - fill_start,
                orders_created: created,
                positions: self.positions.len() as u64,
            };
            self.daily_count += 1;
        }
        self.previous_close_nlv = close_nlv;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ff_abi_version() -> u32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn ff_version() -> *const c_char {
    c"factorfactory-rust-backtest/0.2.0".as_ptr()
}

/// Run one chronological event simulation.
///
/// # Safety
/// Every pointer must reference a readable/writable buffer of the declared
/// length for the duration of this synchronous call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ff_run_v1(
    columns: *const FFColumnsV1,
    config: *const FFConfigV1,
    trades: *mut FFTradeV1,
    trade_capacity: usize,
    daily: *mut FFDailyV1,
    daily_capacity: usize,
    summary: *mut FFSummaryV1,
) -> i32 {
    if columns.is_null() || config.is_null() || summary.is_null() {
        return ERR_INPUT;
    }
    let cols = unsafe { &*columns };
    let cfg = unsafe { *config };
    if cols.abi_version != ABI_VERSION || cfg.abi_version != ABI_VERSION {
        return ERR_ABI;
    }
    if cols.row_count == 0
        || cols.session_count == 0
        || daily_capacity < cols.session_count
        || trade_capacity < cols.row_count
    {
        return ERR_CAPACITY;
    }
    macro_rules! slice {
        ($ptr:expr, $len:expr) => {{
            if $ptr.is_null() {
                return ERR_INPUT;
            }
            unsafe { std::slice::from_raw_parts($ptr, $len) }
        }};
    }
    let offsets = slice!(cols.session_offsets, cols.session_count + 1);
    let days = slice!(cols.day, cols.row_count);
    let symbols = slice!(cols.symbol, cols.row_count);
    let ranks = slice!(cols.univ_rank, cols.row_count);
    let factors = slice!(cols.factor, cols.row_count);
    let opens = slice!(cols.raw_open, cols.row_count);
    let closes = slice!(cols.raw_close, cols.row_count);
    let volumes = slice!(cols.volume, cols.row_count);
    let adv = slice!(cols.adv20_prev, cols.row_count);
    let atr = slice!(cols.atr_pct, cols.row_count);
    let vol20 = slice!(cols.vol20_prev, cols.row_count);
    let adjustment = slice!(cols.adjustment_factor, cols.row_count);
    let can_buy = slice!(cols.can_buy, cols.row_count);
    let can_sell = slice!(cols.can_sell, cols.row_count);
    let out_trades = unsafe { std::slice::from_raw_parts_mut(trades, trade_capacity) };
    let out_daily = unsafe { std::slice::from_raw_parts_mut(daily, daily_capacity) };
    let mut rows = Vec::with_capacity(cols.row_count);
    for i in 0..cols.row_count {
        rows.push(Row {
            index: i,
            day: days[i],
            symbol: symbols[i],
            rank: ranks[i],
            factor: factors[i],
            open: opens[i],
            close: closes[i],
            volume: volumes[i],
            adv: adv[i],
            atr_pct: atr[i],
            vol20: vol20[i],
            adjustment: adjustment[i],
            can_buy: can_buy[i] != 0,
            can_sell: can_sell[i] != 0,
        });
    }
    let mut engine = Engine {
        cfg,
        trades: out_trades,
        daily: out_daily,
        trade_count: 0,
        daily_count: 0,
        cash: cfg.initial_capital,
        positions: BTreeMap::new(),
        last_close: HashMap::new(),
        last_adjustment: HashMap::new(),
        pending: Vec::new(),
        order_seq: 0,
        fill_seq: 0,
        rejected: 0,
        partial: 0,
        previous_close_nlv: cfg.initial_capital,
        cumulative_fees: 0.0,
        cumulative_slippage: 0.0,
        cumulative_borrow: 0.0,
        cumulative_margin: 0.0,
    };
    for session in 0..cols.session_count {
        let start = offsets[session];
        let end = offsets[session + 1];
        if start >= end || end > rows.len() {
            return ERR_INPUT;
        }
        let next = if session + 1 < cols.session_count {
            Some(rows[offsets[session + 1]].day)
        } else {
            None
        };
        engine.run_session(&rows[start..end], next, session);
    }
    unsafe {
        *summary = FFSummaryV1 {
            abi_version: ABI_VERSION,
            status: OK,
            trade_count: engine.trade_count,
            daily_count: engine.daily_count,
            order_count: engine.order_seq,
            rejected_orders: engine.rejected,
            partial_orders: engine.partial,
            final_nlv: engine.previous_close_nlv,
            cumulative_fees: engine.cumulative_fees,
            cumulative_slippage: engine.cumulative_slippage,
            cumulative_borrow: engine.cumulative_borrow,
            cumulative_margin_interest: engine.cumulative_margin,
        };
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn money_rounds_half_up_for_positive_fees() {
        assert_eq!(money(0.205), 0.21);
        assert_eq!(money(1.004), 1.00);
    }

    #[test]
    fn fee_boundaries_match_python_contract() {
        let cfg = FFConfigV1 {
            abi_version: ABI_VERSION,
            market: 2,
            mode: 1,
            direction: 1,
            account_type: 1,
            impact_model: 1,
            universe_n: 500,
            rebalance_every: 5,
            max_positions: 10000,
            initial_capital: 1e6,
            top_fraction: 0.2,
            slippage_bps: 0.0,
            spread_bps: 0.0,
            max_volume_participation: 0.1,
            cash_buffer_fraction: 0.0,
            max_gross_leverage: 2.0,
            max_position_weight: 1.0,
            min_trade_notional: 0.0,
            rebalance_buffer_pct: 0.0,
            long_gross_target: 1.0,
            short_gross_target: 0.0,
            borrow_cost_bps_annual: 0.0,
            margin_interest_bps_annual: 0.0,
            impact_coefficient_bps: 0.0,
        };
        assert_eq!(fees(cfg, 1, 100.0, 10.0, 19_724).3, 0.21);
        assert_eq!(fees(cfg, -1, 100.0, 10.0, 19_724).3, 0.71);
    }
}
