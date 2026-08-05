# Step-event backtest protocol v1

`step_event_v1` is the only protocol used by new backtests. Historical vector
records remain readable and are labelled `legacy_vector`.

## Chronology

For every exchange session the engine advances through these phases:

1. `SESSION_OPEN`
2. `OPEN_CORPORATE_ACTION`
3. `OPEN_EXECUTION`
4. `CLOSE_FINANCING`
5. `CLOSE_SIGNAL`
6. `SESSION_CLOSE`

A DSL value is observed after session `t` closes. The resulting DAY target
orders can execute only at the raw open of the next available panel session.
There is no same-session fill path. Orders that encounter missing prices,
execution eligibility blocks, insufficient A-share cash, or the volume
participation cap are rejected or partially filled and are not silently carried
forward.

The ledger uses raw prices for fills and closing valuation. Adjustment-factor
changes alter share quantity before the open so that economic exposure remains
continuous. Because the current panel does not distinguish splits from cash
dividends, this is explicitly labelled
`split_dividend_reinvestment_proxy`; it is a research approximation rather
than a broker-grade corporate-action ledger.

## Fee profiles

### A-share: `ashare_wan2_no_min_v1`

- Broker commission: `notional × 0.0002`, with no CNY 5 minimum.
- Stamp duty: seller only; 0.10% through 2023-08-27 and 0.05% from
  2023-08-28.
- Transfer fee: both sides; 0.002% through 2022-04-28 and 0.001% from
  2022-04-29.
- Every component is rounded to a cent with `ROUND_HALF_UP`.
- Buy quantities use 100-share lots. The A-share engine is long-only.

The 万2免5 commission is the task policy supplied by the operator. Tax and
transfer-fee source URLs are frozen into every backtest manifest.

### US: `ibkr_pro_fixed_us_v1`

- IBKR Pro Fixed: USD 0.005 per share.
- USD 1 minimum per order.
- 1% of trade value maximum.
- The fixed profile is treated as inclusive of exchange, clearing, and
  regulatory transaction charges; these are not deducted a second time.
- Short positions accrue the configured annual borrow-cost proxy each session.

The official IBKR source URL and the schedule version are frozen into every
backtest manifest.

## Settlement statement and artifacts

Each completed run writes an immutable run directory under `var/backtests`:

- `settlement_statement.parquet`
- `settlement_statement.csv`
- `event_ledger.parquet`
- `daily_ledger.parquet`
- `manifest.json`

Every file has a SHA-256 digest and row count in the manifest. Each settlement
row contains the order/fill identity, signal/scheduled/execution dates,
reference and fill prices, requested/filled/unfilled quantities, all fee
components, slippage, cash before/after, position before/after, and NLV after
the fill.

## Integrity gate

The run is marked `PASS` only when all statement checks pass:

- cash movement equals signed fill cash flow plus fees;
- fee formula and the sum of fee components reconcile;
- gross amount, slippage, and position movement independently recompute;
- signal date precedes execution and scheduled execution equals fill date;
- fill IDs are unique and fills have positive quantity and price;
- buy/sell signs match the position movement;
- every fill uses the frozen fee profile;
- A-share buys obey board lots;
- event phases are chronological;
- long-only runs never end with a negative position;
- NAV is generated from the same stateful ledger.

Regression tests also write and reread CSV/Parquet statements, compare row
counts and fee totals, and verify that deliberate statement tampering fails the
integrity gate.

## Scope boundary

Passing the integrity gate means the engine's timing and accounting are
internally reproducible. It does not cure non-PIT membership, approximate
corporate actions, missing historical bid/ask, US borrow availability, or
broker-specific taxes and account rules. A result remains
`NON_PIT_RESEARCH`, not live-trading approval.
