# Cross-task US long-only factor leaderboard protocol v1

## Purpose

This protocol compares the same frozen cross-task factor expressions used by
the US long-short leaderboard under a US-equity long-only event backtest. It
measures executable top-tail portfolio performance, including market beta. It
is not a pure-alpha estimate or a production-trading approval.

Protocol identifier:

`us_cross_task_long_only_factor_leaderboard_v1`

## Frozen candidate set

The source policy is `us_plus_ashare_price_volume`:

1. include every AST-unique expression referenced by a US research task;
2. include A-share expressions using only `open`, `high`, `low`, `close`,
   `vol`, and `amount`;
3. preserve source provenance and isolate invalid or non-portable expressions;
4. freeze experiment, node, and factor cutoffs before the batch starts.

The intended comparison run uses the same database cutoffs and field policy as
the paired US long-short leaderboard.

## Chronology and direction

- `META_TRAIN`: 2020-01-01 through 2022-12-31.
- `META_HOLDOUT`: 2023-01-01 through 2024-12-31.
- `FACTOR_VAULT`: 2025-01-01 through 2026-08-04.

Raw and negated orientations are compared on `META_TRAIN`. The winning sign is
frozen before `META_HOLDOUT`. Leaderboard ranks use only `META_HOLDOUT`; Vault
candidates are frozen before Vault is opened, and Vault never changes rank.

## Portfolio and event engine

- Target market: US equities.
- Portfolio mode: long-only.
- Universe: rolling Top-500 by 60-session average dollar amount.
- Signal: session `t` close.
- Execution: raw open on session `t+1`.
- Rebalance and holding horizon: five sessions.
- Portfolio: equal-weight top 20% of the eligible cross-section.
- Intended gross and net exposure: approximately 100%.
- Initial capital: USD 1,000,000.
- Maximum participation: 5% of observed daily share volume.
- US share lot: one share.

Every fill, cash movement, position change, NAV point, and statistic comes from
the same chronological event ledger. Negative positions are an integrity
violation in this protocol.

## Costs

Every 0/5/15 BPS scenario includes IBKR Pro Fixed US-equity commission:

- USD 0.005 per share;
- USD 1 minimum per order;
- capped at 1% of trade value.

The named BPS values are additional adverse slippage on every buy and sell.
There is no short-borrow charge because the portfolio cannot hold short
positions.

## Interpretation boundary

Long-only returns combine stock-selection alpha, market beta, sector and style
exposures, and implementation effects. They must not be called pure alpha.

The local US panel is a non-PIT current-constituent research panel. It lacks
historical constituent membership and full delisting history. All outputs
therefore retain `NON_PIT_RESEARCH` and `production_eligible=false`.

The robust score, multiple-testing gate, economic-equivalence grouping,
full-ledger finalist replay, and artifact manifest follow the paired
long-short protocol.
