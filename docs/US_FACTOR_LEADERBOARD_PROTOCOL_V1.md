# Cross-task US long-short factor leaderboard protocol v1

## Purpose

This protocol compares historical factor expressions under one frozen US
equity long-short event backtest. It is a reproducible research ranking, not a
production-trading approval.

Protocol identifier:

`us_cross_task_long_short_factor_leaderboard_v1`

## Frozen candidate set

The database cutoffs for experiments, nodes, and factors are supplied on the
command line and written to `snapshot.json`.

The default US source policy is
`us_plus_ashare_price_volume`:

1. include every AST-unique expression referenced by a US research task;
2. also include A-share expressions whose referenced fields are a subset of
   `open`, `high`, `low`, `close`, `vol`, and `amount`;
3. preserve every source reference and source market;
4. validate every selected expression against the US DSL field whitelist;
5. isolate invalid and non-portable expressions instead of silently dropping
   them.

The snapshot reports source-record counts, AST-unique counts, US/A-share
overlap, selected counts before validation, valid counts, invalid counts, and
excluded counts.

## Chronology and direction

- `META_TRAIN`: 2020-01-01 through 2022-12-31.
- `META_HOLDOUT`: 2023-01-01 through 2024-12-31.
- `FACTOR_VAULT`: 2025-01-01 through 2026-08-04.

The raw and negated orientation are compared on `META_TRAIN`. The sign with
the higher non-overlapping five-session RankIC is frozen before holdout is
opened. Pearson IC is only the exact-zero tie-break.

The leaderboard uses only `META_HOLDOUT`. Vault candidates are selected and
written to `finalists_frozen_before_vault.json` before Vault replay starts.
Vault results never alter the leaderboard score or rank.

## Portfolio and event engine

- Target market: US equities.
- Portfolio mode: long-short.
- Universe: rolling Top-500 by 60-session average dollar amount.
- Signal: session `t` close.
- Execution: raw open on session `t+1`.
- Rebalance and holding horizon: five sessions.
- Long leg: equal-weight top 20%, with total target value equal to one NAV.
- Short leg: equal-weight bottom 20%, with absolute target value equal to one
  NAV.
- Intended gross exposure: approximately 200%; intended net exposure:
  approximately 0%.
- Initial capital: USD 1,000,000.
- Maximum participation: 5% of observed daily share volume.
- US share lot: one share.

Every fill, cash movement, position change, financing charge, NAV point, and
summary statistic comes from the same chronological event ledger.

## Costs

Every 0/5/15 BPS scenario always includes:

- IBKR Pro Fixed US-equity commission: USD 0.005 per share, USD 1 minimum,
  capped at 1% of trade value;
- a constant 300 BPS annual short-borrow pressure proxy, accrued daily on
  short market value.

The named 0/5/15 BPS values are additional adverse slippage on every buy and
sell. They do not replace commissions or borrow financing.

The borrow input is not historical security-level locate or hard-to-borrow
data. It cannot establish production shortability.

## Metrics and ranking

The report contains:

- annualized return, Sharpe, maximum drawdown, turnover, fill rate, execution
  costs, borrow costs, and gross/net exposure;
- non-overlapping Pearson IC, ICIR, RankIC, RankICIR, hit rates, normal
  approximations, and Benjamini-Hochberg q-values;
- 0/5/15 BPS cost monotonicity;
- economic-equivalence grouping by identical frozen evaluation fingerprints.

The robust score is a transparent cross-sectional percentile composite:

- 15 BPS annualized return: 20%;
- 15 BPS Sharpe: 25%;
- Pearson IC: 10%;
- Pearson ICIR: 15%;
- RankIC: 10%;
- RankICIR: 15%;
- cost resilience: 5%.

`practical_pass` additionally requires all three ledger-integrity checks,
monotonic deterioration as slippage increases, positive 15 BPS return and
Sharpe, positive IC and RankIC, and RankIC BH q at or below 0.10.

## Research boundary

The local US panel is a non-PIT current-constituent research panel. It does not
provide historical constituent membership, point-in-time delistings, daily
locates, hard-to-borrow fees, recalls, or forced buy-ins. Corporate actions are
handled through an adjustment-factor exposure-continuity proxy.

All outputs therefore retain `NON_PIT_RESEARCH` and
`production_eligible=false`.

## Artifacts

The batch writes the frozen snapshot, protocol and source hashes, resumable
JSONL results, full CSV/Parquet/JSON leaderboards, dimension rankings, invalid
and excluded-expression inventories, Vault results, finalist settlement
statements, progress, and a final manifest. The HTML renderer creates one
self-contained offline report and a derivative hash manifest.
