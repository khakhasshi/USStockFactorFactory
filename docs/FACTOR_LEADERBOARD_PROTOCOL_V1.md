# Cross-task A-share factor leaderboard protocol v1

## Scope and frozen inventory

The batch inventory is read from PostgreSQL with explicit maximum experiment,
node, and factor IDs. It includes:

- every factor-library row at or below the frozen factor ID;
- every successful or failed research-tree node at or below the frozen node ID;
- A-share, US, active, stopped, and historical experiments;
- every source reference for each expression.

Expressions are deduplicated by their parsed AST, not raw whitespace. Invalid
or mathematically degenerate DSL expressions are retained in
`invalid_expressions.csv`; they are never converted into zero-score results.
After the sign is frozen, rows with an identical high-precision training IC,
holdout IC, three-cost portfolio, fill, turnover, and cost fingerprint are
placed in one ranking-equivalence group. This catches root-level sign aliases
and monotonic wrappers that produce the same evaluated portfolio, while the
full table still preserves every original expression.

## Chronology

The protocol uses the existing immutable market layers:

1. `META_TRAIN`, 2020-01-01 through 2022-12-31: select `+1` or `-1` from
   Rank IC, using Pearson IC only as a zero-rank-IC tiebreak.
2. `META_HOLDOUT`, 2023-01-01 through 2024-12-31: calculate the primary
   leaderboard and all 0/5/15 BPS event backtests.
3. `FACTOR_VAULT`, 2025-01-01 through 2026-08-04: open only after the overall
   finalists and each published dimension champion have been frozen; Vault
   results never change rank or score.

IC observations use non-overlapping five-session cohorts. The final six
sessions of each layer are excluded so a layer's last signal cannot consume a
forward return from the next layer.

## Portfolio and execution

Every expression is replayed as the same A-share portfolio:

- DSL window and cross-sectional stages calculated on the complete panel,
  followed by the rolling top-500 liquidity filter;
- pure long-only, top 20% by the frozen oriented signal;
- CNY 10,000,000 initial capital;
- five-session rebalance cadence;
- signal after close on `t`, fill at raw open on `t+1`;
- 100-share buy lots and 5% maximum volume participation;
- suspension and open limit-state execution proxies;
- raw-price fills and closing valuation, with adjustment-factor share changes.

The three BPS cases are additional two-sided slippage assumptions. All cases
also charge the A-share schedule independently:

- broker commission at 0.02%, with no CNY 5 minimum;
- seller stamp duty using the historical effective-date schedule;
- two-sided transfer fee using the historical effective-date schedule.

## Metrics and ranking

For the oriented signal, the report publishes:

- event-ledger annualized return, Sharpe ratio, drawdown, turnover, fill rate,
  and execution cost at 0, 5, and 15 BPS;
- daily cross-sectional Pearson IC and annualized Pearson ICIR;
- daily cross-sectional Spearman Rank IC and annualized Rank ICIR.

The robust score is a cross-sectional percentile blend calculated only on
economic representatives:

| Dimension | Weight |
|---|---:|
| 15 BPS annualized return | 20% |
| 15 BPS Sharpe | 25% |
| Pearson IC | 10% |
| Pearson ICIR | 15% |
| Rank IC | 10% |
| Rank ICIR | 15% |
| 0-to-15 BPS cost resilience | 5% |

`practical_pass` additionally requires all three online ledger audits to pass,
monotonic degradation from 0 to 5 to 15 BPS, positive 15 BPS return and
Sharpe, positive Pearson IC and Rank IC, and a Rank IC
Benjamini-Hochberg q-value no greater than 0.10 across all
economic-equivalence groups. It is a research filter, not a production
approval.

## Parallel execution and audit

The CLI uses a bounded process pool. Each process loads the A-share panel once,
then reuses it for many expressions while Polars uses a bounded native thread
pool. Every completed expression is fsynced to append-only JSONL and progress
is atomically updated, allowing an interrupted run to resume without
recomputing completed expressions.

The compact path executes the same state machine but reconciles each transient
fill online instead of retaining tens of millions of rows. It checks cash,
fees, fee components, gross amount, slippage, position movement, signal/fill
chronology, fill identity, fee profile, A-share lots, event-phase order, and
long-only positions. Frozen finalists are replayed again with full settlement,
event, and daily ledgers; each artifact is hashed in its manifest.

## Offline HTML derivative

`render_factor_leaderboard_html.py` converts a completed report directory into
one self-contained `leaderboard.html`. It embeds all ranked and duplicate rows,
the invalid-expression appendix, frozen dimension champions, Vault results,
and finalist-ledger summaries. Search, filters, sorting, charts, details, and
filtered CSV export run locally with no CDN or other network dependency.

The sibling `leaderboard_manifest.json` records the HTML hash and hashes of
every source JSON used to render it. The HTML is a presentation derivative; it
does not change the frozen ranking, Vault selection, or source manifest.

## Interpretation boundary

The current A-share panel is a non-PIT current-constituent research panel and
corporate actions remain an adjustment-factor proxy. Passing this protocol
means the comparison is chronological, costed, reproducible, and internally
reconciled. It does not establish survivorship-safe or broker-production
eligibility, and every output therefore remains `NON_PIT_RESEARCH`.
