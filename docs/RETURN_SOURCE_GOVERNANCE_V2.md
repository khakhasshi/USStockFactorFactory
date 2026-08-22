# Training Return Source Governance V2

## Objective

Mechanism labels and expression structure are useful search controls, but they do not prove that two factors earn money from different paths. V2 adds a training-safe return-path layer to the existing controls while preserving the current V4 evaluator and sealed-data boundary.

The governance layer uses only compact return-path signatures already produced from `INNER_PUBLIC + META_TRAIN`. It must never consume `META_HOLDOUT`, `FACTOR_VAULT`, a full-window diagnostic return series, or a live portfolio result.

## Decision rules

- Online admission and meta optimization use positive return-path correlation `>= 0.85` inside the same comparable task scope.
- Full-library P0 diagnostics retain `0.80` as a stricter stress threshold. That report is non-PIT and must not train the Miner or Meta-Agent.
- Clustering is quality-ordered and representative-based. It does not use transitive connected components, so an A-B-C correlation chain cannot collapse A and C when they are not similar to the same representative.
- A strongly negative correlation is a diversifier after direction is frozen. It is not rejected as a duplicate.
- Structural similarity remains a separate admission gate. When V2 is enabled, structural duplicates can be rejected across experiments within the same market and portfolio mode.
- A public-pass candidate rejected by library admission keeps its original evaluator score and pass flag, but the rejection reason is added to its feedback envelope so the next Miner attempt can learn from it.

## Experiment configuration

Governance is opt-in, so an already-running V4 experiment keeps its existing semantics. Enable it only on a stopped experiment or, preferably, a new experiment:

```json
{
  "return_source_governance": {
    "protocol": "training_return_source_governance_v2",
    "correlation_threshold": 0.85,
    "meta_score_weight": 0.30,
    "required_sources": 4,
    "cross_experiment_admission": true
  }
}
```

`meta_score_weight` is limited to `[0, 0.5]`. A value of `0.30` preserves 70% of the legacy task/mechanism score and assigns 30% to the quality of distinct training return sources. Missing source slots are scored as zero, preventing many syntax variants of one path from filling the quota.

## Auditable catalog

Build a current-protocol, read-only catalog from stored factor metadata:

```bash
PYTHONPATH=backend POLARS_MAX_THREADS=2 python3 backend/scripts/build_training_return_source_catalog.py
```

The command creates a fresh directory under `var/reports/` containing:

- `summary.json`: coverage, cluster, redundancy, HHI/effective-source metrics, and recommended config.
- `assignments.csv`: every factor-to-representative assignment and correlation.
- `README.md`: a compact human-readable cluster report.

The command refuses to overwrite a non-empty explicit output directory. Generated catalog artifacts are governance diagnostics, not promotion or trading evidence.

## Weekend rollout sequence

1. Let the active V4 experiment finish or pause it normally; do not hot-reload its worker.
2. Freeze its result and generate the 0.85 training-safe catalog.
3. Create a new experiment with the V2 configuration above.
4. Run the same seed/task budget as the incumbent.
5. Compare evaluator pass rate, gate score, scoped redundancy, effective return sources, mechanism coverage, and rejection reasons.
6. Promote the optimizer only if quality and hard gates are non-degrading while distinct training return sources improve. Otherwise keep the incumbent and retain the audit artifacts.
