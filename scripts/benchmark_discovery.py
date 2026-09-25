"""Run a predeclared equal-budget comparison, without altering research tasks.

Example: .venv/bin/python scripts/benchmark_discovery.py --market us --budget 5
Results are research diagnostics, never factor admission or OOS calibration.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.audit_snapshot import frozen_panel, digest_json
from backend.app.config import get_dsl_fields
from backend.app.discovery_benchmark import compare_arms
from backend.app.eval.harness import evaluate
from backend.app.search_pool import propose_search_seed, ALGORITHM_GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=["us", "ashare"], required=True)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--arms", nargs="+", choices=sorted(ALGORITHM_GROUPS),
                        default=["structured_random", "grammar_enumerative", "evolutionary"])
    parser.add_argument("--universe", type=int, default=500)
    parser.add_argument("--horizon", type=int, choices=[1, 5, 10, 20], default=5)
    parser.add_argument("--mode", choices=["long_only", "long_short"], default="long_only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.budget <= 1000 or len(set(args.arms)) != len(args.arms):
        parser.error("budget must be 1..1000 and arms distinct")
    output = args.output or Path("var/reports/discovery-benchmark") / (args.market + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite/resume a different campaign.
    def save(name, payload):
        temporary = output / (name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False))
        os.replace(temporary, output / name)
    with frozen_panel(None, args.market) as snapshot:
        contract = {k: v for k, v in vars(args).items() if k != "output"}
        contract.update(snapshot=snapshot.manifest, automated_layers=["INNER_PUBLIC", "META_TRAIN"],
                        residual_and_qlib_artifacts="not_preloaded_fallbacks_explicitly_accounted")
        save("contract.json", contract)
        def propose(arm, feedback, rng):
            return propose_search_seed(family="momentum", fields=get_dsl_fields(args.market),
                feedback_nodes=feedback, algorithms=[arm], rng=rng)
        def measure(expression):
            return evaluate(expression, universe_n=args.universe, horizon=args.horizon,
                            portfolio_mode=args.mode, market=args.market)
        def progress(row):
            save("progress.json", {k: v for k, v in row.items() if k != "attempt"})
            save(f"attempt-{row['sequence']:06d}.json", row["attempt"])
            print(json.dumps({k: v for k, v in row.items() if k != "attempt"}), flush=True)
        result = compare_arms(args.arms, args.budget, args.seed, propose, measure, progress=progress)
        result["contract_sha256"] = digest_json(contract)
        save("report.json", result)
        save("progress.json", {"status": "done", "summary": result["summary"]})
        print(str(output.resolve()), flush=True)


if __name__ == "__main__":
    main()
