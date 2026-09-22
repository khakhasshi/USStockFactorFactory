"""Freeze the candidate cohort, preserve old audits, run mandatory V4.3 audits.

PYTHONPATH=backend .venv/bin/python scripts/reaudit_v42_candidates.py
Writes resumable progress.json and one full JSON + event artifact set per factor.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.audit_snapshot import frozen_panel
from app.config import DEFAULT_ENGINE_CONFIG, DIRECTION_POLICY_FIXED, evaluation_config, resolve_engine_tasks
from app.db import SessionLocal, engine
from app.models import Experiment, Factor, Setting, Trial
from app.eval.harness import AUDIT_REVISION, evaluate_full

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temp.replace(path)


async def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    async with SessionLocal() as session:
        factors = list((await session.scalars(select(Factor).where(Factor.evaluation_protocol == "v4.2").order_by(Factor.id))).all())
        experiments = {e.id: e for e in (await session.scalars(select(Experiment))).all()}
        setting = await session.get(Setting, "engine_config")
        all_trials = list((await session.scalars(select(Trial).order_by(Trial.id))).all())
    cohort_file = output / "cohort.json"
    if cohort_file.exists():
        cohort = json.loads(cohort_file.read_text())
        ids = set(cohort["factor_ids"])
        factors = [f for f in factors if f.id in ids]
    else:
        if args.limit:
            factors = factors[:args.limit]
        cohort = {"factor_ids": [f.id for f in factors], "count": len(factors),
                  "revision": AUDIT_REVISION, "frozen_at": datetime.now(timezone.utc).isoformat(),
                  "history_policy": "append_previous_validation_never_delete"}
        save(cohort_file, cohort)
        save(output / "before.json", [{"id": f.id, "public_metrics": f.public_metrics,
            "gate_metrics": f.gate_metrics, "validation_metrics": f.validation_metrics,
            "eligibility": f.eligibility, "research_meta": f.research_meta,
            "lifecycle_stage": f.lifecycle_stage, "evaluated_at": f.evaluated_at} for f in factors])
    progress = {"revision": AUDIT_REVISION, "total": len(factors), "completed": 0, "errors": 0,
                "started_at": datetime.now(timezone.utc).isoformat(), "rows": []}
    groups = {}
    for f in factors:
        cfg = experiments[f.experiment_id].research_config or {}
        groups.setdefault((cfg.get("market", "us"), cfg.get("panel_glob")), []).append(f)
    for (market, panel_glob), group in groups.items():
        with frozen_panel(panel_glob, market) as snapshot:
            save(output / f"snapshot-{market}.json", snapshot.manifest)
            for f in group:
                started = time.monotonic()
                progress["current_factor"] = f.id
                save(output / "progress.json", progress)
                artifact = output / str(f.id)
                artifact.mkdir(exist_ok=True)
                cfg = experiments[f.experiment_id].research_config or {}
                engine_cfg = {**DEFAULT_ENGINE_CONFIG, **(setting.value if setting else {}), **(cfg.get("engine_config") or {})}
                tasks = resolve_engine_tasks(engine_cfg["tasks"], market, cfg.get("portfolio_mode", "long_only" if market == "ashare" else "long_short"), cfg.get("direction", 1), preserve_declared_costs=True)
                task = next((t for t in tasks if t.get("name") == f.task_name), None)
                if task is None:
                    # Missing historical task contract must not silently default.
                    raise ValueError(f"Factor {f.id}: historical task {f.task_name!r} not found")
                history = [t for t in all_trials if t.experiment_id == f.experiment_id and t.task_name == f.task_name]
                overrides = evaluation_config(market, cfg.get("evaluation_config"))
                experiment_attempts = sum(t.experiment_id == f.experiment_id and (t.statistic or {}).get("evaluation_performed", True) is not False for t in all_trials)
                overrides["multiple_testing_trials"] = max(int(overrides["multiple_testing_trials"]), 2 * experiment_attempts, 1)
                direction = int((f.research_meta or {}).get("direction", task.get("direction", cfg.get("direction", 1))))
                try:
                    audit_path = artifact / "audit.json"
                    if args.resume and audit_path.exists():
                        audit = json.loads(audit_path.read_text())
                        if audit.get("audit_revision") != AUDIT_REVISION:
                            raise ValueError("resume audit revision mismatch")
                        prior_panel = (audit.get("audit_provenance") or {}).get("panel") or {}
                        if (prior_panel.get("snapshot_id") != snapshot.manifest["snapshot_id"]
                                or (prior_panel.get("code") or {}).get("code_sha256") != snapshot.manifest["code"]["code_sha256"]):
                            raise ValueError("resume input/code snapshot mismatch; use a new output cohort")
                    else:
                        audit = evaluate_full(f.expression, int(task["universe_n"]), int(task["horizon"]),
                            cfg.get("portfolio_mode", "long_only" if market == "ashare" else "long_short"),
                            direction, panel_glob, task.get("cost_bps"), market, overrides, DIRECTION_POLICY_FIXED,
                            trial_history=history, audit_artifact_dir=artifact / "event")
                        save(audit_path, audit)
                    async with SessionLocal() as session:
                        current = await session.get(Factor, f.id)
                        meta = dict(current.research_meta or {})
                        audits = list(meta.get("audit_history") or [])
                        if current.validation_metrics != audit:
                            audits.append({"evaluation_protocol": current.evaluation_protocol,
                                "evaluated_at": str(current.evaluated_at), "eligibility": current.eligibility,
                                "validation": current.validation_metrics, "lifecycle_stage": current.lifecycle_stage})
                        meta.update(audit_history=audits, last_audit_revision=AUDIT_REVISION,
                            last_audit_artifact=str(audit_path), direction=direction)
                        current.research_meta = meta
                        # Holdout/event/rating NEVER overwrite mining feedback.
                        current.public_metrics = {**audit["public"], "discovery": audit["discovery"], "protocol_version": "v4.2"}
                        current.gate_metrics = audit["gate"]
                        current.validation_metrics = audit
                        current.eligibility = audit["eligibility"]
                        current.lifecycle_stage = audit["eligibility"]["stage"]
                        current.provenance_status = "immutable_v43_event_reaudited"
                        current.evaluated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        session.add(Trial(experiment_id=f.experiment_id, expression_hash=(f.fingerprint or {}).get("expr_hash", "audit"),
                            layer="FULL_AUDIT_V4", task_name=f.task_name, node_id=f.node_id, expression=f.expression,
                            search_method="frozen_v43_reaudit", selected=bool(audit["eligibility"]["eligible"]),
                            failure_reason="; ".join(audit["eligibility"]["failure_reasons"]),
                            statistic={"raw_return_evidence": audit.get("raw_return_evidence"), "audit_revision": AUDIT_REVISION,
                                       "input_snapshot_id": audit["input_snapshot_id"]}))
                        await session.commit()
                    row = {"factor_id": f.id, "market": market, "horizon": task["horizon"], "direction": direction,
                        "grade": audit["eligibility"]["grade"], "eligibility": audit["eligibility"],
                        "rating": audit["rating"]["net"], "event": {k: {"status": v["status"], "stats": v.get("stats"), "failure_reasons": v.get("failure_reasons")} for k, v in audit["event_audit"].get("windows", {}).items()},
                        "overfit": audit["overfit_governance"]["status"], "audit_path": str(audit_path)}
                except Exception as exc:
                    progress["errors"] += 1
                    row = {"factor_id": f.id, "error": type(exc).__name__ + ": " + str(exc)}
                    save(artifact / "error.json", row)
                row["seconds"] = round(time.monotonic() - started, 2)
                progress["rows"].append(row)
                progress["completed"] += 1
                save(output / "progress.json", progress)
                print(json.dumps({"completed": progress["completed"], "total": len(factors), **{k: row[k] for k in ("factor_id", "seconds", "error", "grade", "overfit") if k in row}}, ensure_ascii=False), flush=True)
    progress["finished_at"] = datetime.now(timezone.utc).isoformat()
    save(output / "progress.json", progress)
    lines = ["# V4.3 完整重审", "", f"版本：{AUDIT_REVISION}。冻结候选 {len(factors)} 个；完成 {progress['completed']}，执行错误 {progress['errors']}。",
        "", "历史结果保存在 before.json 和数据库 audit_history；所有结果均为研究证据，不是生产交易批准。",
        "现有试验没有完整逐次原始收益的，DSR/PBO 标记 INSUFFICIENT_DATA，不伪造历史回填。",
        "", "|因子|h|方向|训练|HOLDOUT|Vault|事件门槛|DSR/PBO|事件评级 CAGR|事件 Sharpe|事件 MDD|", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in progress["rows"]:
        if "error" in row:
            lines.append(f"|{row['factor_id']}|—|—|ERROR: {row['error']}||||||||")
            continue
        e = row["eligibility"]
        stats = (row.get("event", {}).get("rating", {}).get("stats") or {})
        pct = lambda key: f"{100 * stats[key]:.2f}%" if stats.get(key) is not None else "—"
        lines.append(f"|[{row['factor_id']}]({row['factor_id']}/audit.json)|{row['horizon']}|{row['direction']:+d}|{e['research_pass']}|{e['holdout_pass']}|{e['vault_pass']}|{e['event_pass']}|{row['overfit']}|{pct('ann_ret')}|{stats.get('sharpe','—')}|{pct('max_dd')}|")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "var/reports/v43-reaudit-20260913"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(run(parser.parse_args()))
