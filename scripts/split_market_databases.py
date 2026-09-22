"""Copy a stopped research database into two EMPTY market-specific databases.

Never deletes or updates the source; preserves IDs and byte-verifies COPY streams.
No trading tables, credentials from trading modules, or runtime jobs are copied.
Global research settings (including LLM providers) are copied without printing them.
Run only after making a restricted pg_dump backup and restoring the research schema
into new destination databases. A failed copy leaves an unpublished database; it
never switches services or overwrites an existing destination.
"""
import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

import asyncpg

TABLES = ["settings", "experiments", "miner_versions", "outer_steps", "nodes",
          "factors", "trials", "llm_call_audits", "backtests", "screener_runs",
          "combination_experiments", "engine_events"]


def predicate(table, market):
    assert table in TABLES and market in {"us", "ashare"}
    chosen = ("SELECT id FROM experiments WHERE "
              f"COALESCE(NULLIF(research_config->>'market',''),'us')='{market}'")
    if table == "settings":
        return "TRUE"
    if table == "experiments":
        return f"id IN ({chosen})"
    return f"experiment_id IN ({chosen})" + (
        " OR experiment_id IS NULL" if table in {"llm_call_audits", "engine_events"} else "")


async def main(args):
    assert len({args.source, args.us, args.ashare}) == 3
    if args.report.exists():
        raise RuntimeError("Refusing to overwrite an existing migration report")
    source = await asyncpg.connect(database=args.source, host="localhost")
    report = {"source_database": args.source, "source_untouched": True, "markets": {}}
    try:
        async with source.transaction(isolation="repeatable_read", readonly=True):
            # Names are intentionally explicit: unknown new tables need review.
            tables = await source.fetch("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            unknown = {r[0] for r in tables} - set(TABLES)
            if any(not name.startswith("live_trading_") for name in unknown):
                raise RuntimeError(f"Unclassified source tables: {sorted(unknown)}")
            for market, database, active in [("us", args.us, 58), ("ashare", args.ashare, 57)]:
                target = await asyncpg.connect(database=database, host="localhost")
                result = {"database": database, "tables": {}, "active_experiment_id": active}
                try:
                    async with target.transaction():
                        for table in TABLES:
                            if await target.fetchval(f'SELECT count(*) FROM "{table}"'):
                                raise RuntimeError(f"Destination is not empty: {database}.{table}")
                        for table in TABLES:
                            key = "key" if table == "settings" else "id"
                            query = f'SELECT * FROM "{table}" WHERE {predicate(table, market)} ORDER BY "{key}"'
                            digest = hashlib.sha256()
                            with tempfile.TemporaryFile() as stream:
                                async def write(chunk):
                                    digest.update(chunk)
                                    stream.write(chunk)
                                exported = await source.copy_from_query(query, output=write, format="binary")
                                stream.seek(0)
                                await target.copy_to_table(table, source=stream, format="binary")
                            copied = hashlib.sha256()
                            async def verify(chunk):
                                copied.update(chunk)
                            await target.copy_from_query(f'SELECT * FROM "{table}" ORDER BY "{key}"',
                                                         output=verify, format="binary")
                            if copied.hexdigest() != digest.hexdigest():
                                raise RuntimeError(f"COPY content mismatch: {market}.{table}")
                            count = int(exported.split()[-1])
                            result["tables"][table] = {"rows": count, "sha256": digest.hexdigest()}
                            if table != "settings":
                                seq = await target.fetchval("SELECT pg_get_serial_sequence($1,'id')", table)
                                if seq:
                                    source_max = await source.fetchval(f'SELECT COALESCE(MAX(id),1) FROM "{table}"')
                                    await target.execute("SELECT setval($1::regclass,$2,true)", seq, source_max)
                            print(f"{market}: {table} {count} rows verified", flush=True)
                        if not await target.fetchval("SELECT EXISTS(SELECT 1 FROM experiments WHERE id=$1)", active):
                            raise RuntimeError("Expected active experiment absent")
                        await target.execute("UPDATE settings SET value=$1::json WHERE key='active_experiment'",
                                             json.dumps({"id": active}))
                        result["backtest_ids"] = [r[0] for r in await target.fetch("SELECT id FROM backtests ORDER BY id")]
                        result["experiment_ids"] = [r[0] for r in await target.fetch("SELECT id FROM experiments ORDER BY id")]
                    report["markets"][market] = result
                finally:
                    await target.close()
    finally:
        await source.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        json.dump(report, handle, indent=2)
    args.report.chmod(0o600)
    print(f"Verified report: {args.report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--us", required=True)
    parser.add_argument("--ashare", required=True)
    parser.add_argument("--report", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
