#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "var" / "audits" / "p0-return-source-diversity-v1"


def main() -> None:
    path = ROOT / "progress.json"
    if not path.exists():
        raise SystemExit(f"尚未产生进度文件: {path}")
    progress = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"phase={progress['phase']}  "
        f"overall={progress['completed']}/{progress['total']} "
        f"({progress['percent']:.1f}%)  "
        f"elapsed={progress['elapsed_seconds']:.0f}s"
    )
    for name, row in progress.get("reports", {}).items():
        print(
            f"  {name}: {row['completed']}/{row['total']} "
            f"({row['percent']:.1f}%)"
        )


if __name__ == "__main__":
    main()

