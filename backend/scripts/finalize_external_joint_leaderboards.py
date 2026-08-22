"""Wait for batch reports and render their self-contained leaderboard HTML."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.render_factor_leaderboard_html import render_report  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, action="append", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.poll_seconds < 2:
        raise SystemExit("--poll-seconds must be at least 2")
    pending = {directory.resolve() for directory in args.report_dir}
    while pending:
        for directory in list(pending):
            html = directory / "leaderboard.html"
            if html.is_file():
                pending.remove(directory)
                continue
            progress_path = directory / "progress.json"
            manifest_path = directory / "manifest.json"
            if not progress_path.is_file() or not manifest_path.is_file():
                continue
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if progress.get("phase") != "complete":
                continue
            rendered = render_report(directory)
            print(
                f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                f"rendered {rendered['embedded_results']} rows · {html}",
                flush=True,
            )
            pending.remove(directory)
        if pending:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
