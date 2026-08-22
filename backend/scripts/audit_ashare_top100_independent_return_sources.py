#!/usr/bin/env python3
"""Replay and cluster frozen Top-100 representatives from four A-share lists.

The four non-overlapping list definitions are:
* internal A-share holdout-event leaderboard;
* internal A-share full-window vector leaderboard;
* external joint leaderboard, A-share-library origin only;
* external joint leaderboard, US-library origin only.

All candidates keep their published direction and are replayed as A-share
long-only benchmark-active returns on the same 2020-to-latest window.  Rows
whose external provenance is ``both`` are deliberately excluded from the two
external lists so one source slot cannot enter both lists.
"""

from __future__ import annotations

from pathlib import Path

import audit_us_top100_independent_return_sources as audit


audit.PROTOCOL = "ashare_four_leaderboard_top100_return_source_audit_v1"
audit.POLICY_LABEL = "FULL_WINDOW_FIXED_DIRECTION_DIVERSITY_DIAGNOSTIC"
audit.REPORT_TITLE = "A股四榜前100独立收益来源审计"
audit.MARKET = "ashare"
audit.DEFAULT_REPORTS = {
    "内部样本外事件榜": "ashare-all-factor-leaderboard-corrected-snapshot-2208-1522",
    "内部全区间双向榜": "ashare-long-only-vector-screen-2020-latest-20260807-133714",
    "外部A股库": "external-joint-ashare-us-to-ashare-long-only-vector-2020-latest-20260821",
    "外部美股库": "external-joint-ashare-us-to-ashare-long-only-vector-2020-latest-20260821",
}
audit.MODE_BY_LIST = {list_id: "long_only" for list_id in audit.DEFAULT_REPORTS}
audit.LIST_FILTERS = {
    "外部A股库": {"origin_scope": "ashare"},
    "外部美股库": {"origin_scope": "us"},
}
audit.PANEL_PROTOCOL_LIST_IDS = ("外部A股库", "外部美股库")
audit.COMMON_START = "2020-01-01"
audit.COMMON_END = "2026-08-20"
audit.RUN_SCRIPT_PATH = Path(__file__).resolve()


if __name__ == "__main__":
    raise SystemExit(audit.main())
