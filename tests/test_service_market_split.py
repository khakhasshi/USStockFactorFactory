import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("market", ["us", "ashare"])
def test_independent_service_rejects_other_market(market):
    other = "us" if market == "ashare" else "ashare"
    code = f"""
import json
from backend.app.config import service_accepts_task, require_service_market
require_service_market({market!r})
try:
    require_service_market({other!r})
except ValueError:
    rejected = True
else:
    rejected = False
print(json.dumps([service_accepts_task({{'market': {market!r}}}),
                  service_accepts_task({{'market': {other!r}}}), rejected]))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, check=True,
                            env={**os.environ, "FF_SERVICE_MARKET": market,
                                 "FF_MARKET": market, "FF_DISABLE_FDC_DEFAULTS": "1"})
    assert json.loads(result.stdout) == [True, False, True]


def test_neutral_service_keeps_existing_compatibility():
    result = subprocess.run([sys.executable, "-c", "from backend.app.config import service_accepts_task; "
                             "assert service_accepts_task({'market': 'us'}); "
                             "assert service_accepts_task({'market': 'ashare'})"],
                            env={**os.environ, "FF_SERVICE_MARKET": "", "FF_SERVICE_ARCHITECTURE": ""},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
