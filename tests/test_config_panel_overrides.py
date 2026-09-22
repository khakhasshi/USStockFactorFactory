import json
import os
import subprocess
import sys


def test_market_panel_globs_can_be_overridden_independently():
    environment = dict(os.environ)
    environment.update(
        {
            "FF_US_PANEL_GLOB": "/srv/panels/us/trade_year=*/*.parquet",
            "FF_ASHARE_PANEL_GLOB": "/srv/panels/ashare/trade_year=*/*.parquet",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from backend.app.config import default_panel_glob; "
                "print(json.dumps([default_panel_glob('us'), "
                "default_panel_glob('ashare')]))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(result.stdout) == [
        "/srv/panels/us/trade_year=*/*.parquet",
        "/srv/panels/ashare/trade_year=*/*.parquet",
    ]
