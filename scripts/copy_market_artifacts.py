"""Copy validated market-owned backtest artifacts using APFS copy-on-write.

Original manifests remain byte-identical, including historical source paths.
No symlink to a writable shared artifact directory and no deletions.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(2**20), b''):
            h.update(data)
    return h.hexdigest()


def main(args):
    manifest = json.loads(args.report.read_text())
    source = args.source.resolve()
    destinations = {"us": source / "var/backtests-us", "ashare": args.ashare / "var/backtests"}
    result = {}
    for market, root in destinations.items():
        root.mkdir(parents=True, exist_ok=True)
        copied, absent = [], []
        for backtest_id in manifest['markets'][market]['backtest_ids']:
            src = source / 'var/backtests' / f'{backtest_id:08d}'
            dest = root / src.name
            if not src.exists():
                absent.append(backtest_id)  # Old vector backtests had no file ledger.
                continue
            if dest.exists():
                raise RuntimeError(f'Refusing to overwrite {dest}')
            subprocess.run(['/bin/cp', '-cR', str(src), str(dest)], check=True)
            for original in src.rglob('*'):
                if original.is_file():
                    twin = dest / original.relative_to(src)
                    if original.stat().st_size != twin.stat().st_size or digest(original) != digest(twin):
                        raise RuntimeError(f'Artifact content mismatch: {twin}')
            copied.append(backtest_id)
        # Combination runs currently all belong to US; copy only that service.
        if market == 'us' and (source / 'var/backtests/combination_lab').exists():
            subprocess.run(['/bin/cp', '-cR', str(source / 'var/backtests/combination_lab'), str(root / 'combination_lab')], check=True)
        result[market] = {'copied_backtest_ids': copied, 'source_without_artifact_ids': absent,
                          'artifact_root': str(root)}
        print(market, 'copied', len(copied), 'historical rows without source files', len(absent), flush=True)
    output = args.report.with_name('verified-artifacts.json')
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
    output.chmod(0o600)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--ashare', type=Path, required=True)
    main(p.parse_args())
