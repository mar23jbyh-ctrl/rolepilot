"""Read-only current source inventory. Never archives, moves or deletes files."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TREES = {'.git', '.venv', 'node_modules', 'data', 'uploads', 'dist', '.codex'}
CACHE_TREES = {'.mypy_cache', '.ruff_cache', '.pytest_cache', '__pycache__'}


def build_inventory():
    entries = []
    for base, dirs, files in os.walk(ROOT):
        for name in list(dirs):
            path = Path(base) / name
            relative = path.relative_to(ROOT).as_posix()
            if name in LOCAL_TREES or name in CACHE_TREES or name == 'evidence':
                classification = ('DELETE_CACHE' if name in CACHE_TREES else
                                  'RETAIN_EVIDENCE' if name == 'evidence' else 'LOCAL_ONLY')
                entries.append({'path': relative, 'class': classification, 'directory': True})
                dirs.remove(name)
        for name in files:
            path = Path(base) / name
            relative = path.relative_to(ROOT).as_posix()
            local = name.startswith('.env') and name != '.env.example'
            entries.append({'path': relative, 'class': 'LOCAL_ONLY' if local else 'KEEP',
                            'directory': False, 'bytes': path.stat().st_size})
    entries.sort(key=lambda row: row['path'])
    return {'schema_version': 'source-inventory-v1',
            'scope': 'Current layout only; no archive instruction or file-content export',
            'files': entries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional NEW JSON under docs/release; no overwrite')
    args = parser.parse_args(argv)
    inventory = build_inventory()
    if args.output:
        output = (ROOT / args.output).resolve()
        if not output.is_relative_to((ROOT / 'docs/release').resolve()) or output.suffix != '.json':
            raise SystemExit('Output must be a JSON file inside docs/release')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(inventory, stream, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(inventory, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
