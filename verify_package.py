#!/usr/bin/env python3
"""Verify release files and the losslessly compressed benchmark datasets."""
import json
from pathlib import Path
from prepare_data import digest, prepare

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, checksum in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f'Missing or changed release file: {name}')
    prepare(verify_only=True)
    print(f"PASS package integrity: {len(manifest['files'])} files; all four original CSV hashes match")


if __name__ == '__main__':
    verify()
