#!/usr/bin/env python3
"""Restore and verify the bundled CSV files without changing their bytes."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare(config_path=ROOT / 'config/four_cities.json', cities=None, verify_only=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    selected = list(config['cities']) if cities is None else cities
    for city in selected:
        spec = config['cities'][city]
        target = (config_path.parent / spec['data']).resolve()
        if 'archive' not in spec:
            if digest(target) != spec['data_sha256']:
                raise ValueError(f'{city}: CSV checksum mismatch')
            continue
        archive = (config_path.parent / spec['archive']).resolve()
        if digest(archive) != spec['archive_sha256']:
            raise ValueError(f'{city}: compressed data checksum mismatch')
        target.parent.mkdir(parents=True, exist_ok=True)
        with (target.parent / '.prepare.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists():
                if digest(target) != spec['data_sha256']:
                    raise ValueError(f'{city}: existing CSV differs; it was not overwritten: {target}')
                print(f'PASS {city}: CSV checksum', flush=True)
                continue
            fd, temporary = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=target.parent)
            try:
                actual = hashlib.sha256()
                with os.fdopen(fd, 'wb') as output, gzip.open(archive, 'rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        actual.update(block)
                        if not verify_only:
                            output.write(block)
                if actual.hexdigest() != spec['data_sha256']:
                    raise ValueError(f'{city}: uncompressed data checksum mismatch')
                if not verify_only:
                    os.replace(temporary, target)
                print(f'PASS {city}: {"archive contents" if verify_only else "restored CSV"}', flush=True)
            finally:
                Path(temporary).unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/four_cities.json')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    prepare(args.config, verify_only=args.verify_only)
