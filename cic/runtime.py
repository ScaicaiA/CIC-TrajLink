"""Run isolation and completed-stage validation for the command-line driver."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from .io import read_json, write_json, sha256
from .protocol import EXPECTED_SEEDS, training_config


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    if config.get('experiment_kind') not in ('formal', 'engineering_smoke'):
        raise ValueError('Unknown experiment kind')
    for seed in EXPECTED_SEEDS:
        training_config(config, seed)
    for city, spec in config['cities'].items():
        if set(spec['fixed_k']) != {str(s) for s in EXPECTED_SEEDS}:
            raise ValueError(f'{city}: all three fixed budget pairs are required')
        for pair in spec['fixed_k'].values():
            if set(pair) != {'kp', 'kc'} or any(type(v) is not int or v < 0 for v in pair.values()):
                raise ValueError(f'{city}: budgets must be nonnegative integers')
    if config['experiment_kind'] == 'formal':
        reference = read_json(Path(__file__).resolve().parents[1] / 'config/four_cities.json')
        if config != reference:
            raise ValueError('Formal runs require the bundled configuration; use a separate engineering_smoke config for tests')
        if config['training']['encoder']['epochs'] != 30 or config['training']['mlp']['epochs'] != 20:
            raise ValueError('Formal runs require 30 encoder and 20 MLP epochs')
    return config


def dependency_versions():
    names = ('torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'threadpoolctl', 'psutil')
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


@contextmanager
def run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is already using this run: {path.parent}') from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + '\n')
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def binding(meta, city, seed, stage):
    return dict(run_signature=meta['run_signature'], city=city, training_seed=seed, stage=stage)


def committed(folder, expected):
    folder = Path(folder)
    marker = folder / '_complete.json'
    if not marker.exists():
        return False
    saved = read_json(marker)
    if saved['binding'] != expected:
        raise ValueError(f'Completed stage belongs to a different configuration: {folder}')
    if not saved['files']:
        raise ValueError(f'Empty completion manifest: {folder}')
    for name, checksum in saved['files'].items():
        file = folder / name
        if not file.is_file() or sha256(file) != checksum:
            raise ValueError(f'Completed stage has a missing or changed file: {file}')
    return True


def commit(folder, expected):
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and p.name != '_complete.json'}
    write_json(folder / '_complete.json', dict(binding=expected, files=files))


def preserve_incomplete(folder, run):
    folder, run = Path(folder), Path(run)
    if folder.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        saved = run / 'incomplete_attempts' / stamp / folder.relative_to(run)
        saved.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(saved)
        print(f'Preserved incomplete stage: {saved}', flush=True)
