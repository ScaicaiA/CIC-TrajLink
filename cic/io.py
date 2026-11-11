"""Small portable I/O helpers; no implicit discovery of historical runs."""
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import random
import threading

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_strings(values):
    return hashlib.sha256('\n'.join(map(str, values)).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve(base, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base) / p).resolve()


def log(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields), ensure_ascii=False, allow_nan=False), flush=True)


def seed_everything(seed, threads=4):
    # No claimed bitwise equivalence across PyTorch versions or devices.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


@contextmanager
def heartbeat(stage, interval=15):
    """Liveness message, never an invented completion percentage."""
    stopped = threading.Event()
    begin = perf_counter()

    def tick():
        while not stopped.wait(interval):
            log(stage, event='heartbeat', elapsed_seconds=round(perf_counter() - begin, 2))

    worker = threading.Thread(target=tick, daemon=True)
    log(stage, event='start')
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()
        log(stage, event='end', elapsed_seconds=round(perf_counter() - begin, 2))


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    paths = (list(root.glob('*.py')) + list((root / 'cic').glob('*.py')) +
             list((root / 'tools').glob('*.py')) + list(root.glob('*.sh')))
    for p in sorted(paths):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()
