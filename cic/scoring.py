"""Build original 10D point relations once per supplied edge; no supplementation."""
from types import SimpleNamespace
from time import perf_counter
import numpy as np
from .point_relations import relation_rows
from .pointwise_mlp import score_rows, pack_rows
from .io import log


def relation_args(cfg):
    return SimpleNamespace(geo_norm=cfg['geo_norm'], geo_sigma=cfg['geo_sigma'])


def make_rows(prepared, pairs, cfg):
    args = relation_args(cfg)
    return [relation_rows(prepared[int(i)], prepared[int(j)], args) for i,j in pairs]


def score_candidates(prepared, pairs, model, device, cfg, label='candidate_scoring'):
    n = len(pairs)
    lengths = np.array([len(p) for p in prepared], np.int64)
    order = np.argsort(lengths[pairs[:,0]]*lengths[pairs[:,1]], kind='stable')
    logits = np.empty(n, np.float32)
    total_rows, last, begin = 0, 0., perf_counter()
    chunk = cfg['score_chunk']
    for start in range(0, n, chunk):
        ids = order[start:start+chunk]
        rows = make_rows(prepared, pairs[ids], cfg)
        logits[ids], _ = score_rows(rows, model, str(device), cfg['score_batch'])
        total_rows += sum(len(r) for r in rows)
        now = perf_counter()
        if now-last >= 15 or start+chunk >= n:
            done = min(start+chunk, n)
            elapsed = now-begin
            log(label, completed=done, total=n, elapsed_seconds=round(elapsed,2),
                estimated_remaining_seconds=round(elapsed*(n-done)/done,2))
            last = now
    return logits, dict(scored_pairs=n, valid_relation_rows=int(total_rows), additional_pairs=0,
                        seconds=perf_counter()-begin)
