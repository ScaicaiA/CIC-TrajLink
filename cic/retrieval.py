"""Exact dual-path cosine retrieval with externally preset integer budgets.

No label argument, recall calculation, budget selection, or missing-pair score.
Rows are processed in blocks to bound temporary similarity memory. Rank ties
use ascending trajectory row index, exactly as the preceding implementation.
"""
import numpy as np
from scipy.sparse import csr_matrix


def poi_vectors(trajectories, idf):
    rows, cols, values, vocab = [], [], [], {}
    for i, t in enumerate(trajectories):
        for poi in sorted({str(p.poi_id) for p in t.points if p.poi_id}):
            weight = float(idf.get(poi, 0.))
            if weight > 0:
                rows.append(i)
                cols.append(vocab.setdefault(poi, len(vocab)))
                values.append(weight)
    x = csr_matrix((np.asarray(values, np.float32), (rows, cols)),
                   shape=(len(trajectories), len(vocab)), dtype=np.float32)
    norm = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    inv = np.divide(1., norm, out=np.zeros_like(norm), where=norm > 0)
    return x.multiply(inv[:, None]).tocsr()


def similarity_blocks(poi, embeddings, block_rows=128):
    if embeddings.shape != (poi.shape[0], 32) or not np.all(np.isfinite(embeddings)):
        raise ValueError('Finite N x 32 trajectory embeddings required')
    norm = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norm, 1., rtol=1e-5, atol=1e-6):
        raise ValueError('Trajectory embeddings must be L2 normalized')
    if block_rows < 1:
        raise ValueError('block_rows must be positive')
    for start in range(0, len(embeddings), block_rows):
        stop = min(start + block_rows, len(embeddings))
        yield start, (poi[start:stop] @ poi.T).toarray().astype(np.float32), embeddings[start:stop] @ embeddings.T


def ordered_neighbors(row, self_index):
    ids = np.flatnonzero(np.isfinite(row) & (row > 0))
    ids = ids[ids != self_index]
    return ids[np.lexsort((ids, -row[ids]))]


def candidates(trajectories, idf, embeddings, kp, kc, block_rows=128):
    for value in (kp, kc):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError('Preset k_p and k_c must be nonnegative integers')
    n = len(trajectories)
    if n < 2:
        return np.empty((0, 2), np.int64), np.empty(0, np.uint8)
    budgets = (min(kp, n-1), min(kc, n-1))
    poi = poi_vectors(trajectories, idf)
    chunks = [[], []]
    for start, ps, cs in similarity_blocks(poi, embeddings, block_rows):
        for channel, sim in enumerate((ps, cs)):
            if budgets[channel] == 0:
                continue
            for offset, row in enumerate(sim):
                i = start + offset
                selected = ordered_neighbors(row, i)[:budgets[channel]]
                chunks[channel].append(np.minimum(i, selected)*n + np.maximum(i, selected))
    per_path = [np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64) for parts in chunks]
    codes = np.union1d(*per_path)
    pairs = np.column_stack((codes//n, codes%n)).astype(np.int64)
    origin = np.zeros(len(pairs), np.uint8)
    for bit, edges in zip((1, 2), per_path):
        origin[np.searchsorted(codes, edges)] |= bit
    return pairs, origin
