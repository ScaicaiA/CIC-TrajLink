"""Observed-edge signed multicut via Greedy Additive Edge Contraction (GAEC).

This pipeline supplies c_ij = 2 * calibrated_p_ij - 1 as input and
leaves this generic solver's optional bias at zero. Its legacy argument name
raw_logits denotes supplied edge values, not a new neural scoring call. Minimize
    E(partition) = sum_{ij in observed_edges} c_ij * [label_i != label_j].
Positive costs support joining; negative costs support separating. An absent
edge has no term at all: it is never given an identity score or negative label.
The bias is applied once per original observed edge, not once per cluster join.

Start from singleton components and greedily join the observed-connected pair
with maximum strictly positive SUM of signed costs across their boundary.
Update boundary costs additively. Stop when no join decreases E strictly.
This is a feasible heuristic, not an exact or globally optimal multicut solver.
There is no user count, capacity, scorer, or label input.

Reference: Keuper et al., Efficient Decomposition of Image and Mesh Graphs by
Lifted Multicuts, ICCV 2015, Sec. 4.1 and Algorithm 1 (ordinary multicut F=empty).
https://openaccess.thecvf.com/content_iccv_2015/papers/Keuper_Efficient_Decomposition_of_ICCV_2015_paper.pdf
The paper prose specifies strictly improving joins. We reject zero-gain joins.
Implementation differs from its adjacency-list/priority-queue implementation:
we keep dense aggregate cost/count buffers with an explicit observation mask,
and one cached best neighbor per active row; worst-case O(n^3) row rescans and
O(n^2) memory, no claim to the paper implementation's asymptotic time bound.
"""
from __future__ import annotations

import math
from time import perf_counter
import numpy as np


def _edges(n, pairs, raw_logits, bias):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    if isinstance(bias, (bool, np.bool_)) or not np.isscalar(bias) or not math.isfinite(float(bias)):
        raise ValueError('bias must be a finite scalar')
    n, bias = int(n), float(bias)
    p = np.asarray(pairs)
    if p.ndim == 1 and p.size == 0:
        p = np.empty((0, 2), np.int64)
    if p.ndim != 2 or p.shape[1] != 2 or (p.size and p.dtype.kind not in 'iu'):
        raise ValueError('pairs must be an integer [E,2] array')
    p = p.astype(np.int64, copy=True)
    w = np.asarray(raw_logits, np.float64)
    if w.ndim != 1 or len(w) != len(p) or not np.all(np.isfinite(w)):
        raise ValueError('raw_logits must be a matching finite vector')
    if np.any(p < 0) or np.any(p >= n) or np.any(p[:, 0] == p[:, 1]):
        raise ValueError('require distinct endpoints within [0,n)')
    input_count = len(p)
    if len(p):
        p.sort(axis=1)
        order = np.lexsort((p[:, 1], p[:, 0]))
        p, w = p[order], w[order]
        repeated = np.all(p[1:] == p[:-1], axis=1)
        if np.any(repeated & (w[1:] != w[:-1])):
            raise ValueError('conflicting scores for one observed undirected edge')
        keep = np.r_[True, ~repeated]
        p, w = p[keep], w[keep]
    with np.errstate(over='ignore', invalid='ignore'):
        costs = w - bias
    if not np.all(np.isfinite(costs)):
        raise ValueError('costs after bias overflowed float64')
    try:
        absolute_sum = math.fsum(np.abs(costs))
    except OverflowError as exc:
        raise ValueError('absolute cost sum exceeds float64') from exc
    if not math.isfinite(absolute_sum):
        raise ValueError('absolute cost sum exceeds float64')
    return n, p, costs, bias, input_count


def signed_gaec(n, pairs, raw_logits, bias=0.0, progress=None):
    """Return pred, diagnostics and merge_trace using only observed edges.

    Cluster ID is its smallest node. Exact equal-gain ties use the
    lexicographically smallest cluster-ID pair. Observed zero costs remain
    structurally present but never cause a zero-gain join. Reversed duplicate
    edges with exactly identical raw logits are deduplicated; conflicting
    duplicates are rejected. No input array is mutated.
    """
    begin = perf_counter()
    n, p, costs, bias, input_count = _edges(n, pairs, raw_logits, bias)
    sums = np.zeros((n, n), np.float64)
    count_dtype = np.uint32 if len(p) < np.iinfo(np.uint32).max else np.uint64
    counts = np.zeros((n, n), count_dtype)
    if len(p):
        i, j = p.T
        sums[i, j] = sums[j, i] = costs
        counts[i, j] = counts[j, i] = 1
    active = np.ones(n, dtype=bool)
    sizes = np.ones(n, dtype=np.int64)
    members = [[i] for i in range(n)]
    best = np.full(n, -1, np.int64)
    best_value = np.full(n, -math.inf)
    trace = []
    row_recomputations = 0
    merge_count = 0

    def emit(phase, **fields):
        if progress is not None:
            progress(dict(phase=phase, elapsed_seconds=perf_counter()-begin,
                          n=n, input_edges=len(p), additional_pairs=0,
                          clusters=n-merge_count, **fields))

    def recompute(i):
        nonlocal row_recomputations
        # counts is an observation mask; an empty accumulator's numerical zero
        # is never mistaken for a measured edge score.
        allowed = np.flatnonzero(active & (counts[i] > 0))
        if len(allowed):
            at = int(np.argmax(sums[i, allowed]))
            best[i] = allowed[at]
            best_value[i] = sums[i, best[i]]
        else:
            best[i], best_value[i] = -1, -math.inf
        row_recomputations += 1

    emit('signed_gaec_initializing')
    for i in range(n):
        recompute(i)
    while n and np.any(active):
        selected = int(np.argmax(best_value))
        gain = float(best_value[selected])
        if best[selected] < 0 or not gain > 0.0:
            break
        a, b = sorted((selected, int(best[selected])))
        if not active[a] or not active[b] or counts[a, b] == 0:
            raise AssertionError('stale or unobserved contraction')
        trace.append(dict(left=a, right=b, left_size=int(sizes[a]),
                          right_size=int(sizes[b]), observed_edges=int(counts[a, b]),
                          gain=float(sums[a, b])))
        old_best = best.copy()
        sums[a] += sums[b]
        counts[a] += counts[b]
        sums[:, a] = sums[a]
        counts[:, a] = counts[a]
        sums[a, a] = 0.0
        counts[a, a] = 0
        sums[b, :] = sums[:, b] = 0.0
        counts[b, :] = counts[:, b] = 0
        active[b] = False
        sizes[a] += sizes[b]
        sizes[b] = 0
        members[a] = sorted(members[a] + members[b])
        members[b] = []
        best[b], best_value[b] = -1, -math.inf
        merge_count += 1
        recompute(a)
        invalid = active & ((old_best == a) | (old_best == b))
        invalid[a] = False
        for i in np.flatnonzero(invalid):
            recompute(int(i))
        # Every other boundary is unchanged except the new a boundary.
        others = np.flatnonzero(active & ~invalid & (np.arange(n) != a) & (counts[:, a] > 0))
        if len(others):
            value = sums[others, a]
            improve = ((value > best_value[others]) |
                       ((value == best_value[others]) & ((best[others] < 0) | (a < best[others]))))
            changed = others[improve]
            best[changed], best_value[changed] = a, value[improve]
        if merge_count % 100 == 0:
            emit('signed_gaec_merging', completed_merges=merge_count)
    pred = np.full(n, -1, np.int64)
    clusters = [members[i] for i in np.flatnonzero(active)]
    for cid, cluster in enumerate(clusters):
        pred[cluster] = cid
    if np.any(pred < 0):
        raise AssertionError('some trajectory was not assigned')
    crossing = pred[p[:, 0]] != pred[p[:, 1]]
    initial_objective = math.fsum(costs)
    final_objective = math.fsum(costs[crossing])
    total_gain = math.fsum(t['gain'] for t in trace)
    disagreement = math.fsum(costs[crossing & (costs > 0)]) - math.fsum(costs[~crossing & (costs < 0)])
    terminal_gain = float(np.max(best_value[active])) if np.any(active) else -math.inf
    diagnostics = dict(
        method='observed-edge signed multicut GAEC; strict-positive joins',
        n=n, input_edges=input_count, observed_edges=len(p),
        duplicate_input_edges=input_count-len(p), bias=bias,
        uses_M=False, uses_K=False, uses_labels=False,
        fixed_candidate_graph=True, additional_pairs=0, new_score_calls=0,
        missing_policy='no objective term; explicit count>0 observation mask',
        cost_definition='raw_logit - bias, applied once per original observed edge',
        positive_cost_edges=int(np.count_nonzero(costs > 0)),
        negative_cost_edges=int(np.count_nonzero(costs < 0)),
        zero_cost_edges=int(np.count_nonzero(costs == 0)),
        clusters=len(clusters), max_cluster_size=max(map(len, clusters), default=0),
        singleton_clusters=sum(len(c) == 1 for c in clusters),
        merge_count=merge_count, row_recomputations=row_recomputations,
        initial_multicut_objective=initial_objective,
        final_multicut_objective=final_objective, positive_merge_gain=total_gain,
        objective_accounting_residual=final_objective-(initial_objective-total_gain),
        weighted_disagreement=disagreement,
        max_remaining_join_gain=terminal_gain if math.isfinite(terminal_gain) else None,
        stopped_without_positive_join=not terminal_gain > 0.0,
        cluster_matrix_bytes=sums.nbytes+counts.nbytes,
        heuristic=True, optimality_guarantee=False,
        elapsed_seconds=perf_counter()-begin)
    emit('signed_gaec_complete', completed_merges=merge_count)
    return dict(pred=pred, diagnostics=diagnostics, merge_trace=trace)
