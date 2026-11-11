"""Deterministic single-node local descent on a fixed observed signed graph.

This is a small coordinate-descent heuristic, not a full Kernighan-Lin solver.
It minimizes sum_{observed(i,j)} c_ij * [pred_i != pred_j]. A positive cost
prefers equality; a negative cost prefers a cut. Missing pairs have no terms.
No labels, target cluster count/capacity, scorer, or extra-edge API exists.

Existing-cluster moves require at least one observed adjacent edge (an explicit
zero edge counts as observed). A fresh singleton is also allowed. Splitting
structurally disconnected parts of a cluster leaves this objective unchanged;
after such a split the descent scans again before declaring convergence.
"""
from __future__ import annotations

import math
from time import perf_counter
import numpy as np

_MAX_PASSES = 1000  # fixed termination safeguard; not a selected model parameter
_EPS_FACTOR = 64.0


def _graph(n, pairs, costs):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    p = np.asarray(pairs)
    if p.size == 0 and p.ndim == 1:
        p = np.empty((0, 2), dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 2 or (p.size and p.dtype.kind not in 'iu'):
        raise ValueError('pairs must be integer [E,2]')
    p = np.array(p, dtype=np.int64, copy=True)
    w = np.asarray(costs, dtype=np.float64)
    if w.ndim != 1 or len(w) != len(p) or not np.all(np.isfinite(w)):
        raise ValueError('costs must be a matching finite vector')
    if np.any(p < 0) or np.any(p >= n) or np.any(p[:, 0] == p[:, 1]):
        raise ValueError('require distinct endpoints in [0,n)')
    original_edges = len(p)
    if len(p):
        p.sort(axis=1)
        order = np.lexsort((p[:, 1], p[:, 0]))
        p, w = p[order], w[order]
        dup = np.all(p[1:] == p[:-1], axis=1)
        if np.any(dup & (w[1:] != w[:-1])):
            raise ValueError('conflicting duplicate edge costs')
        keep = np.r_[True, ~dup]
        p, w = p[keep], w[keep]
    row = np.r_[p[:, 0], p[:, 1]]
    col = np.r_[p[:, 1], p[:, 0]]
    val = np.r_[w, w]
    order = np.lexsort((col, row))
    row, col, val = row[order], col[order], val[order]
    ptr = np.r_[0, np.cumsum(np.bincount(row, minlength=n), dtype=np.int64)]
    absolute_degree = np.bincount(row, weights=np.abs(val), minlength=n)
    try:
        absolute_mass = math.fsum(np.abs(w))
    except OverflowError as exc:
        raise ValueError('absolute cost sum overflows float64') from exc
    if not math.isfinite(absolute_mass) or not np.all(np.isfinite(absolute_degree)):
        raise ValueError('absolute cost sum overflows float64')
    return p, w, ptr, col, val, absolute_degree, absolute_mass, original_edges


def _canonical(pred, n):
    p = np.asarray(pred)
    if p.ndim != 1 or len(p) != n or (p.size and p.dtype.kind not in 'iu'):
        raise ValueError('pred must contain one integer cluster label per node')
    result = np.empty(n, dtype=np.int64)
    mapping = {}
    for v, label in enumerate(p):
        label = int(label)
        if label not in mapping:
            mapping[label] = len(mapping)
        result[v] = mapping[label]
    return result


def _split_components(labels, ptr, nbr):
    n = len(labels)
    split = np.full(n, -1, dtype=np.int64)
    cid = 0
    for seed in range(n):
        if split[seed] >= 0:
            continue
        split[seed] = cid
        queue = [seed]
        at = 0
        while at < len(queue):
            v = queue[at]
            at += 1
            adjacent = nbr[ptr[v]:ptr[v + 1]]
            selected = adjacent[(labels[adjacent] == labels[seed]) & (split[adjacent] < 0)]
            split[selected] = cid
            queue.extend(selected.tolist())
        cid += 1
    old_count = len(np.unique(labels))
    return split, cid - old_count


def _energy(pairs, weights, pred):
    return math.fsum(weights[pred[pairs[:, 0]] != pred[pairs[:, 1]]])


def refine_signed_partition(n, pairs, costs, pred):
    """Return a complete partition and objective/termination diagnostics.

    Costs must already contain any validation-selected bias, applied once per
    original observed edge. No threshold is learned or applied in this solver.
    Node order is fixed ascending; cost ties use the smallest current target
    cluster ID, with the fresh singleton represented by the lowest free ID.
    Current IDs are initialized and finally canonicalized by first node.

    Machine-scale acceptance tolerance is 64*eps*max(1, incident absolute cost).
    It is numerical protection, not a tunable identity/similarity threshold.
    At most 1000 full coordinate passes are allowed. Hitting that safeguard
    returns converged=False and cannot be reported as a completed local optimum.
    """
    begin = perf_counter()
    p, w, ptr, nbr, val, degree_abs, mass_abs, input_edges = _graph(n, pairs, costs)
    labels = _canonical(pred, n)
    energy_before = _energy(p, w, labels)
    initial_cluster_count = len(np.unique(labels))
    sizes = np.bincount(labels, minlength=n)
    traces, passes = [], []
    split_count = 0
    converged = False
    eps = np.finfo(np.float64).eps
    for pass_id in range(1, _MAX_PASSES + 1):
        accepted = 0
        for v in range(n):
            source = int(labels[v])
            lo, hi = int(ptr[v]), int(ptr[v + 1])
            neighbor_labels = labels[nbr[lo:hi]]
            adjacent_count = np.bincount(neighbor_labels, minlength=n)
            sums = np.bincount(neighbor_labels, weights=val[lo:hi], minlength=n)
            targets = np.flatnonzero(adjacent_count > 0)
            targets = targets[targets != source]
            target = -1
            best_sum = -math.inf
            if len(targets):
                which = int(np.argmax(sums[targets]))
                target = int(targets[which])
                best_sum = float(sums[target])
            new_singleton = False
            if sizes[source] > 1:
                free = int(np.flatnonzero(sizes == 0)[0])
                if 0.0 > best_sum or (0.0 == best_sum and (target < 0 or free < target)):
                    target = free
                    best_sum = 0.0
                    new_singleton = True
            if target < 0:
                continue
            source_sum = math.fsum(val[lo:hi][neighbor_labels == source])
            target_sum = 0.0 if new_singleton else math.fsum(val[lo:hi][neighbor_labels == target])
            gain = target_sum - source_sum
            guard = _EPS_FACTOR * eps * max(1.0, float(degree_abs[v]))
            if gain <= guard:
                continue
            if not new_singleton and adjacent_count[target] == 0:
                raise AssertionError('move to unavailable cluster')
            traces.append(dict(pass_index=pass_id, node=v, source=source, target=target,
                               new_singleton=new_singleton, source_sum=source_sum,
                               target_sum=target_sum, gain=gain, numerical_guard=guard))
            labels[v] = target
            sizes[source] -= 1
            sizes[target] += 1
            accepted += 1
        entry = dict(pass_index=pass_id, moves=accepted, connected_component_splits=0)
        if accepted == 0:
            canonical, extra = _split_components(labels, ptr, nbr)
            labels = canonical
            sizes = np.bincount(labels, minlength=n)
            entry['connected_component_splits'] = extra
            split_count += extra
            if extra == 0:
                converged = True
        passes.append(entry)
        if converged:
            break
    # The final split is objective-neutral even when the safety cap was hit.
    labels, extra = _split_components(labels, ptr, nbr)
    split_count += extra
    if extra:
        converged = False
    energy_after = _energy(p, w, labels)
    tolerance = _EPS_FACTOR * eps * max(1.0, mass_abs)
    if energy_after > energy_before + tolerance:
        raise AssertionError('local descent increased the observed signed-cut objective')
    diagnostics = dict(
        algorithm='single-node coordinate descent on observed signed-cut objective',
        is_full_kernighan_lin=False, vertices=int(n), observed_edges=len(p), input_edges=input_edges,
        duplicate_input_edges=input_edges-len(p), observed_zero_cost_edges=int(np.count_nonzero(w == 0)),
        initial_cluster_count=initial_cluster_count, cluster_count=int(len(np.unique(labels))),
        energy_before=energy_before, energy_after=energy_after,
        objective_improvement=energy_before-energy_after,
        sum_accepted_gains=float(math.fsum(t['gain'] for t in traces)),
        objective_check_tolerance=tolerance, passes=len(passes), moves=len(traces),
        pass_trace=passes, move_trace=traces, connected_component_splits=split_count,
        converged=converged, safety_pass_cap=_MAX_PASSES,
        stop_reason='no numerically improving single-node move' if converged else 'pass safeguard reached or final structural split requires another scan',
        numerical_guard='64*float64_eps*max(1, incident absolute cost)',
        fixed_candidate_graph=True, extra_edges=0, additional_pairs=0, new_score_calls=0,
        uses_M=False, uses_K=False, labels_used=False,
        missing_policy='no objective term; existing targets require observed adjacency',
        tie_break='ascending node; smallest current cluster ID, fresh singleton uses lowest free ID',
        structural_split='only disconnected components of existing within-cluster observed edges; no objective change',
        elapsed_seconds=perf_counter()-begin)
    return dict(pred=labels, diagnostics=diagnostics)
