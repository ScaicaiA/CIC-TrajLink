"""Label-free local standardization of a fixed set of observed edge logits.

For node i, mu_i and sigma_i are the population mean and standard deviation
of ONLY the supplied incident edge scores. Missing trajectory pairs are not
samples. For an observed edge ij the output, in exactly the input order, is

    z_ij = 0.5 * [(s_ij-mu_i)/sigma_i + (s_ij-mu_j)/sigma_j].

A zero-standard-deviation endpoint contributes zero, including isolated
nodes (which occur in no output edge). No epsilon, neighbor count, M/K, or
label parameter is used. To keep constant neighborhoods exactly constant,
means/variances are computed in coordinates centered on one actually
observed score per node. A second pass squares the centered deviations.
Floating-point variance underflow to zero follows the same zero rule;
there is no adjustable numerical tolerance or variance floor.

This is a new score-calibration heuristic, not a probability calibration or
an identity guarantee. It can elevate a relatively good but weak raw edge.
It never generates new edges or invokes a model/scorer.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np


def locally_standardized_logits(n, pairs, raw_logits):
    """Return ``(z, diagnostics)`` with input-order float64 z and JSON stats.

    The supplied undirected edge set must be unique. Reversed duplicates
    are rejected rather than double-counted or silently changing the graph.
    No input array is mutated. Node statistics in diagnostics use original
    node IDs so the transform can be audited without labels.
    """
    begin = perf_counter()
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    n = int(n)
    p = np.asarray(pairs)
    if p.size == 0 and p.ndim == 1:
        p = np.empty((0, 2), dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 2 or (p.size and p.dtype.kind not in 'iu'):
        raise ValueError('pairs must be integer [E,2] endpoints')
    p = p.astype(np.int64, copy=False)
    raw = np.asarray(raw_logits, dtype=np.float64)
    if raw.ndim != 1 or len(raw) != len(p) or not np.all(np.isfinite(raw)):
        raise ValueError('raw_logits must be a matching finite vector')
    if np.any(p < 0) or np.any(p >= n) or np.any(p[:, 0] == p[:, 1]):
        raise ValueError('require distinct endpoints in [0,n)')
    if len(p):
        canonical = np.sort(p, axis=1)
        order = np.lexsort((canonical[:, 1], canonical[:, 0]))
        ordered = canonical[order]
        if np.any(np.all(ordered[1:] == ordered[:-1], axis=1)):
            raise ValueError('supplied undirected observed edge set must be unique')

    left, right = p[:, 0], p[:, 1]
    counts = np.bincount(left, minlength=n) + np.bincount(right, minlength=n)
    seen = counts > 0
    reference = np.zeros(n, dtype=np.float64)
    first = np.full(n, len(p), dtype=np.int64)
    index = np.arange(len(p), dtype=np.int64)
    np.minimum.at(first, left, index)
    np.minimum.at(first, right, index)
    reference[seen] = raw[first[seen]]

    # Centering on an actual incident value gives an exactly zero offset
    # for constant neighborhoods, including repeated nonbinary constants.
    with np.errstate(over='ignore', invalid='ignore'):
        offset_left = raw - reference[left]
        offset_right = raw - reference[right]
        offset_sum = (np.bincount(left, weights=offset_left, minlength=n)
                      + np.bincount(right, weights=offset_right, minlength=n))
        mean_offset = np.divide(offset_sum, counts, out=np.zeros(n), where=seen)
        mean = reference + mean_offset
        deviation_left = offset_left - mean_offset[left]
        deviation_right = offset_right - mean_offset[right]
        squared_sum = (np.bincount(left, weights=np.square(deviation_left), minlength=n)
                       + np.bincount(right, weights=np.square(deviation_right), minlength=n))
        variance = np.divide(squared_sum, counts, out=np.zeros(n), where=seen)
        std = np.sqrt(variance)
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
            and np.all(np.isfinite(deviation_left)) and np.all(np.isfinite(deviation_right))):
        raise ValueError('local statistics exceed float64 range')

    variable = std > 0.0
    normalized_left = np.divide(deviation_left, std[left], out=np.zeros(len(p)),
                                where=variable[left])
    normalized_right = np.divide(deviation_right, std[right], out=np.zeros(len(p)),
                                 where=variable[right])
    z = 0.5 * (normalized_left + normalized_right)
    if not np.all(np.isfinite(z)):
        raise ValueError('locally standardized scores are not finite')
    diagnostics = dict(
        algorithm='endpoint-local observed-logit population standardization',
        formula='z_ij = 0.5*((s_ij-mu_i)/sigma_i + (s_ij-mu_j)/sigma_j)',
        vertices=n, observed_edges=len(p), output_edges=len(z),
        output_alignment='exactly supplied pair row order; no sorting/filtering of output',
        additional_pairs=0, new_score_calls=0, fixed_candidate_graph=True,
        uses_M_K=False, labels_used=False, free_hyperparameters=0,
        missing_edge_policy='not observed, not counted in either endpoint statistics',
        variance_convention='population; denominator is observed incident count, ddof=0',
        stable_statistics='actual observed reference shift, then centered two-pass variance',
        zero_std_policy='exact numerical std==0 contributes 0; no epsilon threshold or floor',
        zero_variance_note='squared-deviation underflow to zero follows zero_std_policy',
        isolated_node_count=int(np.count_nonzero(~seen)),
        observed_zero_std_node_count=int(np.count_nonzero(seen & ~variable)),
        observed_nonzero_std_node_count=int(np.count_nonzero(seen & variable)),
        node_observed_degree=counts.tolist(), node_mean=mean.tolist(), node_std=std.tolist(),
        guarantee='label-free relative-score heuristic, not a calibrated same-user probability',
        elapsed_seconds=perf_counter()-begin)
    return z, diagnostics
