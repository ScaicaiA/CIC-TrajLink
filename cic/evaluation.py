"""Post-prediction partition metrics and explicitly scoped resource reporting.

Nothing in this module selects a partition or supplies labels to MeanCut.
Call ``evaluate_partition(labels, pred)`` only after ``pred`` is final. Keep
evaluation outside the ``ResourceMonitor`` context when measuring inference.

Pairwise zero denominators follow the existing experiment (value 0). NMI uses
geometric normalization. ARI, and NMI when both entropies are zero, use the
standard perfect-agreement convention (1), unlike the old hand-written
degenerate-case ARI/NMI implementation. Negative predicted IDs, including -1,
are ordinary cluster IDs: this evaluator never silently discards noise points.
"""
from __future__ import annotations

from collections import Counter
import math
import numbers
import os
import sys
import threading
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None


def _factorize(values, name: str) -> tuple[np.ndarray, int]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional label vector')
    if not len(array):
        raise ValueError('partition evaluation requires at least one sample')
    mapping = {}
    encoded = np.empty(len(array), dtype=np.int64)
    for index, value in enumerate(array):
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or (isinstance(value, numbers.Real) and not math.isfinite(value)):
            raise ValueError(f'{name} contains a missing or nonfinite label')
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f'{name} labels must be hashable scalar values') from exc
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded[index] = mapping[value]
    return encoded, len(mapping)


def _choose2_sum(values) -> int:
    # Python integer arithmetic avoids overflow in a numpy int64 square.
    return sum(int(value) * (int(value) - 1) // 2 for value in np.ravel(values))


def evaluate_partition(labels, pred) -> dict:
    """Evaluate a completed prediction, allowing any number of predicted groups.

    G-Acc is a maximum one-to-one rectangular Hungarian assignment divided by
    N. Unmatched rows/columns contribute zero; over-fragmentation is not hidden
    by assigning multiple predicted clusters to the same true user.

    ``giant_cluster_fraction`` is simply max predicted cluster size / N, with
    no hidden definition of how large a cluster must be to count as "giant".
    Fragmentation diagnostics explicitly measure true-cluster splitting. They
    are evaluation-only and must not become inference stopping criteria.
    """
    begin = perf_counter()
    y, n_true = _factorize(labels, 'labels')
    p, n_pred = _factorize(pred, 'pred')
    if len(y) != len(p):
        raise ValueError('labels and pred must have equal lengths')
    n = len(y)
    table = np.zeros((n_true, n_pred), dtype=np.int64)
    np.add.at(table, (y, p), 1)
    true_sizes = table.sum(axis=1)
    pred_sizes = table.sum(axis=0)
    rows, columns = linear_sum_assignment(-table)
    matched = int(table[rows, columns].sum())

    tp = _choose2_sum(table)
    true_pairs = _choose2_sum(true_sizes)
    pred_pairs = _choose2_sum(pred_sizes)
    precision = tp / pred_pairs if pred_pairs else 0.0
    recall = tp / true_pairs if true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    total_pairs = n * (n - 1) // 2
    if not total_pairs:
        ari = 1.0
    else:
        expected = true_pairs * pred_pairs / total_pairs
        denominator = 0.5 * (true_pairs + pred_pairs) - expected
        ari = (tp - expected) / denominator if denominator else 1.0

    true_prob = true_sizes / n
    pred_prob = pred_sizes / n
    h_true = float(-np.sum(true_prob * np.log(true_prob)))
    h_pred = float(-np.sum(pred_prob * np.log(pred_prob)))
    nonzero_row, nonzero_col = np.nonzero(table)
    joint = table[nonzero_row, nonzero_col] / n
    mi = float(np.sum(joint * np.log(
        joint / (true_prob[nonzero_row] * pred_prob[nonzero_col]))))
    if h_true == 0.0 or h_pred == 0.0:
        nmi = 1.0 if h_true == 0.0 and h_pred == 0.0 else 0.0
    else:
        nmi = min(1.0, max(0.0, mi / math.sqrt(h_true * h_pred)))

    fragments = np.count_nonzero(table, axis=1)
    true_users_per_pred = np.count_nonzero(table, axis=0)
    largest_fragment = table.max(axis=1)
    singleton_count = int(np.count_nonzero(pred_sizes == 1))
    size_hist = Counter(int(value) for value in pred_sizes)
    report = dict(
        Grouping_ACC=matched / n,
        Pairwise_Precision=float(precision), Pairwise_Recall=float(recall),
        Pairwise_F1=float(f1), NMI=float(nmi), ARI=float(ari),
        n_samples=n, n_true_clusters=n_true, n_pred_clusters=n_pred,
        cluster_count=n_pred, max_cluster_size=int(pred_sizes.max()),
        singleton_clusters=singleton_count,
        singleton_sample_fraction=singleton_count / n,
        giant_cluster_fraction=int(pred_sizes.max()) / n,
        cluster_size_histogram={str(size): size_hist[size] for size in sorted(size_hist)},
        top10_cluster_sizes=sorted((int(v) for v in pred_sizes), reverse=True)[:10],
        fragmentation_mean_fragments_per_true_cluster=float(np.mean(fragments)),
        fragmentation_max_fragments_per_true_cluster=int(fragments.max()),
        fragmentation_true_clusters_split_fraction=float(np.mean(fragments > 1)),
        fragmentation_mean_largest_fragment_fraction=float(np.mean(largest_fragment / true_sizes)),
        fragmentation_sample_weighted_loss=1.0 - int(largest_fragment.sum()) / n,
        mixing_mean_true_users_per_pred_cluster=float(np.mean(true_users_per_pred)),
        mixing_pred_clusters_fraction=float(np.mean(true_users_per_pred > 1)),
        pair_counts=dict(true_positive=tp, true_same_user=true_pairs,
                         predicted_same_user=pred_pairs,
                         false_positive=pred_pairs - tp, false_negative=true_pairs - tp),
        metric_conventions=dict(
            grouping_acc='rectangular one-to-one Hungarian; unmatched groups contribute zero',
            pairwise='unordered distinct sample pairs; zero denominator -> 0',
            nmi='geometric normalization; both zero entropies -> 1',
            ari='standard adjusted Rand; identical degenerate partitions -> 1',
            giant_cluster_fraction='largest predicted cluster / N; no threshold',
            fragmentation='true clusters split across predicted clusters; evaluation only',
            negative_predicted_ids='ordinary cluster labels; no samples discarded'))
    report['evaluation_seconds'] = perf_counter() - begin
    return report


def evaluate_candidates_after_prediction(labels, pairs):
    """Test-only audit after a partition is frozen; never chooses retrieval k."""
    y = np.asarray(labels)
    pairs = np.asarray(pairs, dtype=np.int64)
    sizes = np.unique(y, return_counts=True)[1]
    total_positive = int(np.sum(sizes * (sizes-1)//2))
    kept_positive = int(np.sum(y[pairs[:, 0]] == y[pairs[:, 1]]))
    return dict(total_same_user_pairs=total_positive, retrieved_same_user_pairs=kept_positive,
                missed_same_user_pairs=total_positive-kept_positive,
                candidate_pair_recall=kept_positive/total_positive if total_positive else None,
                used_for_selection=False, scope='final target audit only, after saved partition')


def _make_rss_reader() -> tuple[Callable[[], int] | None, str, str | None]:
    """Current process RSS, never ru_maxrss disguised as current RSS."""
    psutil_problem = None
    try:
        import psutil
        process = psutil.Process(os.getpid())
        process.memory_info().rss  # Establish that this backend is usable.
        return lambda: int(process.memory_info().rss), 'psutil', None
    except (ImportError, OSError) as exc:
        psutil_problem = type(exc).__name__
    except Exception as exc:  # Optional backend failure must not stop inference.
        psutil_problem = type(exc).__name__
    if sys.platform.startswith('linux'):
        page_size = int(os.sysconf('SC_PAGE_SIZE'))

        def read_statm():
            with open('/proc/self/statm', encoding='ascii') as handle:
                return int(handle.read().split()[1]) * page_size

        try:
            read_statm()
            return read_statm, 'linux_proc_statm', psutil_problem
        except (OSError, ValueError, IndexError):
            pass
    return None, 'unavailable', psutil_problem


def _process_high_water() -> dict:
    """Return lifetime high-water RSS with explicit native-unit conversion."""
    if resource is None:
        return dict(bytes=None, raw=None, raw_unit='unavailable')
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == 'darwin':
        return dict(bytes=int(raw), raw=raw, raw_unit='bytes')
    if sys.platform.startswith('linux'):
        return dict(bytes=int(raw * 1024), raw=raw, raw_unit='KiB')
    return dict(bytes=None, raw=raw, raw_unit='platform_native_unknown')


class ResourceMonitor:
    """Measure one stage's time and sampled process RSS without reading labels.

    Example::

        with ResourceMonitor(interval_seconds=0.05) as monitor:
            pred = algorithm.fit_predict(features)
        resource_report = monitor.as_dict()
        metrics = evaluate_partition(labels, pred)  # outside monitored stage

    This measures host-process RSS only, not GPU memory or child processes.
    Short RSS spikes can fall between samples; the sampled peak is a lower
    bound on the actual stage peak. If no current-RSS backend is available,
    stage RSS fields are None. ``resource.ru_maxrss`` is reported separately as
    process-lifetime high-water memory and is never relabelled a stage peak.
    """
    def __init__(self, interval_seconds: float = 0.05):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError('interval_seconds must be finite and positive')
        self.interval_seconds = float(interval_seconds)
        self._started = False
        self._finished = False
        self._reader = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._rss_base = self._rss_peak = self._rss_end = None
        self._sample_count = 0
        self._sample_errors = []

    def _sample_rss(self, role=None):
        if role == 'base':
            self._rss_base = None
        elif role == 'end':
            self._rss_end = None
        if self._reader is None:
            return
        try:
            value = int(self._reader())
            if value < 0:
                raise ValueError('negative RSS')
            with self._lock:
                if role == 'base':
                    self._rss_base = value
                self._rss_peak = max(value, self._rss_peak or 0)
                if role == 'end':
                    self._rss_end = value
                self._sample_count += 1
        except Exception as exc:
            with self._lock:
                self._sample_errors.append(type(exc).__name__)

    def _loop(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample_rss()

    def __enter__(self):
        if self._started:
            raise RuntimeError('a ResourceMonitor instance measures exactly one stage')
        self._started = True
        self._reader, self._backend, self._backend_note = _make_rss_reader()
        self._high_water_start = _process_high_water()
        self._sample_rss(role='base')
        if self._reader is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name='stage-rss-sampler')
            self._thread.start()
        self._begin = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._seconds = perf_counter() - self._begin
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample_rss(role='end')
        self._high_water_end = _process_high_water()
        self._finished = True
        return False

    def as_dict(self) -> dict:
        if not self._finished:
            raise RuntimeError('resource report is available after leaving the context')
        start = self._high_water_start['bytes']
        end = self._high_water_end['bytes']
        return dict(
            stage_seconds=self._seconds,
            rss_backend=self._backend, rss_backend_note=self._backend_note,
            stage_rss_base_bytes=self._rss_base,
            stage_rss_peak_bytes=self._rss_peak,
            stage_rss_end_bytes=self._rss_end,
            stage_rss_peak_increase_bytes=(max(0, self._rss_peak - self._rss_base)
                if self._rss_peak is not None and self._rss_base is not None else None),
            process_high_water_start_bytes=start,
            process_high_water_end_bytes=end,
            process_high_water_increase_bytes=(max(0, end - start)
                if end is not None and start is not None else None),
            process_high_water_start_raw=self._high_water_start['raw'],
            process_high_water_end_raw=self._high_water_end['raw'],
            process_high_water_raw_unit=self._high_water_end['raw_unit'],
            sampling_interval_seconds=self.interval_seconds,
            sample_count=self._sample_count,
            rss_sample_errors=list(self._sample_errors),
            stage_peak_is_sampled=True,
            process_high_water_is_lifetime=True,
            rss_includes_children=False,
            accelerator_memory_included=False)
