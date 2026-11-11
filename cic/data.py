"""Original identity split, anonymous records, and labels kept separately."""
from collections import defaultdict, Counter
from dataclasses import dataclass
import csv
import random
import sys

import numpy as np

from .csv_payload import _normalize_text_v4, _parse_points_json_cell_v4
from .preprocessing import TrajectoryRecord
from .io import hash_strings, log, sha256


@dataclass
class Split:
    trajectories: list
    labels: np.ndarray
    users: list


def read_csv(path):
    """Match the original v4 row order, parser and duplicated user::key syntax.

    UUIDs are used ONLY for splitting/supervision/evaluation. The returned
    TrajectoryRecord contains an opaque output key and points, no user field.
    """
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    users = defaultdict(list)
    warnings = []
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.reader(f)
        header = [s.strip().lstrip('\ufeff') for s in next(reader)]
        if header[:5] != ['uuid', 'start_time', 'end_time', 'point_count', 'points_json']:
            raise ValueError(f'Unexpected CSV columns: {header[:5]}')
        for record_no, row in enumerate(reader, 2):
            if not row or not any(str(v).strip() for v in row):
                continue
            if len(row) < 5:
                raise ValueError(f'Malformed CSV record {record_no}')
            if len(row) > 5:
                warnings.append(dict(record=record_no, kind='rejoined_csv_columns'))
                row = row[:4] + [','.join(row[4:])]
            uid = _normalize_text_v4(row[0])
            points = _parse_points_json_cell_v4(row[4], fallback_time=row[1])
            if not uid or not points:
                raise ValueError(f'Empty user/trajectory at record {record_no}')
            try:
                declared = int(float(row[3]))
            except ValueError:
                declared = len(points)
            if declared != len(points):
                warnings.append(dict(record=record_no, kind='declared_count_mismatch',
                                     declared=declared, parsed=len(points)))
            values = np.array([[p.lon, p.lat, p.timestamp] for p in points])
            if not np.all(np.isfinite(values)):
                raise ValueError(f'Nonfinite coordinates/time in record {record_no}')
            key = f'{uid}::{uid}::{record_no - 2}'
            users[uid].append(TrajectoryRecord(key, points))
    return dict(users), warnings


def split_users(users, seed=42):
    """Exactly the original Python random shuffle and rounded 60/20/20 split."""
    ids = sorted(users)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    if n < 10:
        raise ValueError('Need at least ten identities for train/validation/test supervision')
    nt, nv = int(round(.6 * n)), int(round(.2 * n))
    return dict(train=sorted(ids[:nt]), val=sorted(ids[nt:nt + nv]), test=sorted(ids[nt + nv:]))


def load_splits(path, seed=42):
    users, warnings = read_csv(path)
    allocation = split_users(users, seed)
    result, info = {}, {}
    for name, identities in allocation.items():
        records, labels = [], []
        for label, uid in enumerate(identities):
            records.extend(users[uid])
            labels.extend([label] * len(users[uid]))
        result[name] = Split(records, np.asarray(labels, np.int64), identities)
        info[name] = dict(users=len(identities), trajectories=len(records),
            points=sum(len(t.points) for t in records),
            keys_sha256=hash_strings(t.key for t in records),
            identities_sha256=hash_strings(identities))
    if any(set(allocation[a]) & set(allocation[b]) for a,b in [('train','val'),('train','test'),('val','test')]):
        raise AssertionError('Identity overlap')
    manifest = dict(csv_sha256=sha256(path), split_seed=seed, splits=info,
        trajectory_count_histogram={str(k):v for k,v in Counter(map(len, users.values())).items()},
        parser_warnings=warnings, identity_disjoint=True)
    log('data_loaded', **info, parser_warnings=len(warnings))
    return result, manifest


def describe_split(split):
    return dict(users=len(split.users), trajectories=len(split.trajectories),
        points=sum(len(t.points) for t in split.trajectories),
        keys_sha256=hash_strings(t.key for t in split.trajectories),
        identities_sha256=hash_strings(split.users))


def select_identities(split, identities):
    """Select whole users, preserving original trajectory order within a user."""
    selected = sorted(identities)
    if not set(selected) <= set(split.users) or len(set(selected)) != len(selected):
        raise ValueError('Unknown or duplicate source identities')
    lookup = {uid: i for i, uid in enumerate(split.users)}
    records, labels = [], []
    for new_label, uid in enumerate(selected):
        ids = np.flatnonzero(split.labels == lookup[uid])
        records.extend(split.trajectories[i] for i in ids)
        labels.extend([new_label] * len(ids))
    return Split(records, np.asarray(labels, np.int64), selected)


def split_source_users(source, seed=42):
    """Fixed 90/10 USER holdout; nearest integer calibration count, halves up.

    The training random seed is deliberately not accepted by this interface.
    At least two identities per role are required; no silent fallback to val.
    """
    identities = sorted(source.users)
    random.Random(seed).shuffle(identities)
    n_calibration = (len(identities) + 5) // 10
    if min(n_calibration, len(identities) - n_calibration) < 2:
        raise ValueError('90/10 source split needs >=2 users in each role')
    calibration = select_identities(source, identities[:n_calibration])
    neural = select_identities(source, identities[n_calibration:])
    if set(neural.users) & set(calibration.users):
        raise AssertionError('Neural/calibration identity overlap')
    if set(neural.users) | set(calibration.users) != set(source.users):
        raise AssertionError('Source identities lost during 90/10 partition')
    return neural, calibration
