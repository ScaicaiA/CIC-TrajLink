"""Original POI/time/point selection conventions, without identity fields."""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from datetime import datetime,timezone
from typing import Sequence,List,Dict
import json,math,re
import numpy as np

@dataclass(frozen=True)
class TrajectoryPoint:
    poi_id: str
    lon: float
    lat: float
    timestamp: float
    hour: int
    category: str

@dataclass
class TrajectoryRecord:
    key: str
    points: list[TrajectoryPoint]

def load_jsonl(path):
    trajectories=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            points=[TrajectoryPoint(**{k:p[k] for k in TrajectoryPoint.__annotations__}) for p in row['points']]
            if not points:raise ValueError('A trajectory must contain at least one visit')
            trajectories.append(TrajectoryRecord(str(row['key']),points))
    if not trajectories:raise ValueError('Empty input batch')
    if len({t.key for t in trajectories})!=len(trajectories):raise ValueError('Duplicate trajectory keys')
    return trajectories

def _time_bin_from_hour(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 14:
        return "noon"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 24:
        return "evening"
    return "night"


def behavior_token_from_point(point: TrajectoryPoint) -> str:
    """Build a reproducible behavior token from timestamp/hour/category only."""

    try:
        dt = datetime.fromtimestamp(float(point.timestamp), tz=timezone.utc)
        day_type = "weekend" if dt.weekday() >= 5 else "weekday"
    except (OverflowError, OSError, ValueError):
        day_type = "weekday"
    category = str(point.category or "unknown").strip().lower() or "unknown"
    category = re.sub(r"\s+", "_", category)
    return f"{day_type}_{_time_bin_from_hour(int(point.hour))}_{category}"


def compute_relation_split_stats(trajectories: Sequence[TrajectoryRecord]) -> Dict[str, Dict[str, float]]:
    """Anonymous trajectory-level df/idf for POI and behavior tokens."""

    poi_df: Counter = Counter()
    behavior_df: Counter = Counter()
    for traj in trajectories:
        poi_tokens = {p.poi_id for p in traj.points if p.poi_id}
        behavior_tokens = {behavior_token_from_point(p) for p in traj.points}
        poi_df.update(poi_tokens)
        behavior_df.update(behavior_tokens)
    n = len(trajectories)
    poi_idf = {p: float(math.log((n + 1.0) / (df + 1.0)) + 1.0) for p, df in poi_df.items()}
    behavior_idf = {b: float(math.log((n + 1.0) / (df + 1.0)) + 1.0) for b, df in behavior_df.items()}
    return {
        "poi_df": {str(k): float(v) for k, v in poi_df.items()},
        "poi_idf": poi_idf,
        "behavior_df": {str(k): float(v) for k, v in behavior_df.items()},
        "behavior_idf": behavior_idf,
    }


def _day_type_from_timestamp(timestamp: float) -> str:
    try:
        dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        return "weekend" if dt.weekday() >= 5 else "weekday"
    except (OverflowError, OSError, ValueError):
        return "weekday"


def _valid_points(traj: TrajectoryRecord, max_points: int) -> List[TrajectoryPoint]:
    if len(traj.points) <= max_points:
        return list(traj.points)
    # Deterministic spread sampling keeps start/middle/end information.
    idx = np.linspace(0, len(traj.points) - 1, max_points).astype(int)
    return [traj.points[int(i)] for i in idx]
