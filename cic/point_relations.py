"""Vectorized ORIGINAL ten relation features; no approximation or new feature.

prepare_points caches each visit's categorical/time attributes and POI weights
once. relation_rows broadcasts the original float64 formulas, then casts into
original float32 valid rows. It allocates no max_points^2 padding. All methods
must use this same constructor when isolating the effects of summarization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True)
class PreparedPoints:
    longitude: np.ndarray
    latitude: np.ndarray
    cos_latitude: np.ndarray
    poi: np.ndarray
    has_poi: np.ndarray
    category: np.ndarray
    time_bin: np.ndarray
    day_type: np.ndarray
    behavior: np.ndarray
    hour: np.ndarray
    poi_idf: np.ndarray
    poi_log_df: np.ndarray

    def __len__(self):
        return len(self.hour)


def prepare_points(trajectories, split_stats, max_points=32):
    """Prepare records or explicit point lists, preserving their visit order.

    TrajectoryRecord inputs apply original _valid_points(max_points) exactly.
    Explicit lists/tuples are treated as already-selected visits, with no
    additional subsampling. The retained pipeline passes raw trajectory records.
    """
    from . import preprocessing as old
    poi_idf = split_stats.get('poi_idf', {})
    poi_df = split_stats.get('poi_df', {})
    output = []
    for trajectory in trajectories:
        points = (old._valid_points(trajectory, max_points)
                  if hasattr(trajectory, 'points') else list(trajectory))
        lon = np.asarray([float(p.lon) for p in points], dtype=np.float64)
        lat = np.asarray([float(p.lat) for p in points], dtype=np.float64)
        output.append(PreparedPoints(
            longitude=lon, latitude=lat,
            cos_latitude=np.asarray([math.cos(math.radians(float(p.lat))) for p in points]),
            poi=np.asarray([p.poi_id for p in points], dtype=object),
            has_poi=np.asarray([bool(p.poi_id) for p in points]),
            category=np.asarray([p.category or 'unknown' for p in points], dtype=object),
            time_bin=np.asarray([old._time_bin_from_hour(int(p.hour)) for p in points], dtype=object),
            day_type=np.asarray([old._day_type_from_timestamp(p.timestamp) for p in points], dtype=object),
            behavior=np.asarray([old.behavior_token_from_point(p) for p in points], dtype=object),
            hour=np.asarray([int(p.hour) for p in points], dtype=np.int64),
            poi_idf=np.asarray([float(poi_idf.get(p.poi_id, 0.0)) for p in points], dtype=np.float64),
            poi_log_df=np.asarray([math.log1p(float(poi_df.get(p.poi_id, 0.0))) for p in points], dtype=np.float64)))
    return output


def relation_rows(left: PreparedPoints, right: PreparedPoints, args):
    """Return the original row-major valid relations, shape [Li*Lj,10]."""
    ni, nj = len(left), len(right)
    rows = np.empty((ni,nj,10), dtype=np.float32)
    if not ni or not nj:
        return rows.reshape(-1,10)
    same_poi = left.has_poi[:,None] & (left.poi[:,None] == right.poi[None,:])
    delta_phi = np.deg2rad(right.latitude[None,:] - left.latitude[:,None])
    delta_lambda = np.deg2rad(right.longitude[None,:] - left.longitude[:,None])
    hav = (np.sin(delta_phi/2.0)**2
           + left.cos_latitude[:,None]*right.cos_latitude[None,:]*np.sin(delta_lambda/2.0)**2)
    dist = 2.0 * 6_371_000.0 * np.arctan2(np.sqrt(hav), np.sqrt(np.maximum(0.0, 1.0-hav)))
    hours = np.abs(left.hour[:,None] - right.hour[None,:])
    rows[:,:,0] = same_poi
    rows[:,:,1] = np.log1p(dist) / math.log1p(max(args.geo_norm,1.0))
    rows[:,:,2] = np.exp(-dist / max(args.geo_sigma,1e-6))
    rows[:,:,3] = left.category[:,None] == right.category[None,:]
    rows[:,:,4] = left.time_bin[:,None] == right.time_bin[None,:]
    rows[:,:,5] = left.day_type[:,None] == right.day_type[None,:]
    rows[:,:,6] = left.behavior[:,None] == right.behavior[None,:]
    rows[:,:,7] = np.minimum(hours,24-hours) / 12.0
    rows[:,:,8] = np.where(same_poi,left.poi_idf[:,None],0.0)
    rows[:,:,9] = np.where(same_poi,left.poi_log_df[:,None],0.0)
    return rows.reshape(-1,10)
