"""Original robust timestamp and points_json parsing, isolated from old runs."""
from typing import Any, Tuple, List, Mapping
from datetime import datetime, timezone
import ast, json, math, re
import numpy as np
import pandas as pd
from .preprocessing import TrajectoryPoint

def parse_timestamp_to_seconds(value: Any) -> Tuple[float, int]:
    """Parse timestamp into Unix seconds and an hour in [0, 23].

    Numeric values are treated as Unix seconds. Strings are parsed with pandas
    so ISO timestamps with offsets are handled correctly.
    """

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0, 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        sec = float(value)
        hour = datetime.fromtimestamp(sec, tz=timezone.utc).hour
        return sec, int(hour)
    text = str(value)
    try:
        iso_text = text.replace("Z", "+00:00")
        dt_fast = datetime.fromisoformat(iso_text)
        if dt_fast.tzinfo is None:
            dt_fast = dt_fast.replace(tzinfo=timezone.utc)
        sec = float(dt_fast.timestamp())
        return sec, int(dt_fast.astimezone(timezone.utc).hour)
    except Exception:
        pass
    try:
        dt = pd.to_datetime(text, utc=True)
        sec = float(dt.timestamp())
        return sec, int(dt.hour)
    except Exception:
        try:
            sec = float(text)
            hour = datetime.fromtimestamp(sec, tz=timezone.utc).hour
            return sec, int(hour)
        except Exception:
            return 0.0, 0

def _normalize_text_v4(value: Any) -> str:
    """Normalize identifiers without making them learnable model inputs."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()

def _safe_float_v4(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default

def _clean_points_json_text_v4(raw: str) -> str:
    """Recover a JSON string from quoted or comma-expanded points_json cells."""

    text = str(raw).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].replace('""', '"')
    elif text.startswith('[{""') or text.startswith('{""'):
        text = text.replace('""', '"')
    return text

def _extract_point_objects_tolerant_v4(text: str) -> List[Mapping[str, Any]]:
    """Best-effort extraction of point dictionaries from a damaged JSON array.

    Some exported rows contain one malformed point fragment inside an otherwise
    usable points_json payload. If parsing the whole JSON fails, this fallback
    extracts each flat ``{...}`` point object and parses the valid ones. Broken
    fragments are skipped; uuid labels are still not used as model inputs.
    """

    objects: List[Mapping[str, Any]] = []
    skipped = 0
    for match in re.finditer(r"\{[^{}]*\}", text):
        chunk = match.group(0)
        try:
            item = json.loads(chunk)
        except Exception:
            try:
                item = ast.literal_eval(chunk)
            except Exception:
                skipped += 1
                continue
        if isinstance(item, Mapping):
            objects.append(item)
    if objects:
        print(
            f"[data-warning] tolerant points_json parser recovered {len(objects)} "
            f"point objects and skipped {skipped} malformed fragments"
        )
    return objects

def _parse_points_json_cell_v4(raw: str, fallback_time: str = "") -> List[TrajectoryPoint]:
    """Parse one top10 points_json field into trajectory points.

    The input file sometimes has one malformed CSV row where pandas sees many
    columns because JSON commas are not protected. The caller recovers the raw
    points_json text first; this function only parses the JSON payload.
    """

    text = _clean_points_json_text_v4(raw)
    if not text:
        return []
    try:
        rows = json.loads(text)
    except Exception:
        try:
            rows = ast.literal_eval(text)
        except Exception as exc:
            rows = _extract_point_objects_tolerant_v4(text)
            if not rows:
                raise ValueError(f"Cannot parse points_json payload prefix={text[:160]!r}") from exc
    if isinstance(rows, Mapping):
        rows = rows.get("points", [rows])
    points: List[TrajectoryPoint] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        poi_id = item.get("poi_id", item.get("venue_id", item.get("venue", item.get("id", ""))))
        lon = item.get("lon", item.get("lng", item.get("longitude", 0.0)))
        lat = item.get("lat", item.get("latitude", 0.0))
        time_value = item.get(
            "timestamp",
            item.get("time", item.get("datetime", item.get("start_time", fallback_time))),
        )
        sec, parsed_hour = parse_timestamp_to_seconds(time_value)
        hour_value = item.get("hour", parsed_hour)
        try:
            hour = int(float(hour_value))
        except Exception:
            hour = int(parsed_hour)
        category = _normalize_text_v4(item.get("category", "unknown")) or "unknown"
        points.append(
            TrajectoryPoint(
                poi_id=_normalize_text_v4(poi_id),
                lon=_safe_float_v4(lon),
                lat=_safe_float_v4(lat),
                timestamp=sec,
                hour=max(0, min(23, hour)),
                category=category,
            )
        )
    return points
