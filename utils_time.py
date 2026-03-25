from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd


def parse_ts(x) -> pd.Timestamp:
    """Parse timestamp from string/ts-like object."""
    if isinstance(x, pd.Timestamp):
        return x
    return pd.to_datetime(x)


def floor_to_slot(ts: pd.Timestamp, freq_min: int) -> pd.Timestamp:
    """Floor a timestamp to slot boundary."""
    ts = parse_ts(ts)
    minute = (ts.minute // freq_min) * freq_min
    return ts.replace(minute=minute, second=0, microsecond=0)


def minutes_since_midnight(ts: pd.Timestamp) -> int:
    ts = parse_ts(ts)
    return ts.hour * 60 + ts.minute


def slot_index_in_day(ts: pd.Timestamp, freq_min: int) -> int:
    """0..U-1"""
    m = minutes_since_midnight(ts)
    return int(m // freq_min)


def slots_per_day(freq_min: int) -> int:
    return int(1440 // freq_min)


def add_minutes_to_slot_index(slot_idx: int, add_min: float, freq_min: int) -> int:
    """Advance slot index by travel time (minutes). Use ceil to be conservative."""
    return int(slot_idx + math.ceil(float(add_min) / float(freq_min)))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ts_to_day_str(ts: pd.Timestamp) -> str:
    ts = parse_ts(ts)
    return ts.strftime("%Y-%m-%d")


def ts_to_slot_str(ts: pd.Timestamp, freq_min: int) -> str:
    """Human-friendly slot label: YYYY-MM-DD_HH-MM-SS (floored)."""
    ts = floor_to_slot(parse_ts(ts), freq_min)
    return ts.strftime("%Y-%m-%d_%H-%M-%S")