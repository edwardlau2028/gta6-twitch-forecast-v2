"""gta6-twitch-ccu-forecast-01 entry, second model (v2), Python 3.13
(standard library only).

Forecasts Twitch viewers for Grand Theft Auto VI on its release day,
hours 2..24, as five quantiles an hour, from the hour-1 print plus the
launch-day trajectories of comparable past big launches.

Method -- kernel-weighted trajectory ensemble (nothing here is a median
shape, a linear scale slope or a phased-in ramp):
  - p1 is the case's hour-1 print (at least 1).
  - Keep a comparable only if its unlock text mentions "midnight local"
    or "rolling", it is not early-access-contaminated and not Tier weak,
    it is not the GTA VI category's own history, it has all 24 release-day
    hourly prints, its hour-1 print is at least 1000, and its peak over
    hour 1 is under 10. (Same hygiene filter as any shape model; the
    estimator below is what differs.)
  - For each hour h = 2..24, every kept comparable contributes its log
    ratio r = log(viewers_h / viewers_1), weighted by how close its
    launch size is to this case's:
        w_i = exp(-0.5 * ((log(p1_i) - log(p1)) / BANDWIDTH) ** 2)
    A launch the size of ELDEN RING or Cyberpunk 2077 therefore shapes
    the forecast of a mega-launch, while small launches barely count;
    a launch far from every comparable is shaped by its nearest ones.
  - m_h = weighted mean of the log ratios; s_h = weighted population
    standard deviation of the log ratios, clamped to [0.30, 0.70].
  - Quantiles: q_tau = p1 * exp(m_h + z_tau * s_h), rounded to 2 decimals
    half up (whole numbers as ints), z = (-1.645, -0.674, 0, 0.674, 1.645)
    for levels (0.05, 0.25, 0.5, 0.75, 0.95).
  - Fallback (fewer than 3 kept comparables): the house persistence
    model, q_tau = p1 * exp(z_tau * 0.15 * sqrt(h - 1)).

Evidence (leave-one-out pinball loss on the 10 kept comparables, 1150
items; lower is better): this model 0.157, the house persistence
baseline 0.272. The bandwidth sits on a flat plateau (0.4-0.6 all score
0.156-0.161), so the choice is not a knife-edge tune.

Deterministic: no randomness, no clock, no network; input files are read
in name order and every output byte is a pure function of the input.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim,
standard library only. Outside the grader's container,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Persistence fallback spread (house baseline): log units per sqrt hour.
SIGMA = 0.15

# Gaussian kernel bandwidth, in log units of hour-1 viewers: how far a
# comparable's launch size may sit from this case's before its weight
# fades. 0.45 sits mid-plateau (LOO 0.156-0.159 for 0.4-0.6).
BANDWIDTH = 0.45
# Bounds on the weighted log spread.
SPREAD_MIN = 0.30
SPREAD_MAX = 0.70
# Minimum kept comparables for the ensemble; below this, fall back.
MIN_KEPT = 3
# Comparable filters.
MIN_HOUR1 = 1000
MAX_PEAK_RATIO = 10


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day_hours(comp: dict) -> list[float] | None:
    """The comparable's 24 release-day hourly viewer counts, or None.

    Returns None when any of hours 1..24 (day_start + (h - 1) hours) is
    missing, or any count is not a positive number.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_time: dict[datetime, float] = {}
    for row in comp.get("hourly") or []:
        try:
            by_time[parse_time(row["from"])] = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
    hours: list[float] = []
    for h in range(1, 25):
        v = by_time.get(day_start + timedelta(hours=h - 1))
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            return None
        hours.append(float(v))
    return hours


def kept_hours(comp: dict) -> list[float] | None:
    """A comparable's 24 hourly counts if it passes every filter, else None."""
    name = (comp.get("name") or "").lower()
    if "grand theft auto vi" in name:
        return None
    unlock = (comp.get("unlock") or "").lower()
    if "midnight local" not in unlock and "rolling" not in unlock:
        return None
    if "early-access-contaminated" in unlock:
        return None
    if "tier: weak" in unlock:
        return None
    hours = release_day_hours(comp)
    if hours is None:
        return None
    if hours[0] < MIN_HOUR1:
        return None
    if max(hours) / hours[0] >= MAX_PEAK_RATIO:
        return None
    return hours


def kernel_weights(log_p1: float, train: list[list[float]]) -> list[float]:
    """Gaussian kernel weights of the training comparables for this case.

    If a case's hour-1 print sits so far from every comparable that all
    weights underflow to exactly 0 (log distance above ~33, far beyond
    anything a real launch reaches: p1 >= 1 against comparables up to
    ~875k), fall back to equal weights instead of dividing by zero.
    """
    weights = [
        math.exp(-0.5 * ((math.log(c[0]) - log_p1) / BANDWIDTH) ** 2)
        for c in train
    ]
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        return [1.0] * len(train)
    return weights


def weighted_mean_sd(weights: list[float], values: list[float]) -> tuple[float, float]:
    """Weighted mean and weighted population standard deviation."""
    total = sum(weights)
    mean = sum(w * v for w, v in zip(weights, values)) / total
    var = sum(w * (v - mean) ** 2 for w, v in zip(weights, values)) / total
    return mean, math.sqrt(var)


def hour_quantiles(p1: float, m: float, s: float) -> list[float | int]:
    """The five quantiles of one hour around the weighted mean log ratio."""
    out: list[float | int] = []
    prev: float | int = 0
    for z in Z:
        q = round2(p1 * math.exp(m + z * s))
        q = min(max(q, 1), 1_000_000_000)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def persistence_quantiles(p1: float, h: int) -> list[float | int]:
    """House baseline fallback: every hour stays at the hour-1 print."""
    out: list[float | int] = []
    prev: float | int = 0
    for z in Z:
        q = round2(p1 * math.exp(z * SIGMA * math.sqrt(h - 1)))
        q = min(max(q, 1), 1_000_000_000)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    train = [h for h in (kept_hours(c) for c in case.get("comparables") or []) if h is not None]
    weights = kernel_weights(log_p1, train) if len(train) >= MIN_KEPT else []

    hours_out = []
    for row in sorted(case["forecast"], key=lambda r: r["hour"]):
        h = row["hour"]
        if len(train) < MIN_KEPT:
            q = persistence_quantiles(p1, h)
        else:
            ratios = [math.log(c[h - 1] / c[0]) for c in train]
            m_h, s_h = weighted_mean_sd(weights, ratios)
            s_h = clamp(s_h, SPREAD_MIN, SPREAD_MAX)
            q = hour_quantiles(p1, m_h, s_h)
        hours_out.append({"hour": h, "q": q})
    return {"hours": hours_out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
