"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate.

Model: earners and players tracked move separately, and the rate is their ratio.

  rate_i(t) = 100 * E_i(t) / N(t)

where E_i(t) is the number of tracked players who have earned trophy i and N(t) is
the players Exophase tracks. Both are non-decreasing. E grows as players progress
through the game (fast for early-story trophies, slow for endgame grinds); N grows
as new players join and sync (a launch-week surge, then slowing). Forecasting the
two separately, then taking the ratio, beats forecasting the rate directly, because
a rate can fall while earners rise -- dilution -- which is exactly what launch week
does: from launch night (N=638) to mid-day Nov 19 (N~13,400) most rates fell ~20x
while their earner counts barely moved, except the early trophies (Chapter 1,
fishing), whose earners grew almost as fast as N.

Growth is fitted in log-log time: with tau = hours since the case's first read and
x = log(1 + tau), log N and log E_i are each fit by least squares over the case's
distinct reads (consecutive identical reads carry no information -- Exophase
refreshes its counts in steps, so the page sits unchanged between refreshes).
The fit is anchored at the last distinct read, which is the last refresh the case
saw: the forecast grows N and E from there to the target. The slope for N comes
from the recent window (last 36 h of distinct reads) when there are enough of
them, so the fit follows the current, decelerating growth regime rather than the
launch-night surge. Each trophy's earner slope is written as b_i = b_N * g_i,
where g_i (0..2.5) is how much of the player growth turns into earners for that
trophy: it is fit from the data where the data speaks, and shrunk toward a prior
built from the trophy's last rate (early trophies sit high), its tier and its
description (a Chapter is quick, a 100-count grind is slow). Rare trophies, whose
earner counts are tiny and noisy, are shrunk hardest toward the prior.

When the history is flat (launch night: every read identical), there is no slope
to fit, so the model falls back to launch-week priors: N grows with log-log slope
1.1 -- the slope the 638 -> 3,868 -> ~13,400 jumps imply -- and each trophy's
earners grow at b_N * g_prior. The anchor is still the last distinct read, so
the forecast covers the growth hidden since the last refresh, which is what the
practice answers reward: the 04:30 refresh revealed ~4.5 h of growth at once.

Spread: quantiles are a normal on the logit scale around the median, with sigma
combining trend uncertainty (grows with the square root of the horizon), the fit's
residual noise, and a flat-history term for the refresh-jump risk when the history
carries no variation. The spread is honest about a 4.5-day launch-week horizon.

Standard library only, no network, deterministic: the same case always writes
the same bytes. Runs in milliseconds per case.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim. -P keeps this file's directory OFF sys.path; the sys.path line
below puts it back first, so a module shipped beside this file imports.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (see the module docstring for why each is what it is) ---

# Log-log slope of N(t) when the history is flat: the launch-night jumps
# (638 -> 3,868 -> ~13,400 tracked players) sit on log N ~ 1.1 * log(1 + tau).
FLAT_N_SLOPE = 1.1
# How far back the "recent window" for the slope fit reaches, in hours.
RECENT_HOURS = 36.0
# The recent window is used only when it holds this many distinct reads.
RECENT_MIN_READS = 4
# Clips on the fitted slopes (log-log): N never grows faster than this, and no
# trophy's earners grow faster than this, in the fitted regime.
N_SLOPE_MAX = 1.5
G_MAX = 2.5
# Tier ceilings on the forecast rate (percent): a safety valve, not a prediction.
# By day 7 almost no one has the platinum or 100%, while the opening trophy is
# already at ~100%.
TIER_CEILING = {"platinum": 2.0, "gold": 5.0, "silver": 30.0, "bronze": 99.99}
# Tier fallback rates when a trophy has no usable last rate (percent).
TIER_FALLBACK = {"platinum": 0.1, "gold": 0.5, "silver": 1.0, "bronze": 2.0}
# Spread: trend sigma per square-root day of horizon, residual floor/cap, and
# the flat-history term (refresh-jump risk when the history carries no variation).
TREND_SIGMA_PER_SQRT_DAY = 0.30
RESID_FLOOR = 0.12
RESID_CAP = 0.70
FLAT_SIGMA = 0.90
SIGMA_MIN = 0.30
SIGMA_MAX = 1.60

_COUNT_RE = re.compile(r"(\d+)")


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def ols(xs: list[float], ys: list[float]) -> tuple[float | None, float]:
    """Least-squares slope and residual std of y on x. Slope None when x is constant."""
    n = len(xs)
    if n < 2:
        return None, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    std = math.sqrt(sum(r * r for r in resid) / dof) if n > 2 else 0.0
    return slope, std


def g_prior(trophy: dict, rate_last: float) -> float:
    """Prior for g_i: how much of the player growth turns into earners for this trophy.

    Built from the trophy's last rate (early trophies sit high even on launch
    night), its tier, and its description (an early Chapter is quick, a
    big-count grind is slow). The logistic in the rate is steep around 1.5%:
    on launch night the trophies above ~1.5% were the ones the first players
    earned quickly, while the cluster at ~1.1% was mostly still at its first
    few earners -- so the rate alone must not lift the whole 1.1% cluster.
    Chapter numbers separate the early story (Chapter 1, reached in hours)
    from the late story (Chapter 3+, reached in days).
    """
    g = sigmoid((rate_last - 1.5) / 0.2)
    tier = trophy.get("tier", "bronze")
    if tier == "platinum":
        g = 0.05
    elif tier == "gold":
        g = min(g, 0.30)
    if rate_last >= 90.0:
        g = 1.0
    text = ((trophy.get("description") or "") + " " + (trophy.get("name") or "")).lower()
    chapter = re.search(r"chapter\s+(\d+)", text)
    if chapter:
        g = max(g, {1: 0.85, 2: 0.50, 3: 0.25, 4: 0.15}.get(int(chapter.group(1)), 0.10))
    if "arcade game" in text or "fishing challenge" in text:
        g = max(g, 0.70)
    big = [int(m) for m in _COUNT_RE.findall(text)]
    if big and max(big) >= 50:
        g = min(g, 0.50)
    return min(1.0, max(0.0, g))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0

    # No history at all: fall back to tier priors with a wide spread. This
    # should never happen, but a valid forecast beats a crashed run.
    if not history:
        rows = []
        for trophy in trophies:
            med = TIER_FALLBACK.get(trophy.get("tier", "bronze"), 2.0)
            rows.append({"id": trophy["id"], "q": _quantiles(med, SIGMA_MAX)})
        return {"trophies": rows}

    t0 = parse_time(history[0]["at"])
    target = parse_time(case["target"]) if case.get("target") else parse_time(case["at"])
    tau_target = (target - t0).total_seconds() / 3600.0
    x_target = math.log1p(max(tau_target, 0.0))

    # Distinct reads: collapse consecutive identical reads (same N and rates).
    # Exophase refreshes in steps, so the stretches between refreshes are flat
    # and carry no information about the growth slopes.
    distinct: list[tuple[float, int, list]] = []
    for read in history:
        tau = (parse_time(read["at"]) - t0).total_seconds() / 3600.0
        rates = list(read["rates"])
        n = int(read["players_tracked"])
        if distinct and distinct[-1][1] == n and distinct[-1][2] == rates:
            continue
        distinct.append((tau, n, rates))

    last_tau, last_n, last_rates = distinct[-1]
    x_last = math.log1p(max(last_tau, 0.0))
    dx = x_target - x_last

    # If the target is at or before the last read, there is nothing to grow:
    # forecast the last rates with a narrow spread.
    if dx <= 0:
        rows = []
        for i, trophy in enumerate(trophies):
            rate = last_rates[i] if last_rates[i] is not None else TIER_FALLBACK.get(trophy.get("tier", "bronze"), 2.0)
            rows.append({"id": trophy["id"], "q": _quantiles(rate, SIGMA_MIN)})
        return {"trophies": rows}

    flat = len(distinct) < 3
    if flat:
        b_n = FLAT_N_SLOPE
        n_cap = 30.0
    else:
        # Recent window for the N slope: the current growth regime, not the
        # launch-night surge. Fall back to the whole history when the window
        # is too thin.
        window = [(t, n) for (t, n, _r) in distinct if t >= last_tau - RECENT_HOURS]
        if len(window) < RECENT_MIN_READS:
            window = [(t, n) for (t, n, _r) in distinct]
        xs = [math.log1p(max(t, 0.0)) for (t, _n) in window]
        ys = [math.log(n) for (_t, n) in window]
        slope, _resid = ols(xs, ys)
        b_n = 0.0 if slope is None else min(max(slope, 0.0), N_SLOPE_MAX)
        n_cap = 4.0

    n_target = last_n * math.exp(b_n * dx)
    n_target = min(n_target, last_n * n_cap)

    # Per-trophy earner growth, as a multiple g_i of the player slope.
    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = last_rates[i]
        if rate_last is None:
            # No rate in the last read: walk back to the newest read that has one.
            rate_last = None
            for (_t, _n, rates) in reversed(distinct):
                if rates[i] is not None:
                    rate_last = rates[i]
                    break
            if rate_last is None:
                rate_last = TIER_FALLBACK.get(trophy.get("tier", "bronze"), 2.0)
        e_last = rate_last * last_n / 100.0
        prior = g_prior(trophy, rate_last)

        g = prior
        resid = RESID_FLOOR
        if not flat and b_n >= 0.05:
            # Fit this trophy's earner slope on the recent window and shrink
            # it toward the prior. Rare trophies (few earners) are shrunk
            # hardest: their counts are tiny and noisy. The recent window is
            # used only when it holds enough points, as for N above.
            recent = [(t, _n, rates) for (t, _n, rates) in distinct if t >= last_tau - RECENT_HOURS]
            use = recent if len(recent) >= RECENT_MIN_READS else distinct
            pts = []
            for (t, _n, rates) in use:
                r = rates[i]
                if r is not None and r > 0:
                    pts.append((math.log1p(max(t, 0.0)), math.log(r * _n / 100.0)))
            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                slope_i, resid_i = ols(xs, ys)
                if slope_i is not None:
                    w_data = min(len(pts), 20)
                    w_prior = 3.0 + 30.0 / math.sqrt(max(e_last, 0.0) + 1.0)
                    g_fit = slope_i / b_n
                    g = (w_data * g_fit + w_prior * prior) / (w_data + w_prior)
                    resid = min(max(resid_i, RESID_FLOOR), RESID_CAP)
        if rate_last >= 90.0:
            # Nearly every tracked player already has it: earners track players.
            g = max(g, 1.0)
        g = min(max(g, 0.0), G_MAX)

        e_target = max(e_last, 0.5) * math.exp(b_n * g * dx)
        med = 100.0 * e_target / n_target
        ceiling = TIER_CEILING.get(trophy.get("tier", "bronze"), 99.99)
        med = min(max(med, 0.01), ceiling)

        sigma = math.sqrt((TREND_SIGMA_PER_SQRT_DAY ** 2) * horizon_days
                          + resid ** 2
                          + (FLAT_SIGMA ** 2 if flat else 0.0))
        sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
        rows.append({"id": trophy["id"], "q": _quantiles(med, sigma)})
    return {"trophies": rows}


def _quantiles(median_pct: float, sigma: float) -> list:
    """Five quantiles of a logit-normal around the median, rounded to 4 decimals."""
    lg = logit(median_pct)
    out: list[float | int] = []
    prev = None
    for z in Z:
        q = 100.0 / (1.0 + math.exp(-(lg + z * sigma)))
        q = round4(min(max(q, 0.01), 99.99))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
