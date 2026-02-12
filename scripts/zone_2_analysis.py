"""Zone 2 (Below Center) bucket analysis for Elon Tweet Markets.

Implements 7 modeling approaches to predict Zone 2 bucket probabilities,
evaluates via Leave-One-Event-Out CV, and compares against market baseline.

Zone 2 = relative_position 20-40% (below center, not extreme tail).
Market avg price ~7.6%, actual win rate ~7.1% => market OVERPRICES by ~0.4%.
"""

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT / "data" / "analysis" / "bucket_dataset.json"
OUTPUT_PATH = PROJECT / "data" / "analysis" / "zone_2_results.json"

TARGET_ZONE = 2
PROB_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def normalize_probs(probs: list[float]) -> list[float]:
    """Floor at PROB_FLOOR, normalize to sum=1."""
    floored = [max(p, PROB_FLOOR) for p in probs]
    total = sum(floored)
    if total <= 0:
        n = len(floored)
        return [1.0 / n] * n
    return [p / total for p in floored]


def brier_score(probs: list[float], outcomes: list[int]) -> float:
    """Multi-outcome Brier score: mean of (p_i - o_i)^2 across buckets."""
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def zone_brier(probs: list[float], outcomes: list[int], zones: list[int],
               target_zone: int) -> float:
    """Brier score restricted to buckets in the target zone."""
    pairs = [(p, o) for p, o, z in zip(probs, outcomes, zones)
             if z == target_zone]
    if not pairs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = max(min(p, 1.0 - 1e-10), 1e-10)
    return math.log(p / (1.0 - p))


def safe_float(val) -> float | None:
    """Safely convert a value to float, returning None for None/NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------

def extract_events(dataset: dict) -> list[dict]:
    """Extract structured event records from the dataset."""
    events = []
    for evt in dataset["events"]:
        buckets = evt["buckets"]
        if not buckets:
            continue

        market_prices = [b["market_price"] for b in buckets]
        outcomes = [b["is_winner"] for b in buckets]
        zones = [b["zone"] for b in buckets]
        z_scores = [b.get("z_score") for b in buckets]
        lower_bounds = [b["lower_bound"] for b in buckets]
        upper_bounds = [b["upper_bound"] for b in buckets]
        midpoints = [b["midpoint"] for b in buckets]

        # Check if any zone-2 bucket exists
        if TARGET_ZONE not in zones:
            continue

        features = evt.get("features", {})
        market_feats = features.get("market", {})
        temporal_feats = features.get("temporal", {})
        media_feats = features.get("media", {})
        calendar_feats = features.get("calendar", {})

        events.append({
            "slug": evt["event_slug"],
            "market_type": evt.get("market_type"),
            "duration_days": evt.get("duration_days"),
            "tier": evt.get("ground_truth_tier"),
            "xtracker_count": evt.get("xtracker_count"),
            "n_buckets": len(buckets),
            "market_prices": market_prices,
            "outcomes": outcomes,
            "zones": zones,
            "z_scores": z_scores,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "midpoints": midpoints,
            "buckets_raw": buckets,
            # Features
            "crowd_ev": safe_float(market_feats.get("crowd_implied_ev")),
            "crowd_std": safe_float(market_feats.get("crowd_std_dev")),
            "crowd_skewness": safe_float(market_feats.get("crowd_skewness")),
            "crowd_kurtosis": safe_float(market_feats.get("crowd_kurtosis")),
            "distribution_entropy": safe_float(market_feats.get("distribution_entropy")),
            "price_shift_24h": safe_float(market_feats.get("price_shift_24h")),
            "rolling_avg_7d": safe_float(temporal_feats.get("rolling_avg_7d")),
            "trend_7d": safe_float(temporal_feats.get("trend_7d")),
            "rolling_std_7d": safe_float(temporal_feats.get("rolling_std_7d")),
            "elon_musk_vol_7d": safe_float(media_feats.get("elon_musk_vol_7d")),
            "elon_musk_tone_7d": safe_float(media_feats.get("elon_musk_tone_7d")),
            "launches_trailing_7d": safe_float(calendar_feats.get("launches_trailing_7d")),
        })
    return events


# ---------------------------------------------------------------------------
# Approach 1: Negative Binomial from crowd stats
# ---------------------------------------------------------------------------

def approach_negbin(events: list[dict]) -> dict:
    """NegBin distribution fitted from crowd_implied_ev and crowd_std_dev."""
    full_briers = []
    zone_briers = []

    for evt in events:
        crowd_ev = evt["crowd_ev"]
        crowd_std = evt["crowd_std"]

        if crowd_ev is None or crowd_std is None or crowd_std <= 0 or crowd_ev <= 0:
            # Fall back to market prices
            probs = normalize_probs(evt["market_prices"])
        else:
            variance = crowd_std ** 2
            if variance <= crowd_ev:
                # Variance must exceed mean for NegBin; use Poisson fallback
                probs_raw = []
                for lb, ub in zip(evt["lower_bounds"], evt["upper_bounds"]):
                    ub_capped = min(ub, int(crowd_ev * 10))
                    p = stats.poisson.cdf(ub_capped, crowd_ev) - stats.poisson.cdf(lb - 1, crowd_ev)
                    probs_raw.append(max(p, PROB_FLOOR))
                probs = normalize_probs(probs_raw)
            else:
                # NegBin parameters: mean = n*(1-p)/p, var = n*(1-p)/p^2
                # p = mean/var, n = mean^2 / (var - mean)
                p_nb = crowd_ev / variance
                n_nb = (crowd_ev ** 2) / (variance - crowd_ev)

                if n_nb <= 0 or p_nb <= 0 or p_nb >= 1:
                    probs = normalize_probs(evt["market_prices"])
                else:
                    probs_raw = []
                    for lb, ub in zip(evt["lower_bounds"], evt["upper_bounds"]):
                        ub_capped = min(ub, 100000)
                        p_bucket = stats.nbinom.cdf(ub_capped, n_nb, p_nb) - stats.nbinom.cdf(max(lb - 1, 0), n_nb, p_nb)
                        probs_raw.append(max(p_bucket, PROB_FLOOR))
                    probs = normalize_probs(probs_raw)

        full_briers.append(brier_score(probs, evt["outcomes"]))
        zone_briers.append(zone_brier(probs, evt["outcomes"], evt["zones"], TARGET_ZONE))

    valid_zone = [b for b in zone_briers if not math.isnan(b)]
    return {
        "name": "NegBin from crowd stats",
        "full_brier": float(np.mean(full_briers)),
        "zone_brier": float(np.mean(valid_zone)) if valid_zone else None,
        "n_events": len(events),
        "learned": False,
    }


# ---------------------------------------------------------------------------
# Approach 2: Gaussian from crowd stats
# ---------------------------------------------------------------------------

def approach_gaussian(events: list[dict]) -> dict:
    """Normal(crowd_ev, crowd_std) distribution."""
    full_briers = []
    zone_briers = []

    for evt in events:
        crowd_ev = evt["crowd_ev"]
        crowd_std = evt["crowd_std"]

        if crowd_ev is None or crowd_std is None or crowd_std <= 0:
            probs = normalize_probs(evt["market_prices"])
        else:
            dist = stats.norm(loc=crowd_ev, scale=crowd_std)
            probs_raw = []
            for lb, ub in zip(evt["lower_bounds"], evt["upper_bounds"]):
                # For open-ended upper bucket, use 1 - CDF(lower)
                if ub >= 99999:
                    p = 1.0 - dist.cdf(lb - 0.5)
                else:
                    p = dist.cdf(ub + 0.5) - dist.cdf(lb - 0.5)
                probs_raw.append(max(p, PROB_FLOOR))
            probs = normalize_probs(probs_raw)

        full_briers.append(brier_score(probs, evt["outcomes"]))
        zone_briers.append(zone_brier(probs, evt["outcomes"], evt["zones"], TARGET_ZONE))

    valid_zone = [b for b in zone_briers if not math.isnan(b)]
    return {
        "name": "Gaussian from crowd stats",
        "full_brier": float(np.mean(full_briers)),
        "zone_brier": float(np.mean(valid_zone)) if valid_zone else None,
        "n_events": len(events),
        "learned": False,
    }


# ---------------------------------------------------------------------------
# Approach 3: Market baseline
# ---------------------------------------------------------------------------

def approach_market_baseline(events: list[dict]) -> dict:
    """Normalized market prices as-is."""
    full_briers = []
    zone_briers = []

    for evt in events:
        probs = normalize_probs(evt["market_prices"])
        full_briers.append(brier_score(probs, evt["outcomes"]))
        zone_briers.append(zone_brier(probs, evt["outcomes"], evt["zones"], TARGET_ZONE))

    valid_zone = [b for b in zone_briers if not math.isnan(b)]
    return {
        "name": "Market baseline",
        "full_brier": float(np.mean(full_briers)),
        "zone_brier": float(np.mean(valid_zone)) if valid_zone else None,
        "n_events": len(events),
        "learned": False,
    }


# ---------------------------------------------------------------------------
# Approach 4: Platt-calibrated market (LOOCV)
# ---------------------------------------------------------------------------

def approach_platt_calibrated(events: list[dict]) -> dict:
    """Platt scaling on logit(market_price) for Zone 2 buckets, LOOCV."""
    full_briers = []
    zone_briers = []
    baseline_briers = []

    for hold_idx in range(len(events)):
        # Training set: all events except hold_idx
        train_logits = []
        train_labels = []
        for i, evt in enumerate(events):
            if i == hold_idx:
                continue
            for j, z in enumerate(evt["zones"]):
                if z == TARGET_ZONE:
                    mp = evt["market_prices"][j]
                    if mp > PROB_FLOOR:
                        train_logits.append(logit(mp))
                        train_labels.append(evt["outcomes"][j])

        # Fit Platt parameters a, b via max likelihood
        if len(train_logits) < 5 or sum(train_labels) < 1:
            # Not enough data, use baseline
            evt_test = events[hold_idx]
            probs = normalize_probs(evt_test["market_prices"])
            full_briers.append(brier_score(probs, evt_test["outcomes"]))
            zone_briers.append(zone_brier(probs, evt_test["outcomes"],
                                          evt_test["zones"], TARGET_ZONE))
            baseline_probs = normalize_probs(evt_test["market_prices"])
            baseline_briers.append(brier_score(baseline_probs, evt_test["outcomes"]))
            continue

        X = np.array(train_logits)
        y = np.array(train_labels, dtype=float)

        def neg_log_likelihood(params):
            a, b = params
            preds = 1.0 / (1.0 + np.exp(-(a * X + b)))
            preds = np.clip(preds, 1e-10, 1 - 1e-10)
            ll = y * np.log(preds) + (1 - y) * np.log(1 - preds)
            return -np.sum(ll)

        try:
            result = optimize.minimize(neg_log_likelihood, [1.0, 0.0],
                                       method="Nelder-Mead",
                                       options={"maxiter": 2000})
            a_opt, b_opt = result.x
        except Exception:
            a_opt, b_opt = 1.0, 0.0

        # Apply to test event
        evt_test = events[hold_idx]
        adjusted_probs = list(evt_test["market_prices"])  # copy

        for j, z in enumerate(evt_test["zones"]):
            if z == TARGET_ZONE:
                mp = evt_test["market_prices"][j]
                if mp > PROB_FLOOR:
                    adjusted_probs[j] = sigmoid(a_opt * logit(mp) + b_opt)
                else:
                    adjusted_probs[j] = PROB_FLOOR

        probs = normalize_probs(adjusted_probs)
        full_briers.append(brier_score(probs, evt_test["outcomes"]))
        zone_briers.append(zone_brier(probs, evt_test["outcomes"],
                                      evt_test["zones"], TARGET_ZONE))

        baseline_probs = normalize_probs(evt_test["market_prices"])
        baseline_briers.append(brier_score(baseline_probs, evt_test["outcomes"]))

    valid_zone = [b for b in zone_briers if not math.isnan(b)]
    events_beat = sum(1 for f, b in zip(full_briers, baseline_briers) if f < b)
    return {
        "name": "Platt-calibrated market",
        "full_brier": float(np.mean(full_briers)),
        "zone_brier": float(np.mean(valid_zone)) if valid_zone else None,
        "n_events": len(events),
        "events_beating_baseline": events_beat,
        "learned": True,
    }


# ---------------------------------------------------------------------------
# Approach 5: Market + downswing correction (LOOCV)
# ---------------------------------------------------------------------------

def approach_downswing_correction(events: list[dict]) -> dict:
    """Systematic shrinkage of Zone 2 prices, learn optimal shrink_factor."""
    full_briers = []
    zone_briers = []
    baseline_briers = []
    best_shrinks = []

    for hold_idx in range(len(events)):
        # Grid search for optimal shrink on training set
        train_events = [e for i, e in enumerate(events) if i != hold_idx]

        def eval_shrink(shrink):
            total_brier = 0
            for evt in train_events:
                adj = list(evt["market_prices"])
                removed_mass = 0
                non_zone_mass = 0
                for j, z in enumerate(evt["zones"]):
                    if z == TARGET_ZONE:
                        reduction = adj[j] * shrink
                        adj[j] = adj[j] - reduction
                        removed_mass += reduction
                    else:
                        non_zone_mass += adj[j]

                # Redistribute removed mass proportionally to non-zone buckets
                if non_zone_mass > 0 and removed_mass > 0:
                    for j, z in enumerate(evt["zones"]):
                        if z != TARGET_ZONE:
                            adj[j] += removed_mass * (adj[j] / non_zone_mass)

                probs = normalize_probs(adj)
                total_brier += brier_score(probs, evt["outcomes"])
            return total_brier / len(train_events)

        # Search grid
        best_shrink = 0
        best_train_brier = eval_shrink(0)
        for s in np.arange(0.0, 0.50, 0.01):
            tb = eval_shrink(s)
            if tb < best_train_brier:
                best_train_brier = tb
                best_shrink = s

        best_shrinks.append(best_shrink)

        # Apply to test event
        evt_test = events[hold_idx]
        adj = list(evt_test["market_prices"])
        removed_mass = 0
        non_zone_mass = 0
        for j, z in enumerate(evt_test["zones"]):
            if z == TARGET_ZONE:
                reduction = adj[j] * best_shrink
                adj[j] = adj[j] - reduction
                removed_mass += reduction
            else:
                non_zone_mass += adj[j]

        if non_zone_mass > 0 and removed_mass > 0:
            for j, z in enumerate(evt_test["zones"]):
                if z != TARGET_ZONE:
                    adj[j] += removed_mass * (adj[j] / non_zone_mass)

        probs = normalize_probs(adj)
        full_briers.append(brier_score(probs, evt_test["outcomes"]))
        zone_briers.append(zone_brier(probs, evt_test["outcomes"],
                                      evt_test["zones"], TARGET_ZONE))

        baseline_probs = normalize_probs(evt_test["market_prices"])
        baseline_briers.append(brier_score(baseline_probs, evt_test["outcomes"]))

    valid_zone = [b for b in zone_briers if not math.isnan(b)]
    events_beat = sum(1 for f, b in zip(full_briers, baseline_briers) if f < b)
    return {
        "name": "Market + downswing correction",
        "full_brier": float(np.mean(full_briers)),
        "zone_brier": float(np.mean(valid_zone)) if valid_zone else None,
        "n_events": len(events),
        "events_beating_baseline": events_beat,
        "avg_shrink_factor": float(np.mean(best_shrinks)),
        "learned": True,
    }


# ---------------------------------------------------------------------------
# Approach 6: Market + crowd-EV-vs-rolling-avg correction (LOOCV)
# ---------------------------------------------------------------------------

def approach_crowd_vs_rolling(events: list[dict]) -> dict:
    """Conditional Zone 2 adjustment based on crowd EV vs trailing average.

    Logic:
    - If rolling_avg * duration < crowd_ev: crowd is too optimistic (expects more tweets).
      Actual likely lower => Zone 2 (below center) MORE likely => boost Zone 2
    - If rolling_avg * duration > crowd_ev: crowd is too pessimistic.
      Actual likely higher => Zone 2 LESS likely => shrink Zone 2
    - For events without temporal data: use media signals as proxy
    """
    full_briers = []
    zone_briers = []
    baseline_briers = []

    for hold_idx in range(len(events)):
        train_events = [e for i, e in enumerate(events) if i != hold_idx]

        # Learn two parameters: alpha_temporal, alpha_media via grid search
        def eval_params(alpha_t, alpha_m):
            total_brier = 0
            for evt in train_events:
                adj_factor = compute_adj_factor(evt, alpha_t, alpha_m)
                adj = apply_zone_adj(evt, adj_factor)
                probs = normalize_probs(adj)
                total_brier += brier_score(probs, evt["outcomes"])
            return total_brier / len(train_events)

        best_alpha_t, best_alpha_m = 0.0, 0.0
        best_tb = eval_params(0, 0)

        for at in np.arange(-0.3, 0.31, 0.05):
            for am in np.arange(-0.3, 0.31, 0.05):
                tb = eval_params(at, am)
                if tb < best_tb:
                    best_tb = tb
                    best_alpha_t = at
                    best_alpha_m = am

        # Apply to test event
        evt_test = events[hold_idx]
        adj_factor = compute_adj_factor(evt_test, best_alpha_t, best_alpha_m)
        adj = apply_zone_adj(evt_test, adj_factor)
        probs = normalize_probs(adj)

        full_briers.append(brier_score(probs, evt_test["outcomes"]))
        zone_briers.append(zone_brier(probs, evt_test["outcomes"],
                                      evt_test["zones"], TARGET_ZONE))

        baseline_probs = normalize_probs(evt_test["market_prices"])
        baseline_briers.append(brier_score(baseline_probs, evt_test["outcomes"]))

    valid_zone = [b for b in zone_briers if not math.isnan(b)]
    events_beat = sum(1 for f, b in zip(full_briers, baseline_briers) if f < b)
    return {
        "name": "Market + crowd-EV-vs-rolling correction",
        "full_brier": float(np.mean(full_briers)),
        "zone_brier": float(np.mean(valid_zone)) if valid_zone else None,
        "n_events": len(events),
        "events_beating_baseline": events_beat,
        "learned": True,
    }


def compute_adj_factor(evt: dict, alpha_t: float, alpha_m: float) -> float:
    """Compute multiplicative adjustment factor for Zone 2 based on features.

    Positive factor => multiply Zone 2 probs by (1 + factor)
    Negative factor => multiply Zone 2 probs by (1 + factor), shrinking them
    """
    factor = 0.0
    duration = evt.get("duration_days") or 7

    rolling_avg = evt.get("rolling_avg_7d")
    crowd_ev = evt.get("crowd_ev")

    if rolling_avg is not None and crowd_ev is not None and crowd_ev > 0:
        # Expected count from trailing avg
        expected_from_trailing = rolling_avg * duration
        # Relative gap: positive means trailing > crowd (crowd too low)
        # => actual likely higher => Zone 2 (below center) less likely
        relative_gap = (expected_from_trailing - crowd_ev) / crowd_ev
        # Clamp
        relative_gap = max(min(relative_gap, 1.0), -1.0)
        factor += alpha_t * relative_gap
    else:
        # Use media signal as proxy
        elon_vol = evt.get("elon_musk_vol_7d")
        if elon_vol is not None:
            # Higher media vol => more tweeting => less likely in Zone 2 (below center)
            # Normalize: typical vol is ~0.2-0.4, so center at 0.3
            media_signal = -(elon_vol - 0.3)  # negative when high vol
            factor += alpha_m * media_signal

    return factor


def apply_zone_adj(evt: dict, adj_factor: float) -> list[float]:
    """Apply multiplicative adjustment to Zone 2 buckets."""
    adj = list(evt["market_prices"])
    removed_mass = 0
    non_zone_mass = 0

    for j, z in enumerate(evt["zones"]):
        if z == TARGET_ZONE:
            old_p = adj[j]
            new_p = old_p * (1.0 + adj_factor)
            new_p = max(new_p, PROB_FLOOR)
            delta = old_p - new_p
            removed_mass += delta
            adj[j] = new_p
        else:
            non_zone_mass += adj[j]

    # Redistribute delta to other zones
    if non_zone_mass > 0 and abs(removed_mass) > PROB_FLOOR:
        for j, z in enumerate(evt["zones"]):
            if z != TARGET_ZONE:
                adj[j] += removed_mass * (adj[j] / non_zone_mass)
                adj[j] = max(adj[j], PROB_FLOOR)

    return adj


# ---------------------------------------------------------------------------
# Approach 7: Multi-feature adjustment model (LOOCV)
# ---------------------------------------------------------------------------

def approach_multi_feature(events: list[dict]) -> dict:
    """L2-regularized linear model predicting residual from market price."""
    full_briers = []
    zone_briers = []
    baseline_briers = []

    # Pre-compute feature vectors for all zone-2 buckets in each event
    event_features = []
    for evt in events:
        evt_feats = []
        for j, z in enumerate(evt["zones"]):
            if z == TARGET_ZONE:
                feat_vec = build_feature_vector(evt, j)
                evt_feats.append((j, feat_vec))
        event_features.append(evt_feats)

    for hold_idx in range(len(events)):
        # Build training data
        train_X = []
        train_y = []

        for i, evt in enumerate(events):
            if i == hold_idx:
                continue
            for j, feat_vec in event_features[i]:
                if feat_vec is not None:
                    residual = evt["outcomes"][j] - evt["market_prices"][j]
                    train_X.append(feat_vec)
                    train_y.append(residual)

        if len(train_X) < 10:
            # Not enough data
            evt_test = events[hold_idx]
            probs = normalize_probs(evt_test["market_prices"])
            full_briers.append(brier_score(probs, evt_test["outcomes"]))
            zone_briers.append(zone_brier(probs, evt_test["outcomes"],
                                          evt_test["zones"], TARGET_ZONE))
            baseline_probs = normalize_probs(evt_test["market_prices"])
            baseline_briers.append(brier_score(baseline_probs, evt_test["outcomes"]))
            continue

        X = np.array(train_X)
        y = np.array(train_y)

        # Standardize features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_norm = (X - X_mean) / X_std

        # Ridge regression: w = (X^T X + lambda I)^{-1} X^T y
        lam = 1.0  # L2 regularization
        n_feat = X_norm.shape[1]
        try:
            w = np.linalg.solve(
                X_norm.T @ X_norm + lam * np.eye(n_feat),
                X_norm.T @ y
            )
        except np.linalg.LinAlgError:
            w = np.zeros(n_feat)

        # Apply to test event
        evt_test = events[hold_idx]
        adj = list(evt_test["market_prices"])
        removed_mass = 0
        non_zone_mass = 0

        for j, feat_vec in event_features[hold_idx]:
            if feat_vec is not None:
                feat_norm = (np.array(feat_vec) - X_mean) / X_std
                residual_pred = feat_norm @ w
                # Clamp adjustment to prevent wild swings
                residual_pred = max(min(residual_pred, 0.15), -0.15)
                old_p = adj[j]
                new_p = max(old_p + residual_pred, PROB_FLOOR)
                delta = old_p - new_p
                removed_mass += delta
                adj[j] = new_p
            else:
                if evt_test["zones"][j] != TARGET_ZONE:
                    non_zone_mass += adj[j]

        # Recompute non-zone mass after adjustment
        non_zone_mass = sum(adj[j] for j in range(len(adj))
                           if evt_test["zones"][j] != TARGET_ZONE)

        # Redistribute
        if non_zone_mass > 0 and abs(removed_mass) > PROB_FLOOR:
            for j, z in enumerate(evt_test["zones"]):
                if z != TARGET_ZONE:
                    adj[j] += removed_mass * (adj[j] / non_zone_mass)
                    adj[j] = max(adj[j], PROB_FLOOR)

        probs = normalize_probs(adj)
        full_briers.append(brier_score(probs, evt_test["outcomes"]))
        zone_briers.append(zone_brier(probs, evt_test["outcomes"],
                                      evt_test["zones"], TARGET_ZONE))

        baseline_probs = normalize_probs(evt_test["market_prices"])
        baseline_briers.append(brier_score(baseline_probs, evt_test["outcomes"]))

    valid_zone = [b for b in zone_briers if not math.isnan(b)]
    events_beat = sum(1 for f, b in zip(full_briers, baseline_briers) if f < b)
    return {
        "name": "Multi-feature adjustment",
        "full_brier": float(np.mean(full_briers)),
        "zone_brier": float(np.mean(valid_zone)) if valid_zone else None,
        "n_events": len(events),
        "events_beating_baseline": events_beat,
        "learned": True,
    }


def build_feature_vector(evt: dict, bucket_idx: int) -> list[float] | None:
    """Build feature vector for a Zone 2 bucket.

    Features: z_score, market_price, crowd_skewness, crowd_kurtosis,
              price_shift_24h, elon_musk_tone_7d, launches_trailing_7d,
              distribution_entropy
    """
    b = evt["buckets_raw"][bucket_idx]
    z_score = safe_float(b.get("z_score"))
    market_price = b.get("market_price", 0)

    crowd_skewness = evt.get("crowd_skewness")
    crowd_kurtosis = evt.get("crowd_kurtosis")
    price_shift = evt.get("price_shift_24h")
    tone = evt.get("elon_musk_tone_7d")
    launches = evt.get("launches_trailing_7d")
    entropy = evt.get("distribution_entropy")

    # Use 0 for missing values (imputation)
    features = [
        z_score if z_score is not None else 0.0,
        market_price,
        crowd_skewness if crowd_skewness is not None else 0.0,
        crowd_kurtosis if crowd_kurtosis is not None else 0.0,
        price_shift if price_shift is not None else 0.0,
        tone if tone is not None else 0.0,
        launches if launches is not None else 0.0,
        entropy if entropy is not None else 0.0,
    ]

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Zone 2 (Below Center) Bucket Analysis")
    print("=" * 70)

    dataset = load_dataset()
    events = extract_events(dataset)

    print(f"\nLoaded {len(events)} events with Zone 2 buckets")

    # Count zone 2 stats
    n_z2_buckets = 0
    n_z2_winners = 0
    for evt in events:
        for j, z in enumerate(evt["zones"]):
            if z == TARGET_ZONE:
                n_z2_buckets += 1
                if evt["outcomes"][j]:
                    n_z2_winners += 1

    print(f"Zone 2 buckets: {n_z2_buckets}, winners: {n_z2_winners} "
          f"(win rate: {n_z2_winners/n_z2_buckets:.3f})")

    # Tier breakdown
    tier_counts = {}
    for evt in events:
        t = evt["tier"] or "unknown"
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"Tier breakdown: {tier_counts}")

    # Run all approaches
    results = []

    print("\n--- Running Approach 1: NegBin from crowd stats ---")
    r1 = approach_negbin(events)
    results.append(r1)
    print(f"  Full Brier: {r1['full_brier']:.6f}, Zone Brier: {r1['zone_brier']:.6f}")

    print("\n--- Running Approach 2: Gaussian from crowd stats ---")
    r2 = approach_gaussian(events)
    results.append(r2)
    print(f"  Full Brier: {r2['full_brier']:.6f}, Zone Brier: {r2['zone_brier']:.6f}")

    print("\n--- Running Approach 3: Market baseline ---")
    r3 = approach_market_baseline(events)
    results.append(r3)
    print(f"  Full Brier: {r3['full_brier']:.6f}, Zone Brier: {r3['zone_brier']:.6f}")

    print("\n--- Running Approach 4: Platt-calibrated market (LOOCV) ---")
    r4 = approach_platt_calibrated(events)
    results.append(r4)
    print(f"  Full Brier: {r4['full_brier']:.6f}, Zone Brier: {r4['zone_brier']:.6f}")
    print(f"  Events beating baseline: {r4.get('events_beating_baseline', 'N/A')}")

    print("\n--- Running Approach 5: Market + downswing correction (LOOCV) ---")
    r5 = approach_downswing_correction(events)
    results.append(r5)
    print(f"  Full Brier: {r5['full_brier']:.6f}, Zone Brier: {r5['zone_brier']:.6f}")
    print(f"  Events beating baseline: {r5.get('events_beating_baseline', 'N/A')}")
    print(f"  Avg shrink factor: {r5.get('avg_shrink_factor', 'N/A'):.4f}")

    print("\n--- Running Approach 6: Crowd-EV-vs-rolling correction (LOOCV) ---")
    r6 = approach_crowd_vs_rolling(events)
    results.append(r6)
    print(f"  Full Brier: {r6['full_brier']:.6f}, Zone Brier: {r6['zone_brier']:.6f}")
    print(f"  Events beating baseline: {r6.get('events_beating_baseline', 'N/A')}")

    print("\n--- Running Approach 7: Multi-feature adjustment (LOOCV) ---")
    r7 = approach_multi_feature(events)
    results.append(r7)
    print(f"  Full Brier: {r7['full_brier']:.6f}, Zone Brier: {r7['zone_brier']:.6f}")
    print(f"  Events beating baseline: {r7.get('events_beating_baseline', 'N/A')}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline_brier = r3["full_brier"]
    best = min(results, key=lambda r: r["full_brier"])

    header = f"{'Approach':<42} {'Full Brier':>11} {'Zone Brier':>11} {'vs Base':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["full_brier"]):
        delta = r["full_brier"] - baseline_brier
        delta_str = f"{delta:+.6f}"
        zb_str = f"{r['zone_brier']:.6f}" if r['zone_brier'] is not None else "N/A"
        marker = " ***" if r["name"] == best["name"] else ""
        print(f"  {r['name']:<40} {r['full_brier']:.6f}   {zb_str}  {delta_str}{marker}")

    print(f"\nBaseline full Brier: {baseline_brier:.6f}")
    print(f"Best approach: {best['name']} ({best['full_brier']:.6f})")
    print(f"Improvement over baseline: {baseline_brier - best['full_brier']:.6f}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "zone": TARGET_ZONE,
        "n_events": len(events),
        "n_zone_buckets": n_z2_buckets,
        "n_zone_winners": n_z2_winners,
        "zone_win_rate": round(n_z2_winners / n_z2_buckets, 4) if n_z2_buckets > 0 else 0,
        "approaches": results,
        "best_approach": best["name"],
        "best_full_brier": best["full_brier"],
        "baseline_full_brier": baseline_brier,
        "improvement": round(baseline_brier - best["full_brier"], 6),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
