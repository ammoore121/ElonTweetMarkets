"""Zone 3 (center bucket) analysis: 7 approaches to beat market pricing.

Zone 3 = 40-60% position buckets (center of distribution, where peak is).
Market avg price ~8.3%, actual win rate ~9.0% => market underprices center by +0.6%.

Evaluates 7 approaches via LOOCV, outputs results to data/analysis/zone_3_results.json.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit, logit  # sigmoid and logit

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
DATASET = PROJECT / "data" / "analysis" / "bucket_dataset.json"
OUTPUT = PROJECT / "data" / "analysis" / "zone_3_results.json"

TARGET_ZONE = 3
PROB_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset():
    """Load bucket dataset, return list of event dicts."""
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)
    return data["events"]


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    """Floor at PROB_FLOOR and normalize to sum=1."""
    probs = np.maximum(probs, PROB_FLOOR)
    return probs / probs.sum()


def brier_score(probs: np.ndarray, actuals: np.ndarray) -> float:
    """Brier score = mean((pred - actual)^2) summed across buckets."""
    return float(np.sum((probs - actuals) ** 2))


def event_market_probs(event: dict) -> np.ndarray:
    """Get normalized market prices for all buckets in an event."""
    return np.array([b["market_price"] for b in event["buckets"]])


def event_actuals(event: dict) -> np.ndarray:
    """Get binary actual outcomes for all buckets."""
    return np.array([b["is_winner"] for b in event["buckets"]], dtype=float)


def event_zones(event: dict) -> np.ndarray:
    """Get zone assignments for each bucket."""
    return np.array([b["zone"] for b in event["buckets"]])


def event_z_scores(event: dict) -> np.ndarray:
    """Get z-scores for each bucket (distance from crowd EV in std units)."""
    return np.array([
        b["z_score"] if b["z_score"] is not None else 0.0
        for b in event["buckets"]
    ])


def has_temporal(event: dict) -> bool:
    """Check if event has temporal features (gold tier)."""
    t = event.get("features", {}).get("temporal", {})
    return t.get("rolling_avg_7d") is not None


def safe_get(d: dict, *keys, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d if d is not None else default


# ---------------------------------------------------------------------------
# Approach 1: Negative Binomial from crowd stats
# ---------------------------------------------------------------------------

def approach_negbin(event: dict) -> np.ndarray:
    """NegBin distribution from crowd_implied_ev and crowd_std_dev."""
    market = event["features"].get("market", {})
    crowd_ev = market.get("crowd_implied_ev")
    crowd_std = market.get("crowd_std_dev")

    if crowd_ev is None or crowd_std is None or crowd_std <= 0 or crowd_ev <= 0:
        # Fallback to market prices
        return event_market_probs(event)

    mean = crowd_ev
    var = crowd_std ** 2

    if var <= mean:
        # NegBin requires var > mean; use Poisson as fallback
        probs = []
        for b in event["buckets"]:
            lower = max(b["lower_bound"], 0)
            upper = b["upper_bound"]
            if upper >= 99999:
                # P(X >= lower)
                p = float(1.0 - stats.poisson.cdf(lower - 1, mean))
            else:
                p = float(stats.poisson.cdf(upper, mean) - (stats.poisson.cdf(lower - 1, mean) if lower > 0 else 0.0))
            probs.append(max(p, PROB_FLOOR))
        return normalize_probs(np.array(probs))

    # scipy.stats.nbinom(n, p): mean = n*(1-p)/p, var = n*(1-p)/p^2
    # Given desired mean and var:
    #   p_scipy = n / (n + mean)  and  n = mean^2 / (var - mean)
    n_scipy = mean * mean / (var - mean)
    p_scipy = n_scipy / (n_scipy + mean)

    if n_scipy <= 0 or p_scipy <= 0 or p_scipy >= 1:
        return event_market_probs(event)

    dist = stats.nbinom(n_scipy, p_scipy)

    probs = []
    for b in event["buckets"]:
        lower = max(b["lower_bound"], 0)
        upper = b["upper_bound"]
        if upper >= 99999:
            p = float(1.0 - dist.cdf(lower - 1))
        else:
            p = float(dist.cdf(upper) - (dist.cdf(lower - 1) if lower > 0 else 0.0))
        probs.append(max(p, PROB_FLOOR))

    return normalize_probs(np.array(probs))


# ---------------------------------------------------------------------------
# Approach 2: Gaussian from crowd stats
# ---------------------------------------------------------------------------

def approach_gaussian(event: dict) -> np.ndarray:
    """Gaussian(crowd_implied_ev, crowd_std_dev) CDF for bucket probs."""
    market = event["features"].get("market", {})
    crowd_ev = market.get("crowd_implied_ev")
    crowd_std = market.get("crowd_std_dev")

    if crowd_ev is None or crowd_std is None or crowd_std <= 0:
        return event_market_probs(event)

    dist = stats.norm(crowd_ev, crowd_std)
    probs = []
    for b in event["buckets"]:
        lower = b["lower_bound"]
        upper = b["upper_bound"]
        if upper >= 99999:
            # Open-ended top bucket: P(X >= lower - 0.5)
            p = float(1.0 - dist.cdf(lower - 0.5))
        elif lower <= 0:
            # Bottom bucket includes 0: P(X <= upper + 0.5)
            p = float(dist.cdf(upper + 0.5))
        else:
            # Normal bucket: P(lower - 0.5 < X <= upper + 0.5)
            p = float(dist.cdf(upper + 0.5) - dist.cdf(lower - 0.5))
        probs.append(max(p, PROB_FLOOR))

    return normalize_probs(np.array(probs))


# ---------------------------------------------------------------------------
# Approach 3: Market baseline (normalized market prices)
# ---------------------------------------------------------------------------

def approach_market(event: dict) -> np.ndarray:
    """Just use normalized market prices. The baseline to beat."""
    return normalize_probs(event_market_probs(event))


# ---------------------------------------------------------------------------
# Approach 4: Platt-calibrated market
# ---------------------------------------------------------------------------

def platt_transform(market_probs: np.ndarray, zones: np.ndarray,
                    a: float, b: float, target_zone: int) -> np.ndarray:
    """Apply Platt scaling to a specific zone's buckets."""
    probs = market_probs.copy()
    mask = zones == target_zone
    if mask.any():
        p_zone = np.clip(probs[mask], 1e-8, 1 - 1e-8)
        logit_p = np.log(p_zone / (1 - p_zone))
        calibrated = expit(a * logit_p + b)
        probs[mask] = calibrated
    return normalize_probs(probs)


def approach_platt_loocv(events: list[dict], test_idx: int) -> tuple[np.ndarray, dict]:
    """LOOCV: learn Platt params from all events except test_idx, predict test."""
    test_event = events[test_idx]
    train_events = [e for i, e in enumerate(events) if i != test_idx]

    # Collect zone-3 bucket data from training events
    train_market_p = []
    train_actuals = []
    for e in train_events:
        mp = event_market_probs(e)
        ac = event_actuals(e)
        zones = event_zones(e)
        mask = zones == TARGET_ZONE
        if mask.any():
            train_market_p.extend(mp[mask].tolist())
            train_actuals.extend(ac[mask].tolist())

    train_market_p = np.array(train_market_p)
    train_actuals = np.array(train_actuals)

    if len(train_market_p) < 5:
        return normalize_probs(event_market_probs(test_event)), {"a": 1.0, "b": 0.0}

    # Fit Platt scaling: minimize log-loss on training data
    def neg_log_likelihood(params):
        a, b = params
        p_clipped = np.clip(train_market_p, 1e-8, 1 - 1e-8)
        logit_p = np.log(p_clipped / (1 - p_clipped))
        calibrated = expit(a * logit_p + b)
        calibrated = np.clip(calibrated, 1e-8, 1 - 1e-8)
        ll = train_actuals * np.log(calibrated) + (1 - train_actuals) * np.log(1 - calibrated)
        return -ll.sum()

    result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method="Nelder-Mead",
                      options={"maxiter": 1000})
    a_opt, b_opt = result.x

    # Apply to test event
    test_probs = event_market_probs(test_event)
    test_zones = event_zones(test_event)
    pred = platt_transform(test_probs, test_zones, a_opt, b_opt, TARGET_ZONE)

    return pred, {"a": round(a_opt, 4), "b": round(b_opt, 4)}


# ---------------------------------------------------------------------------
# Approach 5: Market + center-boosting sharpening
# ---------------------------------------------------------------------------

def sharpen_transform(market_probs: np.ndarray, alpha: float) -> np.ndarray:
    """Raise all probabilities to power alpha, then renormalize.
    alpha < 1 boosts higher-prob buckets (center), alpha > 1 flattens."""
    probs = np.maximum(market_probs, PROB_FLOOR)
    sharpened = probs ** alpha
    return normalize_probs(sharpened)


def approach_sharpen_loocv(events: list[dict], test_idx: int) -> tuple[np.ndarray, dict]:
    """LOOCV: learn optimal alpha from training events."""
    test_event = events[test_idx]
    train_events = [e for i, e in enumerate(events) if i != test_idx]

    # Find alpha that minimizes total Brier on training events
    def total_brier(alpha):
        total = 0.0
        for e in train_events:
            mp = event_market_probs(e)
            ac = event_actuals(e)
            pred = sharpen_transform(mp, alpha)
            total += brier_score(pred, ac)
        return total

    result = minimize_scalar(total_brier, bounds=(0.3, 3.0), method="bounded",
                             options={"maxiter": 200})
    alpha_opt = result.x

    pred = sharpen_transform(event_market_probs(test_event), alpha_opt)
    return pred, {"alpha": round(alpha_opt, 4)}


# ---------------------------------------------------------------------------
# Approach 6: Market + mean-reversion centering
# ---------------------------------------------------------------------------

def approach_mean_reversion_loocv(events: list[dict], test_idx: int) -> tuple[np.ndarray, dict]:
    """When temporal features exist, shift probability toward model EV.
    For non-gold events, use crowd_skewness to de-skew toward center."""
    test_event = events[test_idx]
    train_events = [e for i, e in enumerate(events) if i != test_idx]

    # Learn 3 params: boost_alpha (how much to boost near target),
    # temporal_blend (how much to trust rolling avg over crowd),
    # skew_dampening (how much to counter skewness)
    def total_brier(params):
        boost_alpha, temporal_blend, skew_damp = params
        total = 0.0
        for e in train_events:
            pred = _apply_mean_reversion(e, boost_alpha, temporal_blend, skew_damp)
            ac = event_actuals(e)
            total += brier_score(pred, ac)
        return total

    result = minimize(total_brier, x0=[0.1, 0.1, 0.01], method="Nelder-Mead",
                      bounds=None, options={"maxiter": 500})
    alpha_opt, blend_opt, skew_opt = result.x

    pred = _apply_mean_reversion(test_event, alpha_opt, blend_opt, skew_opt)
    return pred, {
        "boost_alpha": round(alpha_opt, 4),
        "temporal_blend": round(blend_opt, 4),
        "skew_dampening": round(skew_opt, 4),
    }


def _apply_mean_reversion(event: dict, boost_alpha: float, temporal_blend: float,
                          skew_dampening: float) -> np.ndarray:
    """Apply mean-reversion logic to an event's bucket probabilities.

    boost_alpha: strength of Gaussian boost near target EV
    temporal_blend: how much to shift target toward model EV (when temporal data available)
    skew_dampening: how much to counter market skewness
    """
    market = event["features"].get("market", {})
    crowd_ev = market.get("crowd_implied_ev")
    crowd_std = market.get("crowd_std_dev")
    crowd_skew = market.get("crowd_skewness")

    mp = event_market_probs(event)

    if crowd_ev is None or crowd_std is None or crowd_std <= 0:
        return normalize_probs(mp)

    # Determine target EV
    target_ev = crowd_ev  # default: crowd EV is the target

    temporal = event["features"].get("temporal", {})
    rolling_avg = temporal.get("rolling_avg_7d")
    duration = event.get("duration_days", 7)

    if rolling_avg is not None and rolling_avg > 0:
        model_ev = rolling_avg * duration
        # Blend model EV with crowd EV
        target_ev = crowd_ev + temporal_blend * (model_ev - crowd_ev)

    # De-skew: counter the market's skewness
    if crowd_skew is not None and abs(crowd_skew) > 0:
        skew_shift = -skew_dampening * crowd_skew * crowd_std
        target_ev += skew_shift

    # Boost buckets near target_ev using Gaussian kernel
    midpoints = np.array([b["midpoint"] for b in event["buckets"]])
    distances = np.abs(midpoints - target_ev) / crowd_std
    boost = np.exp(-0.5 * distances ** 2)
    probs = mp * (1 + abs(boost_alpha) * boost)

    return normalize_probs(probs)


# ---------------------------------------------------------------------------
# Approach 7: Multi-feature center model
# ---------------------------------------------------------------------------

def approach_multifeature_loocv(events: list[dict], test_idx: int) -> tuple[np.ndarray, dict]:
    """Per-bucket model using features: |z_score|, market_price, crowd_kurtosis,
    distribution_entropy, elon_musk_vol_delta, price_shift_24h.

    Learn how to adjust zone-3 probabilities based on event-level features.
    """
    test_event = events[test_idx]
    train_events = [e for i, e in enumerate(events) if i != test_idx]

    # Build per-bucket training data for zone 3
    X_train = []
    y_train = []
    p_market_train = []

    for e in train_events:
        market_feats = e["features"].get("market", {})
        media_feats = e["features"].get("media", {})
        zones = event_zones(e)
        z_scores = event_z_scores(e)
        mp = event_market_probs(e)
        ac = event_actuals(e)

        kurtosis = market_feats.get("crowd_kurtosis", 0) or 0
        entropy = market_feats.get("distribution_entropy", 0) or 0
        vol_delta = media_feats.get("elon_musk_vol_delta", 0) or 0
        price_shift = market_feats.get("price_shift_24h", 0) or 0

        for j in range(len(zones)):
            if zones[j] == TARGET_ZONE:
                X_train.append([
                    abs(z_scores[j]),          # distance from center
                    mp[j],                     # market price
                    kurtosis,                  # peakedness
                    entropy,                   # spread
                    vol_delta,                 # news activity change
                    price_shift,               # recent price movement
                ])
                y_train.append(ac[j])
                p_market_train.append(mp[j])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    p_market_train = np.array(p_market_train)

    if len(X_train) < 10:
        return normalize_probs(event_market_probs(test_event)), {"weights": "fallback"}

    # Standardize features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8

    X_norm = (X_train - X_mean) / X_std

    # Learn logistic regression weights: P(win) = sigmoid(w . x + b)
    # Use market_price logit as baseline, learn adjustments
    def neg_log_likelihood(params):
        w = params[:-1]
        bias = params[-1]
        # Logit of market price as anchor
        p_clipped = np.clip(p_market_train, 1e-8, 1 - 1e-8)
        base_logit = np.log(p_clipped / (1 - p_clipped))
        # Adjustment
        adjustment = X_norm @ w + bias
        logit_pred = base_logit + adjustment
        pred = expit(logit_pred)
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        ll = y_train * np.log(pred) + (1 - y_train) * np.log(1 - pred)
        # L2 regularization to prevent overfitting on small dataset
        reg = 0.1 * np.sum(w ** 2)
        return -ll.sum() + reg

    n_features = X_train.shape[1]
    x0 = np.zeros(n_features + 1)
    result = minimize(neg_log_likelihood, x0=x0, method="Nelder-Mead",
                      options={"maxiter": 2000})
    w_opt = result.x[:-1]
    b_opt = result.x[-1]

    # Apply to test event
    test_market_feats = test_event["features"].get("market", {})
    test_media_feats = test_event["features"].get("media", {})
    test_zones = event_zones(test_event)
    test_z_scores = event_z_scores(test_event)
    test_mp = event_market_probs(test_event)

    kurtosis = test_market_feats.get("crowd_kurtosis", 0) or 0
    entropy = test_market_feats.get("distribution_entropy", 0) or 0
    vol_delta = test_media_feats.get("elon_musk_vol_delta", 0) or 0
    price_shift = test_market_feats.get("price_shift_24h", 0) or 0

    probs = test_mp.copy()
    for j in range(len(test_zones)):
        if test_zones[j] == TARGET_ZONE:
            x_j = np.array([
                abs(test_z_scores[j]),
                test_mp[j],
                kurtosis,
                entropy,
                vol_delta,
                price_shift,
            ])
            x_norm_j = (x_j - X_mean) / X_std
            p_clipped = np.clip(test_mp[j], 1e-8, 1 - 1e-8)
            base_logit = np.log(p_clipped / (1 - p_clipped))
            adjustment = x_norm_j @ w_opt + b_opt
            probs[j] = expit(base_logit + adjustment)

    return normalize_probs(probs), {
        "weights": [round(w, 4) for w in w_opt],
        "bias": round(b_opt, 4),
        "features": ["|z_score|", "market_price", "kurtosis", "entropy", "vol_delta", "price_shift_24h"],
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_approaches(events: list[dict]):
    """Run all 7 approaches with LOOCV, compute Brier scores."""
    n = len(events)

    print(f"Evaluating {n} events...")
    print(f"Zone {TARGET_ZONE} buckets: {sum(1 for e in events for b in e['buckets'] if b['zone'] == TARGET_ZONE)}")
    print(f"Zone {TARGET_ZONE} winners: {sum(1 for e in events for b in e['buckets'] if b['zone'] == TARGET_ZONE and b['is_winner'])}")
    print()

    # Storage for per-event Brier scores
    results = {}

    # --- Approaches 1-3: no LOOCV needed (stateless) ---
    for name, fn in [
        ("NegBin_crowd", approach_negbin),
        ("Gaussian_crowd", approach_gaussian),
        ("Market_baseline", approach_market),
    ]:
        full_briers = []
        zone_briers = []
        for i, event in enumerate(events):
            pred = fn(event)
            ac = event_actuals(event)
            zones = event_zones(event)

            full_briers.append(brier_score(pred, ac))

            mask = zones == TARGET_ZONE
            if mask.any():
                zone_briers.append(brier_score(pred[mask], ac[mask]))

        results[name] = {
            "full_event_briers": full_briers,
            "zone_only_briers": zone_briers,
            "params": {},
        }
        avg_full = np.mean(full_briers)
        avg_zone = np.mean(zone_briers) if zone_briers else float("nan")
        print(f"  {name:30s}  full={avg_full:.6f}  zone3={avg_zone:.6f}")

    # --- Approaches 4-7: require LOOCV ---
    # Approach 4: Platt-calibrated
    print("\n  Running LOOCV approaches...")

    for name, fn in [
        ("Platt_calibrated", approach_platt_loocv),
        ("Sharpen_center", approach_sharpen_loocv),
        ("Mean_reversion", approach_mean_reversion_loocv),
        ("Multi_feature", approach_multifeature_loocv),
    ]:
        full_briers = []
        zone_briers = []
        all_params = []

        for i in range(n):
            if i % 20 == 0:
                print(f"    {name}: event {i}/{n}...")
            pred, params = fn(events, i)
            ac = event_actuals(events[i])
            zones = event_zones(events[i])

            full_briers.append(brier_score(pred, ac))

            mask = zones == TARGET_ZONE
            if mask.any():
                zone_briers.append(brier_score(pred[mask], ac[mask]))

            all_params.append(params)

        # Aggregate params (mean of numeric params)
        avg_params = {}
        if all_params and isinstance(all_params[0], dict):
            for key in all_params[0]:
                vals = [p.get(key) for p in all_params if isinstance(p.get(key), (int, float))]
                if vals:
                    avg_params[key] = round(np.mean(vals), 4)
                elif all_params[0].get(key) is not None:
                    avg_params[key] = all_params[0][key]

        results[name] = {
            "full_event_briers": full_briers,
            "zone_only_briers": zone_briers,
            "params": avg_params,
        }
        avg_full = np.mean(full_briers)
        avg_zone = np.mean(zone_briers) if zone_briers else float("nan")
        print(f"  {name:30s}  full={avg_full:.6f}  zone3={avg_zone:.6f}")

    return results


def compile_output(events: list[dict], results: dict) -> dict:
    """Create final output JSON."""
    n_zone_buckets = sum(1 for e in events for b in e["buckets"] if b["zone"] == TARGET_ZONE)
    n_zone_winners = sum(1 for e in events for b in e["buckets"]
                         if b["zone"] == TARGET_ZONE and b["is_winner"])

    baseline_full = np.mean(results["Market_baseline"]["full_event_briers"])
    baseline_zone = np.mean(results["Market_baseline"]["zone_only_briers"])

    approaches = []
    for name, data in results.items():
        avg_full = np.mean(data["full_event_briers"])
        avg_zone = np.mean(data["zone_only_briers"]) if data["zone_only_briers"] else float("nan")

        # Count events where this approach beats baseline
        baseline_briers = results["Market_baseline"]["full_event_briers"]
        n_beat = sum(1 for a_b, m_b in zip(data["full_event_briers"], baseline_briers)
                     if a_b < m_b)

        # Zone-only: events beating baseline
        baseline_zone_briers = results["Market_baseline"]["zone_only_briers"]
        n_beat_zone = sum(1 for a_b, m_b in zip(data["zone_only_briers"], baseline_zone_briers)
                         if a_b < m_b)

        approaches.append({
            "name": name,
            "full_event_brier": round(avg_full, 6),
            "zone_only_brier": round(avg_zone, 6),
            "events_beating_baseline_full": n_beat,
            "events_beating_baseline_zone": n_beat_zone,
            "improvement_vs_baseline_full_pct": round(100 * (baseline_full - avg_full) / baseline_full, 2),
            "improvement_vs_baseline_zone_pct": round(100 * (baseline_zone - avg_zone) / baseline_zone, 2) if not np.isnan(avg_zone) else None,
            "params": data["params"],
        })

    # Sort by full_event_brier
    approaches.sort(key=lambda x: x["full_event_brier"])

    best = approaches[0]

    output = {
        "zone": TARGET_ZONE,
        "n_events": len(events),
        "n_zone_buckets": n_zone_buckets,
        "n_zone_winners": n_zone_winners,
        "zone_win_rate": round(n_zone_winners / n_zone_buckets, 4) if n_zone_buckets > 0 else 0,
        "zone_avg_market_price": round(
            np.mean([b["market_price"] for e in events for b in e["buckets"] if b["zone"] == TARGET_ZONE]),
            4
        ),
        "approaches": approaches,
        "best_approach": best["name"],
        "best_full_brier": best["full_event_brier"],
        "baseline_full_brier": round(baseline_full, 6),
        "baseline_zone_brier": round(baseline_zone, 6),
    }

    return output


def main():
    events = load_dataset()
    print(f"Loaded {len(events)} events from {DATASET}")

    # Zone 3 stats
    z3_buckets = sum(1 for e in events for b in e["buckets"] if b["zone"] == TARGET_ZONE)
    z3_winners = sum(1 for e in events for b in e["buckets"]
                     if b["zone"] == TARGET_ZONE and b["is_winner"])
    gold_events = sum(1 for e in events if has_temporal(e))
    print(f"Zone 3: {z3_buckets} buckets, {z3_winners} winners ({100*z3_winners/z3_buckets:.1f}%)")
    print(f"Gold-tier events (with temporal): {gold_events}")
    print()

    results = evaluate_approaches(events)

    output = compile_output(events, results)

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT}")

    # Print summary
    print("\n" + "=" * 80)
    print("ZONE 3 ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Baseline (Market) full Brier:  {output['baseline_full_brier']:.6f}")
    print(f"Baseline (Market) zone Brier:  {output['baseline_zone_brier']:.6f}")
    print()
    print(f"{'Approach':<30s} {'Full Brier':>12s} {'Zone Brier':>12s} {'Beat(Full)':>10s} {'Beat(Zone)':>10s} {'Improv%':>8s}")
    print("-" * 84)
    for a in output["approaches"]:
        print(f"{a['name']:<30s} {a['full_event_brier']:>12.6f} {a['zone_only_brier']:>12.6f} "
              f"{a['events_beating_baseline_full']:>10d} {a['events_beating_baseline_zone']:>10d} "
              f"{a['improvement_vs_baseline_full_pct']:>+7.2f}%")
    print()
    print(f"Best approach: {output['best_approach']} (full Brier: {output['best_full_brier']:.6f})")


if __name__ == "__main__":
    main()
