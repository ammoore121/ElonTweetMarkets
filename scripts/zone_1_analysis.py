"""
Zone 1 (Low Tail) Analysis for Elon Musk Tweet Count Prediction Markets.

Evaluates 7 modeling approaches for predicting low-tail bucket probabilities.
Uses Leave-One-Event-Out Cross-Validation (LOOCV) for learned approaches.

Output: data/analysis/zone_1_results.json
"""

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(r"G:\My Drive\AI_Projects\Ideas - In Progress\ElonTweetMarkets")
DATASET_PATH = ROOT / "data" / "analysis" / "bucket_dataset.json"
OUTPUT_PATH = ROOT / "data" / "analysis" / "zone_1_results.json"

TARGET_ZONE = 1
PROB_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset():
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def clamp(p, lo=PROB_FLOOR, hi=1 - PROB_FLOOR):
    return max(lo, min(hi, p))


def logit(p):
    p = clamp(p)
    return math.log(p / (1 - p))


def sigmoid(x):
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def normalize_distribution(probs: list[float]) -> list[float]:
    """Floor and normalize to sum to 1."""
    floored = [max(PROB_FLOOR, p) for p in probs]
    total = sum(floored)
    return [p / total for p in floored]


def brier_score(preds: list[float], actuals: list[int]) -> float:
    """Sum of squared errors across buckets."""
    return sum((p - a) ** 2 for p, a in zip(preds, actuals))


def negbin_bucket_prob(mu, var, lower, upper):
    """P(lower <= X <= upper) under NegBin(n, p) parameterized by mean and variance."""
    if var <= mu or mu <= 0:
        # Fallback: variance must exceed mean for NegBin. Use Poisson.
        return stats.poisson.cdf(upper, mu) - stats.poisson.cdf(lower - 1, mu) if lower > 0 else stats.poisson.cdf(upper, mu)
    p_param = mu / var  # scipy p parameter
    n_param = mu * mu / (var - mu)
    if n_param <= 0 or p_param <= 0 or p_param >= 1:
        return stats.poisson.cdf(upper, mu) - (stats.poisson.cdf(lower - 1, mu) if lower > 0 else 0.0)
    try:
        prob_upper = stats.nbinom.cdf(upper, n_param, p_param)
        prob_lower = stats.nbinom.cdf(lower - 1, n_param, p_param) if lower > 0 else 0.0
        return max(prob_upper - prob_lower, PROB_FLOOR)
    except Exception:
        return PROB_FLOOR


def gaussian_bucket_prob(mu, sigma, lower, upper):
    """P(lower <= X <= upper) under Normal(mu, sigma)."""
    if sigma <= 0:
        sigma = 1.0
    # Use continuity correction: treat bucket as [lower - 0.5, upper + 0.5]
    prob = stats.norm.cdf(upper + 0.5, mu, sigma) - stats.norm.cdf(lower - 0.5, mu, sigma)
    return max(prob, PROB_FLOOR)


# ---------------------------------------------------------------------------
# Approach 1: NegBin from crowd stats
# ---------------------------------------------------------------------------

def approach_negbin(event):
    """Parametric NegBin using crowd_implied_ev and crowd_std_dev."""
    features = event["features"]
    mu = features["market"]["crowd_implied_ev"]
    sigma = features["market"]["crowd_std_dev"]
    if mu is None or sigma is None or mu <= 0:
        return None
    var = sigma ** 2

    preds = []
    for b in event["buckets"]:
        prob = negbin_bucket_prob(mu, var, b["lower_bound"], b["upper_bound"])
        preds.append(prob)
    return normalize_distribution(preds)


# ---------------------------------------------------------------------------
# Approach 2: Gaussian from crowd stats
# ---------------------------------------------------------------------------

def approach_gaussian(event):
    """Parametric Gaussian using crowd_implied_ev and crowd_std_dev."""
    features = event["features"]
    mu = features["market"]["crowd_implied_ev"]
    sigma = features["market"]["crowd_std_dev"]
    if mu is None or sigma is None or mu <= 0:
        return None

    preds = []
    for b in event["buckets"]:
        prob = gaussian_bucket_prob(mu, sigma, b["lower_bound"], b["upper_bound"])
        preds.append(prob)
    return normalize_distribution(preds)


# ---------------------------------------------------------------------------
# Approach 3: Market baseline
# ---------------------------------------------------------------------------

def approach_market_baseline(event):
    """Use normalized market prices as-is."""
    return normalize_distribution([b["market_price"] for b in event["buckets"]])


# ---------------------------------------------------------------------------
# Approach 4: Platt-calibrated market (LOOCV)
# ---------------------------------------------------------------------------

def platt_predict(market_price, a, b_param):
    """Apply Platt calibration to a market price."""
    return sigmoid(a * logit(market_price) + b_param)


def platt_neg_log_likelihood(params, market_prices, actuals):
    """Negative log-likelihood for Platt calibration."""
    a, b = params
    nll = 0.0
    for mp, y in zip(market_prices, actuals):
        p = clamp(platt_predict(mp, a, b))
        nll -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return nll


def approach_platt_calibrated(event, all_events, event_idx):
    """Platt calibration on Zone 1 buckets, trained LOOCV."""
    # Collect training data from all other events
    train_prices = []
    train_actuals = []
    for i, ev in enumerate(all_events):
        if i == event_idx:
            continue
        for b in ev["buckets"]:
            if b["zone"] == TARGET_ZONE:
                train_prices.append(b["market_price"])
                train_actuals.append(b["is_winner"])

    if len(train_prices) < 5:
        return None

    # Fit Platt calibration
    try:
        result = optimize.minimize(
            platt_neg_log_likelihood, [1.0, 0.0],
            args=(train_prices, train_actuals),
            method="Nelder-Mead",
            options={"maxiter": 5000}
        )
        a_opt, b_opt = result.x
    except Exception:
        return None

    # Apply to this event
    preds = []
    for b in event["buckets"]:
        if b["zone"] == TARGET_ZONE:
            preds.append(clamp(platt_predict(b["market_price"], a_opt, b_opt)))
        else:
            preds.append(b["market_price"])
    return normalize_distribution(preds), {"a": float(a_opt), "b": float(b_opt)}


# ---------------------------------------------------------------------------
# Approach 5: Market + z-score tail adjustment (LOOCV)
# ---------------------------------------------------------------------------

def approach_zscore_adjustment(event, all_events, event_idx):
    """Adjust Zone 1 market prices based on z-score magnitude."""
    # Train: learn a linear adjustment factor: p_adj = p_market * exp(alpha * |z_score| + beta)
    # Fit alpha, beta to minimize Brier score on training data

    # Collect training data per event for Brier optimization
    train_events = [ev for i, ev in enumerate(all_events) if i != event_idx]

    def objective(params):
        alpha, beta = params
        total_brier = 0.0
        n_events = 0
        for ev in train_events:
            preds = []
            actuals = []
            for b in ev["buckets"]:
                if b["zone"] == TARGET_ZONE:
                    z = abs(b["z_score"]) if b["z_score"] is not None else 1.0
                    exponent = max(-10, min(10, alpha * z + beta))
                    adj = math.exp(exponent)
                    preds.append(max(b["market_price"] * adj, PROB_FLOOR))
                else:
                    preds.append(b["market_price"])
                actuals.append(b["is_winner"])
            preds = normalize_distribution(preds)
            total_brier += brier_score(preds, actuals)
            n_events += 1
        return total_brier / max(n_events, 1)

    try:
        result = optimize.minimize(
            objective, [0.0, 0.0],
            method="Nelder-Mead",
            options={"maxiter": 3000}
        )
        alpha_opt, beta_opt = result.x
    except Exception:
        return None

    # Apply
    preds = []
    for b in event["buckets"]:
        if b["zone"] == TARGET_ZONE:
            z = abs(b["z_score"]) if b["z_score"] is not None else 1.0
            exponent = max(-10, min(10, alpha_opt * z + beta_opt))
            adj = math.exp(exponent)
            preds.append(max(b["market_price"] * adj, PROB_FLOOR))
        else:
            preds.append(b["market_price"])
    return normalize_distribution(preds), {"alpha": float(alpha_opt), "beta": float(beta_opt)}


# ---------------------------------------------------------------------------
# Approach 6: Regime-conditional tail pricing (LOOCV)
# ---------------------------------------------------------------------------

def approach_regime_conditional(event, all_events, event_idx):
    """Use temporal features to adjust Zone 1 probabilities when available."""
    features = event["features"]
    duration = event["duration_days"]

    # Check if temporal features are available for this event
    has_temporal = (
        features["temporal"]["rolling_avg_7d"] is not None
        and features["market"]["crowd_implied_ev"] is not None
    )

    if not has_temporal:
        # Fall back to market baseline
        return normalize_distribution([b["market_price"] for b in event["buckets"]]), {"fallback": True}

    # Collect training events with temporal features
    train_events = []
    for i, ev in enumerate(all_events):
        if i == event_idx:
            continue
        if ev["features"]["temporal"]["rolling_avg_7d"] is not None:
            train_events.append(ev)

    if len(train_events) < 5:
        return normalize_distribution([b["market_price"] for b in event["buckets"]]), {"fallback": True, "reason": "insufficient training"}

    # Learn: when rolling_avg * duration < crowd_ev, boost zone 1
    #         when rolling_avg * duration > crowd_ev, shrink zone 1
    # Adjustment: p_adj = p_market * exp(gamma * ratio_deviation)
    # ratio_deviation = (crowd_ev - rolling_avg * duration) / crowd_ev

    def objective(params):
        gamma = params[0]
        total_brier = 0.0
        n = 0
        for ev in train_events:
            ra = ev["features"]["temporal"]["rolling_avg_7d"]
            cev = ev["features"]["market"]["crowd_implied_ev"]
            dur = ev["duration_days"]
            if cev is None or cev <= 0 or ra is None:
                continue
            ratio_dev = (cev - ra * dur) / cev

            preds = []
            actuals = []
            for b in ev["buckets"]:
                if b["zone"] == TARGET_ZONE:
                    # Positive ratio_dev means crowd expects MORE than trailing avg suggests
                    # => trailing avg says counts will be lower => boost low tail
                    exponent = max(-10, min(10, -gamma * ratio_dev))
                    adj = math.exp(exponent)  # negative sign: if crowd > trailing, boost low tail
                    preds.append(max(b["market_price"] * adj, PROB_FLOOR))
                else:
                    preds.append(b["market_price"])
                actuals.append(b["is_winner"])
            preds = normalize_distribution(preds)
            total_brier += brier_score(preds, actuals)
            n += 1
        return total_brier / max(n, 1)

    try:
        result = optimize.minimize(
            objective, [0.0],
            method="Nelder-Mead",
            options={"maxiter": 3000}
        )
        gamma_opt = result.x[0]
    except Exception:
        return normalize_distribution([b["market_price"] for b in event["buckets"]]), {"fallback": True, "reason": "optimization failed"}

    # Apply
    ra = features["temporal"]["rolling_avg_7d"]
    cev = features["market"]["crowd_implied_ev"]
    if cev is None or cev <= 0:
        return normalize_distribution([b["market_price"] for b in event["buckets"]]), {"fallback": True}

    ratio_dev = (cev - ra * duration) / cev

    preds = []
    for b in event["buckets"]:
        if b["zone"] == TARGET_ZONE:
            exponent = max(-10, min(10, -gamma_opt * ratio_dev))
            adj = math.exp(exponent)
            preds.append(max(b["market_price"] * adj, PROB_FLOOR))
        else:
            preds.append(b["market_price"])
    return normalize_distribution(preds), {"gamma": float(gamma_opt), "ratio_dev": float(ratio_dev)}


# ---------------------------------------------------------------------------
# Approach 7: Multi-feature tail model (LOOCV)
# ---------------------------------------------------------------------------

def extract_tail_features(event, bucket):
    """Extract feature vector for a Zone 1 bucket."""
    feats = event["features"]
    f = []

    # 1. z_score
    f.append(bucket["z_score"] if bucket["z_score"] is not None else 0.0)

    # 2. logit of market price
    f.append(logit(bucket["market_price"]))

    # 3. GDELT tone (negative tone -> more tweets -> less likely low tail)
    tone = feats["media"].get("elon_musk_tone_7d")
    f.append(tone if tone is not None else 0.0)

    # 4. launches_trailing_7d (busy SpaceX -> fewer tweets -> more likely low tail)
    launches = feats["calendar"].get("launches_trailing_7d")
    f.append(launches if launches is not None else 0.0)

    # 5. crowd_skewness
    skew = feats["market"].get("crowd_skewness")
    f.append(skew if skew is not None else 0.0)

    # 6. price_shift_24h
    shift = feats["market"].get("price_shift_24h")
    f.append(shift / 100.0 if shift is not None else 0.0)  # scale

    # 7. relative_position (how far into the distribution this bucket is)
    f.append(bucket["relative_position"])

    # 8. crowd CV (variability)
    cev = feats["market"].get("crowd_implied_ev")
    csd = feats["market"].get("crowd_std_dev")
    if cev and csd and cev > 0:
        f.append(csd / cev)
    else:
        f.append(0.0)

    return f


def approach_multi_feature(event, all_events, event_idx):
    """Logistic regression on multiple features for Zone 1 buckets."""
    # Collect training data
    X_train = []
    y_train = []
    for i, ev in enumerate(all_events):
        if i == event_idx:
            continue
        for b in ev["buckets"]:
            if b["zone"] == TARGET_ZONE:
                X_train.append(extract_tail_features(ev, b))
                y_train.append(b["is_winner"])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if len(X_train) < 10:
        return None

    n_features = X_train.shape[1]

    # Standardize features
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0
    X_train_scaled = (X_train - means) / stds

    # Logistic regression with L2 regularization
    def neg_log_likelihood(params):
        w = params[:n_features]
        b = params[n_features]
        reg = 0.1 * np.sum(w ** 2)  # L2 penalty
        logits = X_train_scaled @ w + b
        # Clip for numerical stability
        logits = np.clip(logits, -500, 500)
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, PROB_FLOOR, 1 - PROB_FLOOR)
        nll = -np.sum(y_train * np.log(probs) + (1 - y_train) * np.log(1 - probs))
        return nll + reg

    try:
        init = np.zeros(n_features + 1)
        # Initialize bias to base rate logit
        base_rate = max(y_train.mean(), PROB_FLOOR)
        init[-1] = math.log(base_rate / (1 - base_rate))
        result = optimize.minimize(
            neg_log_likelihood, init,
            method="L-BFGS-B",
            options={"maxiter": 5000}
        )
        w_opt = result.x[:n_features]
        b_opt = result.x[n_features]
    except Exception:
        return None

    # Predict for this event
    preds = []
    for b in event["buckets"]:
        if b["zone"] == TARGET_ZONE:
            x = np.array(extract_tail_features(event, b))
            x_scaled = (x - means) / stds
            logit_val = float(x_scaled @ w_opt + b_opt)
            p = sigmoid(logit_val)
            preds.append(clamp(p))
        else:
            preds.append(b["market_price"])

    feature_names = ["z_score", "logit_market_price", "gdelt_tone", "launches_7d",
                     "crowd_skewness", "price_shift_24h", "relative_position", "crowd_cv"]
    weights = {name: float(w) for name, w in zip(feature_names, w_opt)}
    return normalize_distribution(preds), {"weights": weights, "bias": float(b_opt)}


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate_approach(name, predict_fn, events, is_learned=False):
    """Run LOOCV evaluation for an approach."""
    full_briers = []
    zone_briers = []
    baseline_briers = []
    params_collected = []
    n_better = 0
    n_evaluated = 0
    n_skipped = 0

    for i, event in enumerate(events):
        if i % 20 == 0:
            print(f"    Event {i}/{len(events)}...", flush=True)
        # Get predictions
        if is_learned:
            result = predict_fn(event, events, i)
        else:
            result = predict_fn(event)

        # Handle return format
        params = {}
        if result is None:
            n_skipped += 1
            continue
        if isinstance(result, tuple):
            preds, params = result
        else:
            preds = result

        if preds is None:
            n_skipped += 1
            continue

        params_collected.append(params)

        # Actuals
        actuals = [b["is_winner"] for b in event["buckets"]]

        # Full event Brier
        full_brier = brier_score(preds, actuals)
        full_briers.append(full_brier)

        # Zone-only Brier
        zone_preds = [p for p, b in zip(preds, event["buckets"]) if b["zone"] == TARGET_ZONE]
        zone_actuals = [a for a, b in zip(actuals, event["buckets"]) if b["zone"] == TARGET_ZONE]
        if zone_preds:
            zone_brier = brier_score(zone_preds, zone_actuals)
            zone_briers.append(zone_brier)

        # Baseline comparison
        baseline_preds = normalize_distribution([b["market_price"] for b in event["buckets"]])
        baseline_brier = brier_score(baseline_preds, actuals)
        baseline_briers.append(baseline_brier)
        n_evaluated += 1

        if full_brier < baseline_brier:
            n_better += 1

    if not full_briers:
        return {
            "name": name,
            "full_event_brier": None,
            "zone_only_brier": None,
            "events_beating_baseline": 0,
            "n_evaluated": 0,
            "n_skipped": n_skipped,
            "params": {},
            "error": "No events evaluated"
        }

    return {
        "name": name,
        "full_event_brier": float(np.mean(full_briers)),
        "zone_only_brier": float(np.mean(zone_briers)) if zone_briers else None,
        "events_beating_baseline": n_better,
        "n_evaluated": n_evaluated,
        "n_skipped": n_skipped,
        "baseline_full_brier": float(np.mean(baseline_briers)),
        "brier_improvement": float(np.mean(baseline_briers) - np.mean(full_briers)),
        "params": params_collected[-1] if params_collected else {}
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ZONE 1 (LOW TAIL) ANALYSIS")
    print("=" * 70)

    # Load data
    data = load_dataset()
    events = data["events"]
    summary = data["summary"]

    print(f"\nDataset: {len(events)} events, {summary['zone_summary']['1']['n_buckets']} Zone 1 buckets")
    print(f"Zone 1 winners: {summary['zone_summary']['1']['n_winners']} ({summary['zone_summary']['1']['win_rate']:.1%})")
    print(f"Zone 1 avg market price: {summary['zone_summary']['1']['avg_market_price']:.4f}")

    # Count events with temporal features
    n_temporal = sum(1 for ev in events if ev["features"]["temporal"]["rolling_avg_7d"] is not None)
    print(f"Events with temporal features: {n_temporal}/{len(events)}")

    # Run approaches
    results = {
        "zone": TARGET_ZONE,
        "n_events": len(events),
        "n_zone_buckets": summary["zone_summary"]["1"]["n_buckets"],
        "n_zone_winners": summary["zone_summary"]["1"]["n_winners"],
        "approaches": []
    }

    approaches = [
        ("negbin_crowd", approach_negbin, False),
        ("gaussian_crowd", approach_gaussian, False),
        ("market_baseline", approach_market_baseline, False),
        ("platt_calibrated", approach_platt_calibrated, True),
        ("zscore_adjustment", approach_zscore_adjustment, True),
        ("regime_conditional", approach_regime_conditional, True),
        ("multi_feature", approach_multi_feature, True),
    ]

    for name, fn, is_learned in approaches:
        print(f"\n{'─' * 50}")
        print(f"Running: {name} {'(LOOCV)' if is_learned else '(parametric)'}")
        print(f"{'─' * 50}")

        try:
            result = evaluate_approach(name, fn, events, is_learned)
            results["approaches"].append(result)

            fb = result["full_event_brier"]
            zb = result["zone_only_brier"]
            nb = result["events_beating_baseline"]
            ne = result["n_evaluated"]
            ns = result["n_skipped"]
            imp = result.get("brier_improvement", 0)

            print(f"  Full-event Brier:  {fb:.6f}" if fb else "  Full-event Brier:  N/A")
            print(f"  Zone-1-only Brier: {zb:.6f}" if zb else "  Zone-1-only Brier: N/A")
            print(f"  Events beating baseline: {nb}/{ne}")
            print(f"  Brier improvement: {imp:+.6f}" if imp else "  Brier improvement: N/A")
            if ns > 0:
                print(f"  Skipped events: {ns}")
            if result.get("params"):
                # Print params compactly
                p = result["params"]
                if isinstance(p, dict):
                    for k, v in p.items():
                        if isinstance(v, dict):
                            print(f"  {k}: {json.dumps(v, indent=None)}")
                        elif isinstance(v, float):
                            print(f"  {k}: {v:.4f}")
                        else:
                            print(f"  {k}: {v}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["approaches"].append({
                "name": name,
                "full_event_brier": None,
                "zone_only_brier": None,
                "events_beating_baseline": 0,
                "n_evaluated": 0,
                "error": str(e),
                "params": {}
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid = [a for a in results["approaches"] if a["full_event_brier"] is not None]
    if valid:
        # Sort by full event Brier (lower is better)
        valid.sort(key=lambda x: x["full_event_brier"])
        baseline = next((a for a in valid if a["name"] == "market_baseline"), None)
        baseline_brier = baseline["full_event_brier"] if baseline else None

        print(f"\n{'Approach':<25} {'Full Brier':>12} {'Zone Brier':>12} {'vs Baseline':>12} {'Beat Base':>10}")
        print("-" * 75)
        for a in valid:
            fb = a["full_event_brier"]
            zb = a["zone_only_brier"]
            diff = (fb - baseline_brier) if baseline_brier else 0
            nb = a["events_beating_baseline"]
            ne = a["n_evaluated"]
            print(f"{a['name']:<25} {fb:>12.6f} {zb:>12.6f} {diff:>+12.6f} {nb:>5}/{ne}")

        best = valid[0]
        results["best_approach"] = best["name"]
        results["best_full_brier"] = best["full_event_brier"]
        results["baseline_full_brier"] = baseline_brier

        print(f"\nBest approach: {best['name']} (Brier: {best['full_event_brier']:.6f})")
        if baseline_brier:
            print(f"Baseline Brier: {baseline_brier:.6f}")
            print(f"Improvement: {baseline_brier - best['full_event_brier']:+.6f}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
