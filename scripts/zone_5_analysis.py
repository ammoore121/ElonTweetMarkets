"""
Zone 5 (High Tail) Analysis for Elon Musk Tweet Count Prediction Markets.

Zone 5 = top 20% position buckets (far-above-expected-count tail).
Market avg price ~4.7%, actual win rate ~4.4% => slightly overpriced by ~0.4%.

7 approaches evaluated via LOOCV (leave-one-event-out cross-validation):
1. NegBin from crowd stats
2. Gaussian from crowd stats
3. Market baseline (no correction)
4. Platt-calibrated market
5. Market + tail-specific shrinkage
6. Market + volatility-conditional tail
7. Multi-feature tail model
"""

import json
import math
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize
from scipy.special import expit, logit

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DATASET_PATH = PROJECT_DIR / "data" / "analysis" / "bucket_dataset.json"
OUTPUT_PATH = PROJECT_DIR / "data" / "analysis" / "zone_5_results.json"

PROB_FLOOR = 1e-6
TARGET_ZONE = 5


def load_dataset():
    """Load bucket dataset and return events list."""
    with open(DATASET_PATH) as f:
        data = json.load(f)
    return data["events"], data["summary"]


def normalize_probs(probs):
    """Normalize probability vector to sum to 1, with floor."""
    probs = np.array(probs, dtype=np.float64)
    probs = np.maximum(probs, PROB_FLOOR)
    return probs / probs.sum()


def brier_score(probs, outcomes):
    """Brier score for a single event. Lower is better."""
    probs = np.array(probs, dtype=np.float64)
    outcomes = np.array(outcomes, dtype=np.float64)
    return float(np.sum((probs - outcomes) ** 2))


def zone5_brier(probs, outcomes, zones):
    """Brier score computed only on Zone 5 buckets."""
    probs = np.array(probs)
    outcomes = np.array(outcomes)
    zones = np.array(zones)
    mask = zones == TARGET_ZONE
    if mask.sum() == 0:
        return 0.0
    return float(np.sum((probs[mask] - outcomes[mask]) ** 2))


def get_event_arrays(event):
    """Extract parallel arrays from an event's buckets."""
    buckets = event["buckets"]
    market_prices = np.array([b["market_price"] for b in buckets])
    outcomes = np.array([b["is_winner"] for b in buckets])
    zones = np.array([b["zone"] for b in buckets])
    lower_bounds = np.array([b["lower_bound"] for b in buckets])
    upper_bounds = np.array([b["upper_bound"] for b in buckets])
    z_scores = np.array([b["z_score"] for b in buckets])
    midpoints = np.array([b["midpoint"] for b in buckets])
    widths = np.array([b["width"] for b in buckets])
    return {
        "market_prices": market_prices,
        "outcomes": outcomes,
        "zones": zones,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "z_scores": z_scores,
        "midpoints": midpoints,
        "widths": widths,
        "n_buckets": len(buckets),
    }


# ---------------------------------------------------------------------------
# Approach 1: Negative Binomial from crowd stats
# ---------------------------------------------------------------------------
def approach_negbin(event, arrays):
    """
    Fit NegBin distribution from crowd implied EV and std dev.
    NegBin has heavier tail than Gaussian -- better for extreme outcomes.
    """
    crowd_ev = event["features"]["market"].get("crowd_implied_ev")
    crowd_std = event["features"]["market"].get("crowd_std_dev")

    if crowd_ev is None or crowd_std is None or crowd_ev <= 0 or crowd_std <= 0:
        # Fallback to market prices
        return normalize_probs(arrays["market_prices"])

    mean = crowd_ev
    var = crowd_std ** 2

    # NegBin requires var > mean for valid parameters
    if var <= mean:
        var = mean * 1.1  # slight overdispersion fallback

    # NegBin parameterization: n = mean^2 / (var - mean), p = mean / var
    n_nb = mean ** 2 / (var - mean)
    p_nb = mean / var  # scipy uses p as probability of success

    # scipy.stats.nbinom: P(X=k) where X ~ NegBin(n, p)
    # scipy parameterization: n=n_nb, p=p_nb => mean = n(1-p)/p
    # We need: mean = n_nb * (1-p_nb) / p_nb
    # Check: n_nb * (1-p_nb) / p_nb = (mean^2/(var-mean)) * ((var-mean)/var) / (mean/var)
    #       = (mean^2/(var-mean)) * (var-mean)/(var) * var/mean = mean. Correct.

    probs = np.zeros(arrays["n_buckets"])
    for i in range(arrays["n_buckets"]):
        lb = arrays["lower_bounds"][i]
        ub = arrays["upper_bounds"][i]
        if ub >= 99999:
            # Open-ended bucket: P(X >= lb)
            probs[i] = 1.0 - sp_stats.nbinom.cdf(lb - 1, n_nb, p_nb) if lb > 0 else 1.0
        else:
            # P(lb <= X <= ub)
            cdf_ub = sp_stats.nbinom.cdf(ub, n_nb, p_nb)
            cdf_lb = sp_stats.nbinom.cdf(lb - 1, n_nb, p_nb) if lb > 0 else 0.0
            probs[i] = cdf_ub - cdf_lb

    return normalize_probs(probs)


# ---------------------------------------------------------------------------
# Approach 2: Gaussian from crowd stats
# ---------------------------------------------------------------------------
def approach_gaussian(event, arrays):
    """
    Fit Gaussian from crowd implied EV and std dev.
    Thinner tails than NegBin -- likely worse for Zone 5.
    """
    crowd_ev = event["features"]["market"].get("crowd_implied_ev")
    crowd_std = event["features"]["market"].get("crowd_std_dev")

    if crowd_ev is None or crowd_std is None or crowd_ev <= 0 or crowd_std <= 0:
        return normalize_probs(arrays["market_prices"])

    probs = np.zeros(arrays["n_buckets"])
    for i in range(arrays["n_buckets"]):
        lb = arrays["lower_bounds"][i]
        ub = arrays["upper_bounds"][i]
        if ub >= 99999:
            probs[i] = 1.0 - sp_stats.norm.cdf(lb - 0.5, crowd_ev, crowd_std)
        else:
            probs[i] = sp_stats.norm.cdf(ub + 0.5, crowd_ev, crowd_std) - \
                        sp_stats.norm.cdf(lb - 0.5, crowd_ev, crowd_std)

    return normalize_probs(probs)


# ---------------------------------------------------------------------------
# Approach 3: Market baseline
# ---------------------------------------------------------------------------
def approach_market_baseline(event, arrays):
    """Just use market prices as-is (normalized). This IS the baseline."""
    return normalize_probs(arrays["market_prices"])


# ---------------------------------------------------------------------------
# Approach 4: Platt-calibrated market
# ---------------------------------------------------------------------------
def approach_platt_calibrated(event, arrays, a, b):
    """
    P_corrected = sigmoid(a * logit(P_market) + b)
    Applied ONLY to Zone 5 buckets. Non-zone-5 keeps market prices.
    """
    probs = arrays["market_prices"].copy()
    z5_mask = arrays["zones"] == TARGET_ZONE

    for i in range(arrays["n_buckets"]):
        if z5_mask[i]:
            p = np.clip(probs[i], 1e-6, 1 - 1e-6)
            probs[i] = expit(a * logit(p) + b)

    return normalize_probs(probs)


def train_platt(train_events, all_arrays):
    """Learn Platt parameters (a, b) from training events."""
    # Collect all zone-5 (market_price, outcome) pairs from training events
    market_ps = []
    outcomes = []
    for idx in train_events:
        arr = all_arrays[idx]
        z5_mask = arr["zones"] == TARGET_ZONE
        market_ps.extend(arr["market_prices"][z5_mask].tolist())
        outcomes.extend(arr["outcomes"][z5_mask].tolist())

    market_ps = np.array(market_ps)
    outcomes = np.array(outcomes)

    if len(market_ps) < 5:
        return 1.0, 0.0

    # Optimize log-loss
    def neg_log_likelihood(params):
        a, b = params
        ps_clipped = np.clip(market_ps, 1e-6, 1 - 1e-6)
        calibrated = expit(a * logit(ps_clipped) + b)
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        ll = outcomes * np.log(calibrated) + (1 - outcomes) * np.log(1 - calibrated)
        return -np.mean(ll)

    result = minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead",
                      options={"maxiter": 1000})
    return result.x[0], result.x[1]


# ---------------------------------------------------------------------------
# Approach 5: Market + tail-specific shrinkage
# ---------------------------------------------------------------------------
def approach_tail_shrinkage(event, arrays, shrink_regular, shrink_open):
    """
    Zone 5 is overpriced by ~0.4% on average => systematically shrink.
    Open-ended buckets (upper_bound >= 99999) get extra shrinkage.
    """
    probs = arrays["market_prices"].copy()
    z5_mask = arrays["zones"] == TARGET_ZONE

    for i in range(arrays["n_buckets"]):
        if z5_mask[i]:
            if arrays["upper_bounds"][i] >= 99999:
                probs[i] = probs[i] * (1 - shrink_open)
            else:
                probs[i] = probs[i] * (1 - shrink_regular)

    return normalize_probs(probs)


def train_tail_shrinkage(train_events, all_arrays, events):
    """Learn shrinkage factors from training events by two-stage grid search."""
    best_brier = float("inf")
    best_params = (0.0, 0.0)

    # Stage 1: Coarse grid
    for shrink_reg in np.arange(0.0, 0.60, 0.10):
        for shrink_open in np.arange(0.0, 0.80, 0.10):
            total_brier = 0.0
            for idx in train_events:
                arr = all_arrays[idx]
                probs = approach_tail_shrinkage(events[idx], arr, shrink_reg, shrink_open)
                total_brier += brier_score(probs, arr["outcomes"])
            if total_brier < best_brier:
                best_brier = total_brier
                best_params = (shrink_reg, shrink_open)

    # Stage 2: Fine grid around best
    center_reg, center_open = best_params
    for shrink_reg in np.arange(max(0, center_reg - 0.10), min(0.60, center_reg + 0.12), 0.02):
        for shrink_open in np.arange(max(0, center_open - 0.10), min(0.80, center_open + 0.12), 0.02):
            total_brier = 0.0
            for idx in train_events:
                arr = all_arrays[idx]
                probs = approach_tail_shrinkage(events[idx], arr, shrink_reg, shrink_open)
                total_brier += brier_score(probs, arr["outcomes"])
            if total_brier < best_brier:
                best_brier = total_brier
                best_params = (shrink_reg, shrink_open)

    return best_params


# ---------------------------------------------------------------------------
# Approach 6: Market + volatility-conditional tail
# ---------------------------------------------------------------------------
def get_cv(event):
    """Get coefficient of variation. Use cv_14d if available, else from crowd stats."""
    cv = event["features"]["temporal"].get("cv_14d")
    if cv is not None:
        return cv
    # Fallback: crowd_std_dev / crowd_implied_ev
    crowd_ev = event["features"]["market"].get("crowd_implied_ev")
    crowd_std = event["features"]["market"].get("crowd_std_dev")
    if crowd_ev and crowd_std and crowd_ev > 0:
        return crowd_std / crowd_ev
    return None


def approach_vol_conditional(event, arrays, threshold, boost_factor, shrink_factor):
    """
    High variance regimes => fatter tails => boost Zone 5.
    Low variance regimes => thinner tails => shrink Zone 5.
    """
    probs = arrays["market_prices"].copy()
    z5_mask = arrays["zones"] == TARGET_ZONE
    cv = get_cv(event)

    if cv is None:
        return normalize_probs(probs)

    for i in range(arrays["n_buckets"]):
        if z5_mask[i]:
            if cv > threshold:
                probs[i] = probs[i] * (1 + boost_factor * (cv - threshold))
            else:
                probs[i] = probs[i] * (1 - shrink_factor)

    return normalize_probs(probs)


def train_vol_conditional(train_events, all_arrays, events):
    """Learn vol-conditional parameters from training events (two-stage grid)."""
    best_brier = float("inf")
    best_params = (0.3, 0.0, 0.0)

    # Stage 1: Coarse grid
    for threshold in np.arange(0.15, 0.55, 0.10):
        for boost in np.arange(0.0, 2.0, 0.4):
            for shrink in np.arange(0.0, 0.50, 0.10):
                total_brier = 0.0
                for idx in train_events:
                    arr = all_arrays[idx]
                    probs = approach_vol_conditional(
                        events[idx], arr, threshold, boost, shrink
                    )
                    total_brier += brier_score(probs, arr["outcomes"])
                if total_brier < best_brier:
                    best_brier = total_brier
                    best_params = (threshold, boost, shrink)

    # Stage 2: Fine grid around best
    ct, cb, cs = best_params
    for threshold in np.arange(max(0.10, ct - 0.10), min(0.60, ct + 0.12), 0.03):
        for boost in np.arange(max(0.0, cb - 0.4), min(2.5, cb + 0.5), 0.1):
            for shrink in np.arange(max(0.0, cs - 0.10), min(0.55, cs + 0.12), 0.03):
                total_brier = 0.0
                for idx in train_events:
                    arr = all_arrays[idx]
                    probs = approach_vol_conditional(
                        events[idx], arr, threshold, boost, shrink
                    )
                    total_brier += brier_score(probs, arr["outcomes"])
                if total_brier < best_brier:
                    best_brier = total_brier
                    best_params = (threshold, boost, shrink)

    return best_params


# ---------------------------------------------------------------------------
# Approach 7: Multi-feature tail model
# ---------------------------------------------------------------------------
def extract_z5_features(event, bucket):
    """Extract features for a Zone 5 bucket prediction."""
    features = {}
    features["z_score"] = bucket["z_score"]
    features["market_price"] = bucket["market_price"]
    features["is_open_ended"] = 1.0 if bucket["upper_bound"] >= 99999 else 0.0

    mkt = event["features"]["market"]
    features["crowd_kurtosis"] = mkt.get("crowd_kurtosis", 0.0) or 0.0
    features["distribution_entropy"] = mkt.get("distribution_entropy", 0.0) or 0.0

    media = event["features"]["media"]
    features["elon_musk_tone_delta"] = media.get("elon_musk_tone_delta", 0.0) or 0.0

    temporal = event["features"]["temporal"]
    features["regime_ratio"] = temporal.get("regime_ratio") or 0.0
    features["cv_14d"] = temporal.get("cv_14d") or 0.0

    return features


FEATURE_KEYS = [
    "z_score", "market_price", "is_open_ended", "crowd_kurtosis",
    "distribution_entropy", "elon_musk_tone_delta", "regime_ratio", "cv_14d",
]


def approach_multi_feature(event, arrays, weights, intercept):
    """
    Linear model in logit space for Zone 5 buckets.
    logit(P_corrected) = intercept + sum(w_i * x_i)
    Non-zone-5 keeps market prices.
    """
    probs = arrays["market_prices"].copy()
    z5_mask = arrays["zones"] == TARGET_ZONE
    buckets = event["buckets"]

    for i in range(arrays["n_buckets"]):
        if z5_mask[i]:
            feats = extract_z5_features(event, buckets[i])
            feat_vec = np.array([feats[k] for k in FEATURE_KEYS])
            logit_p = intercept + np.dot(weights, feat_vec)
            probs[i] = expit(logit_p)

    return normalize_probs(probs)


def train_multi_feature(train_events, all_arrays, events):
    """Learn multi-feature model weights via L-BFGS-B on log-loss."""
    # Collect all zone-5 observations from training
    X_list = []
    y_list = []
    for idx in train_events:
        arr = all_arrays[idx]
        z5_mask = arr["zones"] == TARGET_ZONE
        buckets = events[idx]["buckets"]
        for i in range(arr["n_buckets"]):
            if z5_mask[i]:
                feats = extract_z5_features(events[idx], buckets[i])
                feat_vec = [feats[k] for k in FEATURE_KEYS]
                X_list.append(feat_vec)
                y_list.append(arr["outcomes"][i])

    if len(X_list) < 10:
        return np.zeros(len(FEATURE_KEYS)), logit(0.044)  # prior = base rate

    X = np.array(X_list)
    y = np.array(y_list)

    # Standardize features (except market_price which is already on [0,1])
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-8] = 1.0
    X_std = (X - means) / stds

    def neg_log_likelihood(params):
        w = params[:len(FEATURE_KEYS)]
        b = params[len(FEATURE_KEYS)]
        logits = X_std @ w + b
        preds = expit(logits)
        preds = np.clip(preds, 1e-6, 1 - 1e-6)
        ll = y * np.log(preds) + (1 - y) * np.log(1 - preds)
        # L2 regularization
        reg = 0.1 * np.sum(w ** 2)
        return -np.mean(ll) + reg

    x0 = np.zeros(len(FEATURE_KEYS) + 1)
    x0[-1] = logit(0.044)  # initialize intercept at base rate

    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                      options={"maxiter": 500})

    w_std = result.x[:len(FEATURE_KEYS)]
    b = result.x[len(FEATURE_KEYS)]

    # Convert back from standardized space
    weights = w_std / stds
    intercept = b - np.dot(w_std / stds, means)

    return weights, intercept


# ---------------------------------------------------------------------------
# LOOCV Engine
# ---------------------------------------------------------------------------
def run_loocv(events, approach_name):
    """Run leave-one-event-out cross-validation for an approach."""
    n = len(events)
    all_arrays = [get_event_arrays(e) for e in events]

    full_briers = []
    z5_briers = []
    z5_correct = 0
    z5_total_buckets = 0
    z5_winners = 0
    param_log = []

    needs_training = approach_name in ("platt_calibrated", "tail_shrinkage",
                                       "vol_conditional", "multi_feature")

    for hold_out in range(n):
        if needs_training and hold_out % 20 == 0:
            print(f"    LOOCV fold {hold_out+1}/{n}...", flush=True)
        arr = all_arrays[hold_out]
        train_indices = [i for i in range(n) if i != hold_out]

        # Generate predictions based on approach
        if approach_name == "negbin":
            probs = approach_negbin(events[hold_out], arr)

        elif approach_name == "gaussian":
            probs = approach_gaussian(events[hold_out], arr)

        elif approach_name == "market_baseline":
            probs = approach_market_baseline(events[hold_out], arr)

        elif approach_name == "platt_calibrated":
            a, b = train_platt(train_indices, all_arrays)
            probs = approach_platt_calibrated(events[hold_out], arr, a, b)
            if hold_out < 3:
                param_log.append({"a": round(a, 4), "b": round(b, 4)})

        elif approach_name == "tail_shrinkage":
            shrink_reg, shrink_open = train_tail_shrinkage(
                train_indices, all_arrays, events
            )
            probs = approach_tail_shrinkage(events[hold_out], arr, shrink_reg, shrink_open)
            if hold_out < 3:
                param_log.append({
                    "shrink_regular": round(shrink_reg, 3),
                    "shrink_open": round(shrink_open, 3),
                })

        elif approach_name == "vol_conditional":
            threshold, boost, shrink = train_vol_conditional(
                train_indices, all_arrays, events
            )
            probs = approach_vol_conditional(
                events[hold_out], arr, threshold, boost, shrink
            )
            if hold_out < 3:
                param_log.append({
                    "threshold": round(threshold, 3),
                    "boost": round(boost, 3),
                    "shrink": round(shrink, 3),
                })

        elif approach_name == "multi_feature":
            weights, intercept = train_multi_feature(train_indices, all_arrays, events)
            probs = approach_multi_feature(events[hold_out], arr, weights, intercept)
            if hold_out < 3:
                param_log.append({
                    "weights": {k: round(float(w), 4) for k, w in zip(FEATURE_KEYS, weights)},
                    "intercept": round(float(intercept), 4),
                })

        else:
            raise ValueError(f"Unknown approach: {approach_name}")

        # Score
        fb = brier_score(probs, arr["outcomes"])
        zb = zone5_brier(probs, arr["outcomes"], arr["zones"])
        full_briers.append(fb)
        z5_briers.append(zb)

        # Zone 5 accuracy: did we assign highest zone-5 prob to the actual winner?
        z5_mask = arr["zones"] == TARGET_ZONE
        z5_outcomes = arr["outcomes"][z5_mask]
        z5_total_buckets += z5_mask.sum()
        z5_winners += z5_outcomes.sum()

    result = {
        "approach": approach_name,
        "full_brier_mean": round(float(np.mean(full_briers)), 6),
        "full_brier_median": round(float(np.median(full_briers)), 6),
        "full_brier_std": round(float(np.std(full_briers)), 6),
        "z5_brier_mean": round(float(np.mean(z5_briers)), 6),
        "z5_brier_median": round(float(np.median(z5_briers)), 6),
        "full_brier_sum": round(float(np.sum(full_briers)), 4),
        "z5_brier_sum": round(float(np.sum(z5_briers)), 4),
        "n_events": n,
        "z5_total_buckets": int(z5_total_buckets),
        "z5_total_winners": int(z5_winners),
    }

    if param_log:
        result["sample_params"] = param_log

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Zone 5 (High Tail) Analysis - Elon Tweet Prediction Markets")
    print("=" * 70)

    events, summary = load_dataset()
    print(f"\nLoaded {len(events)} events, {summary['zone_summary']['5']['n_buckets']} Zone 5 buckets")
    print(f"Zone 5 win rate: {summary['zone_summary']['5']['win_rate']:.4f}")
    print(f"Zone 5 avg market price: {summary['zone_summary']['5']['avg_market_price']:.4f}")
    print(f"Zone 5 calibration gap: {summary['zone_summary']['5']['calibration_gap']:.4f}")

    # Quick data summary
    has_temporal = sum(1 for e in events
                       if e["features"]["temporal"].get("rolling_avg_7d") is not None)
    has_cv = sum(1 for e in events if get_cv(e) is not None)
    print(f"\nEvents with temporal features: {has_temporal}")
    print(f"Events with CV available: {has_cv}")

    approaches = [
        "negbin",
        "gaussian",
        "market_baseline",
        "platt_calibrated",
        "tail_shrinkage",
        "vol_conditional",
        "multi_feature",
    ]

    results = []

    for i, name in enumerate(approaches):
        print(f"\n{'─' * 60}")
        print(f"[{i+1}/7] Running approach: {name}")
        print(f"{'─' * 60}")

        t0 = time.time()
        result = run_loocv(events, name)
        elapsed = time.time() - t0
        result["runtime_seconds"] = round(elapsed, 1)
        results.append(result)

        print(f"  Full Brier (mean): {result['full_brier_mean']:.6f}")
        print(f"  Full Brier (sum):  {result['full_brier_sum']:.4f}")
        print(f"  Z5 Brier (mean):   {result['z5_brier_mean']:.6f}")
        print(f"  Z5 Brier (sum):    {result['z5_brier_sum']:.4f}")
        print(f"  Runtime:           {elapsed:.1f}s")
        if "sample_params" in result:
            print(f"  Sample params:     {result['sample_params'][0]}")

    # Summary comparison
    baseline_result = next(r for r in results if r["approach"] == "market_baseline")
    baseline_full = baseline_result["full_brier_mean"]
    baseline_z5 = baseline_result["z5_brier_mean"]

    print(f"\n{'=' * 70}")
    print("SUMMARY - Sorted by Full Brier Score (lower = better)")
    print(f"{'=' * 70}")
    print(f"{'Approach':<25} {'Full Brier':>12} {'vs Baseline':>12} {'Z5 Brier':>12} {'Z5 vs Base':>12}")
    print(f"{'─' * 73}")

    sorted_results = sorted(results, key=lambda r: r["full_brier_mean"])
    for r in sorted_results:
        delta_full = r["full_brier_mean"] - baseline_full
        delta_z5 = r["z5_brier_mean"] - baseline_z5
        sign_full = "+" if delta_full >= 0 else ""
        sign_z5 = "+" if delta_z5 >= 0 else ""
        print(f"  {r['approach']:<23} {r['full_brier_mean']:>12.6f} {sign_full}{delta_full:>11.6f} "
              f"{r['z5_brier_mean']:>12.6f} {sign_z5}{delta_z5:>11.6f}")

    best = sorted_results[0]

    # Save results
    output = {
        "zone": TARGET_ZONE,
        "n_events": len(events),
        "n_zone_buckets": summary["zone_summary"]["5"]["n_buckets"],
        "n_zone_winners": summary["zone_summary"]["5"]["n_winners"],
        "zone_win_rate": summary["zone_summary"]["5"]["win_rate"],
        "zone_avg_market_price": summary["zone_summary"]["5"]["avg_market_price"],
        "zone_calibration_gap": summary["zone_summary"]["5"]["calibration_gap"],
        "approaches": results,
        "best_approach": best["approach"],
        "best_full_brier": best["full_brier_mean"],
        "baseline_full_brier": baseline_full,
        "improvement_vs_baseline": round(baseline_full - best["full_brier_mean"], 6),
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"\nBest approach: {best['approach']}")
    print(f"Best full Brier: {best['full_brier_mean']:.6f}")
    print(f"Baseline full Brier: {baseline_full:.6f}")
    print(f"Improvement: {baseline_full - best['full_brier_mean']:.6f}")


if __name__ == "__main__":
    main()
