"""
Zone 4 (Above Center) Bucket Analysis for Elon Tweet Markets
=============================================================
Zone 4 = buckets at relative_position 0.6-0.8 (above center).
Market avg price ~4.2%, win rate 4.3% (approximately calibrated).

7 approaches with LOOCV to find if we can beat market prices.
"""

import json
import math
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit, logit  # sigmoid and logit

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(r"G:\My Drive\AI_Projects\Ideas - In Progress\ElonTweetMarkets")
DATASET_PATH = BASE_DIR / "data" / "analysis" / "bucket_dataset.json"
OUTPUT_PATH = BASE_DIR / "data" / "analysis" / "zone_4_results.json"

PROB_FLOOR = 1e-6
TARGET_ZONE = 4


def load_dataset():
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)
    return data


def normalize_probs(probs):
    """Normalize list of probs to sum to 1.0, with floor."""
    probs = [max(p, PROB_FLOOR) for p in probs]
    total = sum(probs)
    if total <= 0:
        n = len(probs)
        return [1.0 / n] * n
    return [p / total for p in probs]


def brier_score(probs, outcomes):
    """Brier score for a single event (multi-outcome)."""
    assert len(probs) == len(outcomes)
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes))


def zone_brier_score(probs, outcomes, zones):
    """Brier score only for zone-4 buckets in this event."""
    total = 0.0
    count = 0
    for p, o, z in zip(probs, outcomes, zones):
        if z == TARGET_ZONE:
            total += (p - o) ** 2
            count += 1
    return total, count


# -------------------------------------------------------------------
# Approach 1: Negative Binomial from crowd stats
# -------------------------------------------------------------------
def approach_negbin(events):
    """Use crowd_implied_ev and crowd_std_dev to fit NegBin, compute bucket probs."""
    results = []
    for event in events:
        feat = event["features"]["market"]
        mu = feat["crowd_implied_ev"]
        sigma = feat["crowd_std_dev"]

        if mu is None or sigma is None or sigma <= 0 or mu <= 0:
            # Fallback to market prices
            results.append(None)
            continue

        var = sigma ** 2
        if var <= mu:
            # NegBin requires var > mean; fallback to Poisson
            # Use Poisson approximation
            dist = stats.poisson(mu)
            probs = []
            for b in event["buckets"]:
                lo, hi = b["lower_bound"], b["upper_bound"]
                hi_capped = min(hi, 99999)
                p = dist.cdf(hi_capped) - dist.cdf(lo - 1) if lo > 0 else dist.cdf(hi_capped)
                probs.append(max(p, PROB_FLOOR))
            results.append(normalize_probs(probs))
        else:
            # NegBin parameterization: n, p where mean = n*(1-p)/p, var = n*(1-p)/p^2
            # => p = mu / var, n = mu^2 / (var - mu)
            p_nb = mu / var
            n_nb = mu ** 2 / (var - mu)

            if n_nb <= 0 or p_nb <= 0 or p_nb >= 1:
                results.append(None)
                continue

            dist = stats.nbinom(n_nb, p_nb)
            probs = []
            for b in event["buckets"]:
                lo, hi = b["lower_bound"], b["upper_bound"]
                hi_capped = min(hi, 99999)
                prob = dist.cdf(hi_capped) - (dist.cdf(lo - 1) if lo > 0 else 0)
                probs.append(max(prob, PROB_FLOOR))
            results.append(normalize_probs(probs))

    return results


# -------------------------------------------------------------------
# Approach 2: Gaussian from crowd stats
# -------------------------------------------------------------------
def approach_gaussian(events):
    """Normal(crowd_implied_ev, crowd_std_dev) -> CDF differences."""
    results = []
    for event in events:
        feat = event["features"]["market"]
        mu = feat["crowd_implied_ev"]
        sigma = feat["crowd_std_dev"]

        if mu is None or sigma is None or sigma <= 0:
            results.append(None)
            continue

        dist = stats.norm(mu, sigma)
        probs = []
        for b in event["buckets"]:
            lo, hi = b["lower_bound"], b["upper_bound"]
            # Upper bound: use hi + 0.5 for continuity correction
            p_hi = dist.cdf(hi + 0.5) if hi < 99999 else 1.0
            p_lo = dist.cdf(lo - 0.5) if lo > 0 else 0.0
            prob = p_hi - p_lo
            probs.append(max(prob, PROB_FLOOR))
        results.append(normalize_probs(probs))

    return results


# -------------------------------------------------------------------
# Approach 3: Market baseline (no correction)
# -------------------------------------------------------------------
def approach_market_baseline(events):
    """Just use market prices as-is (normalized)."""
    results = []
    for event in events:
        probs = [b["market_price"] for b in event["buckets"]]
        results.append(normalize_probs(probs))
    return results


# -------------------------------------------------------------------
# Approach 4: Platt-calibrated market (LOOCV)
# -------------------------------------------------------------------
def approach_platt_calibrated(events):
    """
    For zone-4 buckets: P_corrected = sigmoid(a * logit(P_market) + b)
    Learn a, b via LOOCV (leave-one-event-out).
    """
    n = len(events)
    results = [None] * n

    # Pre-extract zone-4 data for each event
    event_zone4_data = []
    for event in events:
        z4 = []
        for b in event["buckets"]:
            if b["zone"] == TARGET_ZONE:
                mp = max(b["market_price"], PROB_FLOOR)
                mp = min(mp, 1 - PROB_FLOOR)
                z4.append((mp, b["is_winner"]))
        event_zone4_data.append(z4)

    for i in range(n):
        # Train on all except i
        train_x = []
        train_y = []
        for j in range(n):
            if j == i:
                continue
            for mp, win in event_zone4_data[j]:
                train_x.append(logit(mp))
                train_y.append(win)

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        if len(train_x) == 0 or train_y.sum() == 0:
            # No training data or no winners -> use market
            probs = [b["market_price"] for b in events[i]["buckets"]]
            results[i] = normalize_probs(probs)
            continue

        # Optimize a, b to minimize log loss on training set
        def neg_log_lik(params):
            a, b = params
            pred = expit(a * train_x + b)
            pred = np.clip(pred, PROB_FLOOR, 1 - PROB_FLOOR)
            ll = train_y * np.log(pred) + (1 - train_y) * np.log(1 - pred)
            return -np.mean(ll)

        res = minimize(neg_log_lik, [1.0, 0.0], method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-6})
        a_opt, b_opt = res.x

        # Apply to event i
        probs = []
        for b in events[i]["buckets"]:
            mp = max(b["market_price"], PROB_FLOOR)
            mp = min(mp, 1 - PROB_FLOOR)
            if b["zone"] == TARGET_ZONE:
                corrected = expit(a_opt * logit(mp) + b_opt)
                probs.append(max(corrected, PROB_FLOOR))
            else:
                probs.append(mp)
        results[i] = normalize_probs(probs)

    return results


# -------------------------------------------------------------------
# Approach 5: Market + upside shrinkage (LOOCV)
# -------------------------------------------------------------------
def approach_upside_shrinkage(events):
    """
    Since crowd captures 105% of upswings (slight overestimate of upside),
    shrink zone-4 prices: P_corrected = P_market * (1 - shrink_factor)
    Learn optimal shrink_factor via LOOCV.
    """
    n = len(events)
    results = [None] * n

    for i in range(n):
        # Train on all except i: find shrink_factor minimizing Brier
        def train_brier(shrink):
            total_brier = 0.0
            count = 0
            for j in range(n):
                if j == i:
                    continue
                probs = []
                outcomes = []
                for b in events[j]["buckets"]:
                    mp = max(b["market_price"], PROB_FLOOR)
                    if b["zone"] == TARGET_ZONE:
                        corrected = mp * (1 - shrink)
                        probs.append(max(corrected, PROB_FLOOR))
                    else:
                        probs.append(mp)
                    outcomes.append(b["is_winner"])
                probs = normalize_probs(probs)
                total_brier += brier_score(probs, outcomes)
                count += 1
            return total_brier / count if count > 0 else 999

        # Search shrink_factor in [0, 0.9]
        res = minimize_scalar(train_brier, bounds=(0.0, 0.9), method='bounded',
                              options={'maxiter': 200})
        best_shrink = res.x

        # Apply to event i
        probs = []
        for b in events[i]["buckets"]:
            mp = max(b["market_price"], PROB_FLOOR)
            if b["zone"] == TARGET_ZONE:
                corrected = mp * (1 - best_shrink)
                probs.append(max(corrected, PROB_FLOOR))
            else:
                probs.append(mp)
        results[i] = normalize_probs(probs)

    return results


# -------------------------------------------------------------------
# Approach 6: Market + momentum correction (LOOCV)
# -------------------------------------------------------------------
def approach_momentum(events):
    """
    If temporal features exist: use trend_7d.
    Otherwise: use price_shift_24h as momentum proxy.
    Positive momentum -> boost zone 4 (above center more likely).
    Negative momentum -> shrink zone 4.

    P_corrected = P_market * (1 + alpha * momentum_signal)
    Learn alpha via LOOCV.
    """
    n = len(events)
    results = [None] * n

    # Extract momentum signal for each event
    momentum_signals = []
    for event in events:
        temporal = event["features"]["temporal"]
        market = event["features"]["market"]

        # Prefer trend_7d if available (gold tier)
        if temporal.get("trend_7d") is not None:
            # Normalize trend by rolling avg to get % change
            avg = temporal.get("rolling_avg_7d")
            if avg and avg > 0:
                signal = temporal["trend_7d"] / avg
            else:
                signal = temporal["trend_7d"] / 100.0  # rough normalization
        elif market.get("price_shift_24h") is not None:
            # price_shift_24h is in absolute count units, normalize
            ev = market.get("crowd_implied_ev", 100)
            if ev and ev > 0:
                signal = market["price_shift_24h"] / ev
            else:
                signal = market["price_shift_24h"] / 100.0
        else:
            signal = 0.0

        momentum_signals.append(signal)

    for i in range(n):
        def train_brier(alpha):
            total_brier = 0.0
            count = 0
            for j in range(n):
                if j == i:
                    continue
                probs = []
                outcomes = []
                for b in events[j]["buckets"]:
                    mp = max(b["market_price"], PROB_FLOOR)
                    if b["zone"] == TARGET_ZONE:
                        correction = 1.0 + alpha * momentum_signals[j]
                        correction = max(correction, 0.01)  # prevent negative
                        corrected = mp * correction
                        probs.append(max(corrected, PROB_FLOOR))
                    else:
                        probs.append(mp)
                    outcomes.append(b["is_winner"])
                probs = normalize_probs(probs)
                total_brier += brier_score(probs, outcomes)
                count += 1
            return total_brier / count if count > 0 else 999

        res = minimize_scalar(train_brier, bounds=(-5.0, 5.0), method='bounded',
                              options={'maxiter': 200})
        best_alpha = res.x

        # Apply to event i
        probs = []
        for b in events[i]["buckets"]:
            mp = max(b["market_price"], PROB_FLOOR)
            if b["zone"] == TARGET_ZONE:
                correction = 1.0 + best_alpha * momentum_signals[i]
                correction = max(correction, 0.01)
                corrected = mp * correction
                probs.append(max(corrected, PROB_FLOOR))
            else:
                probs.append(mp)
        results[i] = normalize_probs(probs)

    return results


# -------------------------------------------------------------------
# Approach 7: Multi-feature model (LOOCV)
# -------------------------------------------------------------------
def approach_multi_feature(events):
    """
    Features for zone-4 adjustment multiplier:
    - z_score (bucket-level)
    - market_price (bucket-level)
    - crowd_skewness (positive skew -> more upside probability)
    - price_shift_24h (recent momentum)
    - elon_musk_vol_delta (news vol change)
    - launches_trailing_7d (busy launch week -> fewer tweets -> less zone 4)

    Model: P_corrected = P_market * exp(w . features)
    where features are standardized.
    Learn w via LOOCV with L2 regularization.
    """
    n = len(events)
    results = [None] * n

    # Pre-extract features for each event's zone-4 buckets
    event_z4_features = []  # list of (features_matrix, market_prices, outcomes)

    for event in events:
        feat = event["features"]
        market = feat["market"]
        media = feat.get("media", {})
        calendar = feat.get("calendar", {})

        z4_feats = []
        z4_prices = []
        z4_outcomes = []

        for b in event["buckets"]:
            if b["zone"] == TARGET_ZONE:
                # Extract features
                f = [
                    b.get("z_score", 0) or 0,
                    b.get("market_price", 0.04) or 0.04,
                    market.get("crowd_skewness", 0) or 0,
                    (market.get("price_shift_24h", 0) or 0) / max(market.get("crowd_implied_ev", 100) or 100, 1),
                    media.get("elon_musk_vol_delta", 0) or 0,
                    calendar.get("launches_trailing_7d", 0) or 0,
                ]
                z4_feats.append(f)
                z4_prices.append(max(b["market_price"], PROB_FLOOR))
                z4_outcomes.append(b["is_winner"])

        event_z4_features.append((
            np.array(z4_feats) if z4_feats else np.empty((0, 6)),
            z4_prices,
            z4_outcomes
        ))

    # Compute global mean/std for standardization
    all_feats = []
    for feats, _, _ in event_z4_features:
        if feats.shape[0] > 0:
            all_feats.append(feats)
    if all_feats:
        all_feats = np.vstack(all_feats)
        global_mean = np.mean(all_feats, axis=0)
        global_std = np.std(all_feats, axis=0)
        global_std[global_std < 1e-10] = 1.0
    else:
        global_mean = np.zeros(6)
        global_std = np.ones(6)

    reg_lambda = 1.0  # L2 regularization strength

    for i in range(n):
        # Collect training data (all events except i)
        train_feats = []
        train_prices = []
        train_outcomes = []

        for j in range(n):
            if j == i:
                continue
            feats, prices, outcomes = event_z4_features[j]
            if feats.shape[0] > 0:
                train_feats.append(feats)
                train_prices.extend(prices)
                train_outcomes.extend(outcomes)

        if not train_feats or sum(train_outcomes) == 0:
            # No zone-4 winners in training -> use market
            probs = [b["market_price"] for b in events[i]["buckets"]]
            results[i] = normalize_probs(probs)
            continue

        train_feats = np.vstack(train_feats)
        train_feats_std = (train_feats - global_mean) / global_std
        train_prices = np.array(train_prices)
        train_outcomes = np.array(train_outcomes)

        # Optimize weights: minimize Brier of adjusted probs + L2 penalty
        def obj(w):
            # P_corrected = P_market * exp(w . features)
            adjustments = np.exp(np.clip(train_feats_std @ w, -3, 3))
            pred = train_prices * adjustments
            pred = np.clip(pred, PROB_FLOOR, 1 - PROB_FLOOR)
            brier = np.mean((pred - train_outcomes) ** 2)
            penalty = reg_lambda * np.sum(w ** 2)
            return brier + penalty

        w0 = np.zeros(6)
        res = minimize(obj, w0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        w_opt = res.x

        # Apply to event i
        probs = []
        test_feats, test_prices, _ = event_z4_features[i]
        z4_idx = 0

        for b in events[i]["buckets"]:
            mp = max(b["market_price"], PROB_FLOOR)
            if b["zone"] == TARGET_ZONE and z4_idx < test_feats.shape[0]:
                f_std = (test_feats[z4_idx] - global_mean) / global_std
                adjustment = np.exp(np.clip(f_std @ w_opt, -3, 3))
                corrected = mp * adjustment
                probs.append(max(corrected, PROB_FLOOR))
                z4_idx += 1
            else:
                probs.append(mp)

        results[i] = normalize_probs(probs)

    return results


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate(events, predicted_probs, approach_name):
    """Evaluate predicted probabilities against outcomes."""
    full_brier_total = 0.0
    zone_brier_total = 0.0
    zone_count = 0
    n_events = 0
    events_beating_baseline = 0

    # Also compute baseline brier per event for comparison
    baseline_full_briers = []
    approach_full_briers = []

    for idx, event in enumerate(events):
        if predicted_probs[idx] is None:
            continue

        probs = predicted_probs[idx]
        outcomes = [b["is_winner"] for b in event["buckets"]]
        zones = [b["zone"] for b in event["buckets"]]

        # Baseline: market prices
        baseline_probs = normalize_probs([b["market_price"] for b in event["buckets"]])

        # Full event Brier
        fb = brier_score(probs, outcomes)
        fb_base = brier_score(baseline_probs, outcomes)

        full_brier_total += fb
        baseline_full_briers.append(fb_base)
        approach_full_briers.append(fb)
        n_events += 1

        if fb < fb_base:
            events_beating_baseline += 1

        # Zone-4 Brier
        zb, zc = zone_brier_score(probs, outcomes, zones)
        zone_brier_total += zb
        zone_count += zc

    avg_full_brier = full_brier_total / n_events if n_events > 0 else 999
    avg_zone_brier = zone_brier_total / zone_count if zone_count > 0 else 999
    avg_baseline = sum(baseline_full_briers) / len(baseline_full_briers) if baseline_full_briers else 999

    return {
        "approach": approach_name,
        "n_events_evaluated": n_events,
        "full_brier": round(avg_full_brier, 6),
        "zone_4_brier": round(avg_zone_brier, 6),
        "baseline_full_brier": round(avg_baseline, 6),
        "brier_improvement": round(avg_baseline - avg_full_brier, 6),
        "events_beating_baseline": events_beating_baseline,
        "pct_events_beating_baseline": round(events_beating_baseline / n_events * 100, 1) if n_events > 0 else 0,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Zone 4 (Above Center) Bucket Analysis")
    print("=" * 70)

    data = load_dataset()
    events = data["events"]
    print(f"\nLoaded {len(events)} events")

    # Count zone-4 stats
    z4_total = 0
    z4_winners = 0
    z4_price_sum = 0
    for event in events:
        for b in event["buckets"]:
            if b["zone"] == TARGET_ZONE:
                z4_total += 1
                z4_winners += b["is_winner"]
                z4_price_sum += b["market_price"]

    print(f"Zone 4 buckets: {z4_total}")
    print(f"Zone 4 winners: {z4_winners} ({z4_winners/z4_total*100:.1f}%)")
    print(f"Zone 4 avg market price: {z4_price_sum/z4_total:.4f}")

    # Gold tier count
    gold_count = sum(1 for e in events if e["ground_truth_tier"] == "gold")
    print(f"Gold tier events (have temporal features): {gold_count}")

    # ---------------------------------------------------------------
    # Run all 7 approaches
    # ---------------------------------------------------------------
    approaches = {}

    print("\n" + "-" * 70)
    print("Running Approach 1: Negative Binomial from crowd stats...")
    approaches["1_negbin"] = approach_negbin(events)
    fallback_count = sum(1 for x in approaches["1_negbin"] if x is None)
    print(f"  Computed for {len(events) - fallback_count}/{len(events)} events")
    # Fill None with market baseline
    baseline = approach_market_baseline(events)
    for idx in range(len(events)):
        if approaches["1_negbin"][idx] is None:
            approaches["1_negbin"][idx] = baseline[idx]

    print("Running Approach 2: Gaussian from crowd stats...")
    approaches["2_gaussian"] = approach_gaussian(events)
    fallback_count = sum(1 for x in approaches["2_gaussian"] if x is None)
    print(f"  Computed for {len(events) - fallback_count}/{len(events)} events")
    for idx in range(len(events)):
        if approaches["2_gaussian"][idx] is None:
            approaches["2_gaussian"][idx] = baseline[idx]

    print("Running Approach 3: Market baseline...")
    approaches["3_market_baseline"] = baseline

    print("Running Approach 4: Platt-calibrated market (LOOCV)...")
    approaches["4_platt_calibrated"] = approach_platt_calibrated(events)

    print("Running Approach 5: Market + upside shrinkage (LOOCV)...")
    approaches["5_upside_shrinkage"] = approach_upside_shrinkage(events)

    print("Running Approach 6: Market + momentum correction (LOOCV)...")
    approaches["6_momentum"] = approach_momentum(events)

    print("Running Approach 7: Multi-feature model (LOOCV)...")
    approaches["7_multi_feature"] = approach_multi_feature(events)

    # ---------------------------------------------------------------
    # Evaluate all
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    approach_names = {
        "1_negbin": "NegBin from Crowd Stats",
        "2_gaussian": "Gaussian from Crowd Stats",
        "3_market_baseline": "Market Baseline (no correction)",
        "4_platt_calibrated": "Platt-Calibrated Market",
        "5_upside_shrinkage": "Market + Upside Shrinkage",
        "6_momentum": "Market + Momentum Correction",
        "7_multi_feature": "Multi-Feature Model",
    }

    all_results = []
    for key, name in approach_names.items():
        result = evaluate(events, approaches[key], name)
        all_results.append(result)

        marker = " ***" if result["brier_improvement"] > 0 else ""
        print(f"\n  {name}:")
        print(f"    Full Brier:     {result['full_brier']:.6f}  (baseline: {result['baseline_full_brier']:.6f})")
        print(f"    Zone-4 Brier:   {result['zone_4_brier']:.6f}")
        print(f"    Improvement:    {result['brier_improvement']:+.6f}{marker}")
        print(f"    Events beating: {result['events_beating_baseline']}/{result['n_events_evaluated']} ({result['pct_events_beating_baseline']:.1f}%)")

    # Find best
    best = min(all_results, key=lambda r: r["full_brier"])
    baseline_result = [r for r in all_results if r["approach"] == "Market Baseline (no correction)"][0]

    print("\n" + "=" * 70)
    print(f"BEST APPROACH: {best['approach']}")
    print(f"  Full Brier:   {best['full_brier']:.6f} vs baseline {baseline_result['full_brier']:.6f}")
    print(f"  Improvement:  {best['brier_improvement']:+.6f}")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Detailed analysis: what does Zone 4 look like per approach?
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ZONE 4 DETAILED BREAKDOWN")
    print("-" * 70)

    for key, name in approach_names.items():
        probs_list = approaches[key]
        z4_pred = []
        z4_actual = []
        for idx, event in enumerate(events):
            if probs_list[idx] is None:
                continue
            for bi, b in enumerate(event["buckets"]):
                if b["zone"] == TARGET_ZONE:
                    z4_pred.append(probs_list[idx][bi])
                    z4_actual.append(b["is_winner"])

        z4_pred = np.array(z4_pred)
        z4_actual = np.array(z4_actual)

        if len(z4_pred) > 0:
            avg_pred = np.mean(z4_pred)
            avg_actual = np.mean(z4_actual)
            print(f"\n  {name}:")
            print(f"    Avg predicted P(win):  {avg_pred:.4f}")
            print(f"    Actual win rate:       {avg_actual:.4f}")
            print(f"    Calibration gap:       {avg_actual - avg_pred:+.4f}")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    output = {
        "zone": TARGET_ZONE,
        "zone_description": "Above Center (relative position 0.6-0.8)",
        "n_events": len(events),
        "n_zone_buckets": z4_total,
        "n_zone_winners": z4_winners,
        "zone_win_rate": round(z4_winners / z4_total, 4),
        "zone_avg_market_price": round(z4_price_sum / z4_total, 4),
        "approaches": all_results,
        "best_approach": best["approach"],
        "best_full_brier": best["full_brier"],
        "baseline_full_brier": baseline_result["full_brier"],
        "improvement_over_baseline": round(best["full_brier"] - baseline_result["full_brier"], 6),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
