"""
Deep Exploratory Data Analysis of ElonTweetMarkets Backtest Dataset
====================================================================
Analyzes temporal, GDELT, market, and regime features to find patterns
that could help build a model that beats the crowd (market prices).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(r"G:\My Drive\AI_Projects\Ideas - In Progress\ElonTweetMarkets")
BACKTEST_DIR = BASE_DIR / "data" / "backtest"

# ============================================================================
# SECTION 0: Load All Gold-Tier Events
# ============================================================================
def load_gold_events():
    """Load all gold-tier events with features and metadata."""
    idx = json.loads((BACKTEST_DIR / "backtest_index.json").read_text())
    gold = [e for e in idx["events"] if e["ground_truth_tier"] == "gold"]

    events = []
    for e in gold:
        out_dir = Path(e["output_dir"])
        meta_path = out_dir / "metadata.json"
        feat_path = out_dir / "features.json"
        prices_path = out_dir / "prices.parquet"

        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        feat = json.loads(feat_path.read_text()) if feat_path.exists() else {}

        record = {
            "slug": e["event_slug"],
            "start_date": e["start_date"],
            "end_date": e["end_date"],
            "market_type": e["market_type"],
            "xtracker_count": e["xtracker_count"],
            "winning_bucket": e["winning_bucket"],
            "has_temporal": e.get("has_temporal_features", False),
            "has_market": e.get("has_market_features", False),
            "duration_days": meta.get("duration_days"),
            "n_buckets": meta.get("n_buckets"),
        }

        # Flatten features
        for group_name in ["temporal", "gdelt", "spacex", "market"]:
            group = feat.get(group_name, {})
            if isinstance(group, dict):
                for k, v in group.items():
                    record[f"{group_name}__{k}"] = v

        # Parse bucket boundaries from metadata
        if "buckets" in meta:
            record["buckets"] = meta["buckets"]

        events.append(record)

    df = pd.DataFrame(events)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_subheader(title):
    print(f"\n--- {title} ---\n")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def main():
    df = load_gold_events()

    print_header("SECTION 0: DATASET OVERVIEW")
    print(f"Total gold events: {len(df)}")
    print(f"With temporal features: {df['has_temporal'].sum()}")
    print(f"With market features: {df['has_market'].sum()}")
    print(f"Market types: {df['market_type'].value_counts().to_dict()}")
    print(f"Date range: {df['start_date'].min()} to {df['end_date'].max()}")
    print(f"\nXTracker count stats:")
    print(f"  Mean:   {df['xtracker_count'].mean():.1f}")
    print(f"  Median: {df['xtracker_count'].median():.1f}")
    print(f"  Std:    {df['xtracker_count'].std():.1f}")
    print(f"  Min:    {df['xtracker_count'].min():.1f}")
    print(f"  Max:    {df['xtracker_count'].max():.1f}")
    print(f"  CV:     {df['xtracker_count'].std() / df['xtracker_count'].mean():.3f}")

    # By market type
    print_subheader("XTracker Count by Market Type")
    for mtype in df["market_type"].unique():
        sub = df[df["market_type"] == mtype]
        print(f"  {mtype:8s}: n={len(sub):3d}, mean={sub['xtracker_count'].mean():7.1f}, "
              f"std={sub['xtracker_count'].std():7.1f}, "
              f"range=[{sub['xtracker_count'].min():.0f}, {sub['xtracker_count'].max():.0f}]")

    # ========================================================================
    # SECTION 1: TEMPORAL FEATURE ANALYSIS
    # ========================================================================
    print_header("SECTION 1: TEMPORAL FEATURE ANALYSIS")

    temporal_df = df[df["has_temporal"]].copy()
    print(f"Events with temporal features: {len(temporal_df)}")

    # For weekly events, compute daily rate
    temporal_df["daily_rate"] = temporal_df["xtracker_count"] / temporal_df["duration_days"]

    # 1a. Rolling average correlation with actual count
    print_subheader("1a. Rolling Average vs Actual Count Correlation")

    # We need to account for duration: rolling avg is daily, actual is total
    # So compare rolling_avg * duration vs actual
    for avg_col in ["temporal__rolling_avg_7d", "temporal__rolling_avg_14d", "temporal__rolling_avg_28d"]:
        valid = temporal_df.dropna(subset=[avg_col, "xtracker_count"])
        if len(valid) < 3:
            continue
        # Predicted total = rolling_avg_daily * duration
        predicted = valid[avg_col] * valid["duration_days"]
        actual = valid["xtracker_count"]
        corr = predicted.corr(actual)
        mae = (predicted - actual).abs().mean()
        mape = ((predicted - actual).abs() / actual).mean() * 100
        bias = (predicted - actual).mean()
        print(f"  {avg_col.split('__')[1]:16s}: corr={corr:.4f}, MAE={mae:.1f}, MAPE={mape:.1f}%, bias={bias:+.1f}")

    # 1b. Deviation from rolling averages
    print_subheader("1b. How Much Does Actual Deviate From Rolling Averages?")

    for avg_col, label in [("temporal__rolling_avg_7d", "7d avg"),
                            ("temporal__rolling_avg_14d", "14d avg")]:
        valid = temporal_df.dropna(subset=[avg_col]).copy()
        if len(valid) < 3:
            continue
        predicted = valid[avg_col] * valid["duration_days"]
        actual = valid["xtracker_count"]
        pct_deviation = ((actual - predicted) / predicted) * 100

        print(f"  {label} -> actual % deviation:")
        print(f"    Mean deviation:   {pct_deviation.mean():+.1f}%")
        print(f"    Std deviation:    {pct_deviation.std():.1f}%")
        print(f"    Median deviation: {pct_deviation.median():+.1f}%")
        print(f"    Range: [{pct_deviation.min():+.1f}%, {pct_deviation.max():+.1f}%]")

        # How often does it undershoot vs overshoot?
        over = (pct_deviation > 0).sum()
        under = (pct_deviation < 0).sum()
        print(f"    Overshoot: {over}/{len(pct_deviation)} ({over/len(pct_deviation)*100:.0f}%), "
              f"Undershoot: {under}/{len(pct_deviation)} ({under/len(pct_deviation)*100:.0f}%)")

    # 1c. Autocorrelation / Momentum vs Mean Reversion
    print_subheader("1c. Autocorrelation: Momentum vs Mean Reversion")

    # Focus on weekly events sorted by date
    weekly = temporal_df[temporal_df["market_type"] == "weekly"].sort_values("start_date").copy()
    if len(weekly) > 3:
        weekly["daily_rate"] = weekly["xtracker_count"] / weekly["duration_days"]

        # Lag-1 autocorrelation of daily rates
        rates = weekly["daily_rate"].values
        if len(rates) > 2:
            from scipy.stats import pearsonr
            r, p = pearsonr(rates[:-1], rates[1:])
            print(f"  Lag-1 autocorrelation of daily tweet rate (weekly markets):")
            print(f"    r = {r:.4f}, p = {p:.4f}")
            print(f"    Interpretation: {'Momentum (positive)' if r > 0 else 'Mean reversion (negative)'}")

        # Lag-1 of count levels
        counts = weekly["xtracker_count"].values
        if len(counts) > 2:
            r2, p2 = pearsonr(counts[:-1], counts[1:])
            print(f"  Lag-1 autocorrelation of total weekly counts:")
            print(f"    r = {r2:.4f}, p = {p2:.4f}")

        # Week-over-week changes
        changes = np.diff(rates)
        if len(changes) > 2:
            r3, p3 = pearsonr(changes[:-1], changes[1:])
            print(f"  Lag-1 autocorrelation of CHANGES in daily rate:")
            print(f"    r = {r3:.4f}, p = {p3:.4f}")
            print(f"    Interpretation: {'Trend continuation' if r3 > 0 else 'Mean reversion in changes'}")

    # 1d. Does trend_7d/trend_14d predict direction?
    print_subheader("1d. Trend Features vs Actual Direction")

    for trend_col in ["temporal__trend_7d", "temporal__trend_14d"]:
        valid = temporal_df.dropna(subset=[trend_col, "temporal__rolling_avg_7d"]).copy()
        if len(valid) < 5:
            print(f"  {trend_col}: only {len(valid)} events with non-null trend, skipping")
            continue
        # Did the actual overshoot or undershoot the rolling avg?
        predicted = valid["temporal__rolling_avg_7d"] * valid["duration_days"]
        actual_direction = np.sign(valid["xtracker_count"] - predicted)
        trend_direction = np.sign(valid[trend_col])
        agree = (actual_direction == trend_direction).sum()
        total = len(valid)
        print(f"  {trend_col.split('__')[1]:10s}: trend direction matches actual direction "
              f"in {agree}/{total} cases ({agree/total*100:.0f}%)")

    # 1e. Yesterday's count as predictor
    print_subheader("1e. Yesterday Count vs Daily Rate")
    valid = temporal_df.dropna(subset=["temporal__yesterday_count"]).copy()
    if len(valid) > 3:
        corr = valid["temporal__yesterday_count"].corr(valid["daily_rate"])
        print(f"  Corr(yesterday_count, actual_daily_rate) = {corr:.4f}")

        # yesterday * duration vs actual
        predicted = valid["temporal__yesterday_count"] * valid["duration_days"]
        actual = valid["xtracker_count"]
        mae = (predicted - actual).abs().mean()
        mape = ((predicted - actual).abs() / actual).mean() * 100
        bias = (predicted - actual).mean()
        print(f"  If we just use yesterday*duration: MAE={mae:.1f}, MAPE={mape:.1f}%, bias={bias:+.1f}")

    # 1f. Coefficient of Variation as signal
    print_subheader("1f. CV (Coefficient of Variation) as Regime Uncertainty Signal")
    valid = temporal_df.dropna(subset=["temporal__cv_14d"]).copy()
    if len(valid) > 3:
        # High CV = uncertain regime. Do high-CV periods have bigger prediction errors?
        predicted = valid["temporal__rolling_avg_7d"] * valid["duration_days"]
        abs_pct_error = ((valid["xtracker_count"] - predicted) / predicted).abs() * 100

        median_cv = valid["temporal__cv_14d"].median()
        high_cv = abs_pct_error[valid["temporal__cv_14d"] > median_cv]
        low_cv = abs_pct_error[valid["temporal__cv_14d"] <= median_cv]

        print(f"  Median CV_14d: {median_cv:.4f}")
        print(f"  High CV events (>{median_cv:.3f}): mean abs pct error = {high_cv.mean():.1f}% (n={len(high_cv)})")
        print(f"  Low CV events  (<={median_cv:.3f}): mean abs pct error = {low_cv.mean():.1f}% (n={len(low_cv)})")
        print(f"  CV_14d corr with abs_pct_error: {valid['temporal__cv_14d'].corr(abs_pct_error):.4f}")

    # ========================================================================
    # SECTION 2: GDELT FEATURE ANALYSIS
    # ========================================================================
    print_header("SECTION 2: GDELT FEATURE ANALYSIS")

    gdelt_cols = [c for c in df.columns if c.startswith("gdelt__")]
    valid_gdelt = df.dropna(subset=["temporal__rolling_avg_7d"] + [c for c in gdelt_cols if c in df.columns])

    if len(valid_gdelt) > 5:
        print(f"Events with GDELT features: {len(valid_gdelt)}")

        # 2a. Correlation of GDELT volumes with daily rate
        print_subheader("2a. GDELT Volume Correlations with Daily Tweet Rate")

        valid_gdelt_rate = valid_gdelt.copy()
        valid_gdelt_rate["daily_rate"] = valid_gdelt_rate["xtracker_count"] / valid_gdelt_rate["duration_days"]

        for col in gdelt_cols:
            if col in valid_gdelt_rate.columns:
                vals = valid_gdelt_rate[col].dropna()
                if len(vals) > 5:
                    corr = vals.corr(valid_gdelt_rate.loc[vals.index, "daily_rate"])
                    print(f"  {col.split('__')[1]:25s}: corr={corr:+.4f} (n={len(vals)})")

        # 2b. GDELT delta (change) as leading indicator
        print_subheader("2b. GDELT Volume Deltas as Leading Indicators")
        delta_cols = [c for c in gdelt_cols if "delta" in c]

        for col in delta_cols:
            valid_sub = valid_gdelt_rate.dropna(subset=[col])
            if len(valid_sub) > 5:
                # Does positive GDELT delta predict higher-than-average tweet rate?
                predicted = valid_sub["temporal__rolling_avg_7d"] * valid_sub["duration_days"]
                surprise = valid_sub["xtracker_count"] - predicted
                corr = valid_sub[col].corr(surprise)
                print(f"  {col.split('__')[1]:25s}: corr with count surprise = {corr:+.4f}")

        # 2c. GDELT tone as predictor
        print_subheader("2c. GDELT Tone Features")
        tone_cols = [c for c in gdelt_cols if "tone" in c]
        for col in tone_cols:
            valid_sub = valid_gdelt_rate.dropna(subset=[col])
            if len(valid_sub) > 5:
                corr = valid_sub[col].corr(valid_sub["daily_rate"])
                print(f"  {col.split('__')[1]:25s}: corr with daily_rate = {corr:+.4f} (n={len(valid_sub)})")
    else:
        print("Not enough events with both GDELT and temporal features for analysis.")

    # ========================================================================
    # SECTION 3: SPACEX FEATURE ANALYSIS
    # ========================================================================
    print_header("SECTION 3: SPACEX FEATURE ANALYSIS")

    spacex_cols = [c for c in df.columns if c.startswith("spacex__")]
    valid_spacex = df.dropna(subset=["temporal__rolling_avg_7d"]).copy()
    valid_spacex["daily_rate"] = valid_spacex["xtracker_count"] / valid_spacex["duration_days"]

    if len(valid_spacex) > 5:
        print_subheader("3a. SpaceX Launch Correlations")
        for col in spacex_cols:
            valid_sub = valid_spacex.dropna(subset=[col])
            if len(valid_sub) > 5:
                corr = valid_sub[col].corr(valid_sub["daily_rate"])
                print(f"  {col.split('__')[1]:25s}: corr with daily_rate = {corr:+.4f} (n={len(valid_sub)})")

        # Does launches_during_event predict surprise?
        print_subheader("3b. SpaceX Launches vs Count Surprise")
        valid_sub = valid_spacex.dropna(subset=["spacex__launches_during_event"])
        if len(valid_sub) > 5:
            predicted = valid_sub["temporal__rolling_avg_7d"] * valid_sub["duration_days"]
            surprise = valid_sub["xtracker_count"] - predicted
            corr = valid_sub["spacex__launches_during_event"].corr(surprise)
            print(f"  launches_during_event corr with count surprise: {corr:+.4f}")

    # ========================================================================
    # SECTION 4: MARKET EFFICIENCY ANALYSIS
    # ========================================================================
    print_header("SECTION 4: MARKET EFFICIENCY ANALYSIS (Crowd vs Actual)")

    market_df = df[df["has_market"]].copy()
    print(f"Events with market features: {len(market_df)}")

    if len(market_df) > 3:
        # 4a. Crowd implied EV vs actual
        print_subheader("4a. Crowd Implied EV vs Actual Count")

        valid = market_df.dropna(subset=["market__crowd_implied_ev"]).copy()
        crowd_ev = valid["market__crowd_implied_ev"]
        actual = valid["xtracker_count"]

        corr = crowd_ev.corr(actual)
        mae = (crowd_ev - actual).abs().mean()
        mape = ((crowd_ev - actual).abs() / actual).mean() * 100
        bias = (crowd_ev - actual).mean()
        rmse = np.sqrt(((crowd_ev - actual) ** 2).mean())

        print(f"  Correlation: {corr:.4f}")
        print(f"  MAE:  {mae:.1f}")
        print(f"  RMSE: {rmse:.1f}")
        print(f"  MAPE: {mape:.1f}%")
        print(f"  Bias: {bias:+.1f} (positive = crowd overestimates)")

        # 4b. By market type
        print_subheader("4b. Crowd Accuracy by Market Type")
        for mtype in valid["market_type"].unique():
            sub = valid[valid["market_type"] == mtype]
            if len(sub) < 2:
                continue
            c_ev = sub["market__crowd_implied_ev"]
            act = sub["xtracker_count"]
            sub_mae = (c_ev - act).abs().mean()
            sub_mape = ((c_ev - act).abs() / act).mean() * 100
            sub_bias = (c_ev - act).mean()
            sub_corr = c_ev.corr(act)
            print(f"  {mtype:8s}: n={len(sub):3d}, corr={sub_corr:.4f}, MAE={sub_mae:.1f}, "
                  f"MAPE={sub_mape:.1f}%, bias={sub_bias:+.1f}")

        # 4c. Where does the crowd systematically err?
        print_subheader("4c. Crowd Error Analysis - Systematic Biases")

        valid["crowd_error"] = valid["market__crowd_implied_ev"] - valid["xtracker_count"]
        valid["crowd_error_pct"] = valid["crowd_error"] / valid["xtracker_count"] * 100

        # Split into quintiles by actual count
        valid["count_quintile"] = pd.qcut(valid["xtracker_count"], q=min(5, len(valid)//3),
                                           labels=False, duplicates='drop')

        print("  Crowd error by actual count quintile:")
        for q in sorted(valid["count_quintile"].dropna().unique()):
            sub = valid[valid["count_quintile"] == q]
            print(f"    Q{int(q)}: actual_range=[{sub['xtracker_count'].min():.0f}, {sub['xtracker_count'].max():.0f}], "
                  f"mean_error={sub['crowd_error'].mean():+.1f}, "
                  f"mean_error_pct={sub['crowd_error_pct'].mean():+.1f}%")

        # 4d. Does crowd under/overpredict based on recent trend?
        print_subheader("4d. Crowd Error vs Recent Trend")

        valid_trend = valid.dropna(subset=["temporal__rolling_avg_7d"])
        if len(valid_trend) > 5:
            # regime_ratio > 1 means recent > long-term (uptrend)
            valid_trend_r = valid_trend.dropna(subset=["temporal__regime_ratio"])
            if len(valid_trend_r) > 5:
                up = valid_trend_r[valid_trend_r["temporal__regime_ratio"] > 1.0]
                down = valid_trend_r[valid_trend_r["temporal__regime_ratio"] <= 1.0]
                print(f"  During UPTREND periods (regime_ratio > 1): n={len(up)}")
                if len(up) > 0:
                    print(f"    Mean crowd error: {up['crowd_error'].mean():+.1f} ({up['crowd_error_pct'].mean():+.1f}%)")
                print(f"  During DOWNTREND periods (regime_ratio <= 1): n={len(down)}")
                if len(down) > 0:
                    print(f"    Mean crowd error: {down['crowd_error'].mean():+.1f} ({down['crowd_error_pct'].mean():+.1f}%)")

        # 4e. Can we use rolling avg + duration to beat the crowd?
        print_subheader("4e. Simple Model Comparison: Rolling Avg * Duration vs Crowd EV")

        valid_comp = valid.dropna(subset=["temporal__rolling_avg_7d"]).copy()
        if len(valid_comp) > 3:
            for avg_col, label in [("temporal__rolling_avg_7d", "7d_avg"),
                                    ("temporal__rolling_avg_14d", "14d_avg"),
                                    ("temporal__rolling_avg_28d", "28d_avg")]:
                sub = valid_comp.dropna(subset=[avg_col])
                if len(sub) < 3:
                    continue
                model_pred = sub[avg_col] * sub["duration_days"]
                crowd_pred = sub["market__crowd_implied_ev"]
                actual_vals = sub["xtracker_count"]

                model_mae = (model_pred - actual_vals).abs().mean()
                crowd_mae = (crowd_pred - actual_vals).abs().mean()
                model_mape = ((model_pred - actual_vals).abs() / actual_vals).mean() * 100
                crowd_mape = ((crowd_pred - actual_vals).abs() / actual_vals).mean() * 100

                print(f"  {label:8s}: model_MAE={model_mae:.1f} vs crowd_MAE={crowd_mae:.1f} "
                      f"({'+' if model_mae > crowd_mae else '-'}{abs(model_mae - crowd_mae):.1f}), "
                      f"model_MAPE={model_mape:.1f}% vs crowd_MAPE={crowd_mape:.1f}%")

        # 4f. Crowd confidence vs accuracy
        print_subheader("4f. Crowd Confidence (Entropy / Std Dev) vs Accuracy")

        for conf_col, label in [("market__distribution_entropy", "entropy"),
                                 ("market__crowd_std_dev", "crowd_std_dev")]:
            valid_sub = valid.dropna(subset=[conf_col]).copy()
            if len(valid_sub) < 5:
                continue
            abs_error = valid_sub["crowd_error"].abs()
            corr = valid_sub[conf_col].corr(abs_error)
            print(f"  {label:15s} corr with |crowd_error|: {corr:+.4f}")

            median_conf = valid_sub[conf_col].median()
            high = abs_error[valid_sub[conf_col] > median_conf]
            low = abs_error[valid_sub[conf_col] <= median_conf]
            print(f"    High {label}: mean |error| = {high.mean():.1f} (n={len(high)})")
            print(f"    Low  {label}: mean |error| = {low.mean():.1f} (n={len(low)})")

    # ========================================================================
    # SECTION 5: REGIME ANALYSIS
    # ========================================================================
    print_header("SECTION 5: REGIME ANALYSIS")

    # Focus on weekly events for regime analysis (most comparable)
    weekly = df[df["market_type"] == "weekly"].sort_values("start_date").copy()
    print(f"Weekly events for regime analysis: {len(weekly)}")

    if len(weekly) > 5:
        weekly["daily_rate"] = weekly["xtracker_count"] / weekly["duration_days"]

        # 5a. Classify into regimes
        print_subheader("5a. Regime Classification (by daily tweet rate)")

        rates = weekly["daily_rate"].values
        q33 = np.percentile(rates, 33)
        q67 = np.percentile(rates, 67)

        weekly["regime"] = "medium"
        weekly.loc[weekly["daily_rate"] <= q33, "regime"] = "low"
        weekly.loc[weekly["daily_rate"] >= q67, "regime"] = "high"

        print(f"  Regime thresholds: low <= {q33:.1f} tweets/day, high >= {q67:.1f} tweets/day")
        for regime in ["low", "medium", "high"]:
            sub = weekly[weekly["regime"] == regime]
            print(f"  {regime:7s}: n={len(sub)}, mean_daily_rate={sub['daily_rate'].mean():.1f}, "
                  f"mean_count={sub['xtracker_count'].mean():.1f}, "
                  f"range=[{sub['xtracker_count'].min():.0f}-{sub['xtracker_count'].max():.0f}]")

        # 5b. Regime transitions
        print_subheader("5b. Regime Transitions")

        regimes = weekly["regime"].values
        transitions = defaultdict(int)
        for i in range(len(regimes) - 1):
            transitions[f"{regimes[i]} -> {regimes[i+1]}"] += 1

        print("  Transition counts:")
        for trans, count in sorted(transitions.items(), key=lambda x: -x[1]):
            print(f"    {trans}: {count}")

        # Transition matrix as probabilities
        print("\n  Transition probabilities:")
        for from_regime in ["low", "medium", "high"]:
            from_events = [i for i in range(len(regimes)-1) if regimes[i] == from_regime]
            if not from_events:
                continue
            total = len(from_events)
            for to_regime in ["low", "medium", "high"]:
                count = sum(1 for i in from_events if regimes[i+1] == to_regime)
                print(f"    P({to_regime:7s} | {from_regime:7s}) = {count/total:.2f} ({count}/{total})")

        # 5c. Does the crowd lag regime shifts?
        print_subheader("5c. Crowd Lag During Regime Shifts")

        weekly_market = weekly[weekly["has_market"]].copy()
        if len(weekly_market) > 5:
            valid_m = weekly_market.dropna(subset=["market__crowd_implied_ev"])
            valid_m = valid_m.copy()
            valid_m["crowd_error"] = valid_m["market__crowd_implied_ev"] - valid_m["xtracker_count"]
            valid_m["crowd_error_pct"] = valid_m["crowd_error"] / valid_m["xtracker_count"] * 100

            # After regime changes
            regimes_m = valid_m["regime"].values
            dates_m = valid_m["start_date"].values

            # Identify transition events (regime different from previous)
            transition_idx = []
            same_idx = []
            for i in range(1, len(regimes_m)):
                if regimes_m[i] != regimes_m[i-1]:
                    transition_idx.append(valid_m.index[i])
                else:
                    same_idx.append(valid_m.index[i])

            if transition_idx and same_idx:
                trans_errors = valid_m.loc[transition_idx, "crowd_error_pct"].abs()
                same_errors = valid_m.loc[same_idx, "crowd_error_pct"].abs()
                print(f"  During regime TRANSITIONS: mean |error| = {trans_errors.mean():.1f}% (n={len(trans_errors)})")
                print(f"  During regime CONTINUATIONS: mean |error| = {same_errors.mean():.1f}% (n={len(same_errors)})")
                print(f"  --> Crowd error is {trans_errors.mean()/same_errors.mean():.2f}x larger during transitions")

                # Specifically: which transitions cause biggest errors?
                print("\n  Crowd error by transition type:")
                for i in range(1, len(regimes_m)):
                    idx = valid_m.index[i]
                    if regimes_m[i] != regimes_m[i-1]:
                        trans = f"{regimes_m[i-1]} -> {regimes_m[i]}"
                        err = valid_m.loc[idx, "crowd_error_pct"]
                        ev = valid_m.loc[idx, "market__crowd_implied_ev"]
                        act = valid_m.loc[idx, "xtracker_count"]
                        print(f"    {valid_m.loc[idx, 'start_date'].date()} {trans:20s}: "
                              f"crowd_ev={ev:.0f}, actual={act:.0f}, error={err:+.1f}%")

        # 5d. Can trailing features detect regimes?
        print_subheader("5d. Can Trailing Features Predict Regime?")

        weekly_feat = weekly.dropna(subset=["temporal__rolling_avg_7d"]).copy()
        if len(weekly_feat) > 5:
            # Which features distinguish regimes?
            feat_cols = [c for c in weekly_feat.columns if c.startswith("temporal__") or c.startswith("gdelt__")]

            print("  Feature means by regime:")
            for col in feat_cols:
                valid_col = weekly_feat.dropna(subset=[col])
                if len(valid_col) < 5:
                    continue
                means = {}
                for regime in ["low", "medium", "high"]:
                    sub = valid_col[valid_col["regime"] == regime]
                    if len(sub) > 0:
                        means[regime] = sub[col].mean()

                if len(means) >= 2:
                    spread = max(means.values()) - min(means.values())
                    overall_std = valid_col[col].std()
                    if overall_std > 0:
                        effect_size = spread / overall_std
                        if effect_size > 0.5:  # Only show meaningful ones
                            print(f"    {col.split('__')[1]:25s}: "
                                  f"low={means.get('low', 'N/A'):>8.2f}, "
                                  f"med={means.get('medium', 'N/A'):>8.2f}, "
                                  f"high={means.get('high', 'N/A'):>8.2f}, "
                                  f"effect_size={effect_size:.2f}")

    # ========================================================================
    # SECTION 6: FEATURE IMPORTANCE RANKING
    # ========================================================================
    print_header("SECTION 6: OVERALL FEATURE IMPORTANCE")

    # Compute correlation of every numeric feature with daily_rate and with crowd error
    all_feat = df.dropna(subset=["temporal__rolling_avg_7d"]).copy()
    all_feat["daily_rate"] = all_feat["xtracker_count"] / all_feat["duration_days"]
    all_feat["crowd_error"] = all_feat.get("market__crowd_implied_ev", np.nan) - all_feat["xtracker_count"]

    feat_cols = [c for c in all_feat.columns if "__" in c and c not in
                 ["market__crowd_implied_ev", "market__crowd_std_dev"]]

    print_subheader("6a. Correlation with Daily Tweet Rate")
    corrs = []
    for col in feat_cols:
        valid_sub = all_feat[[col, "daily_rate"]].dropna()
        if len(valid_sub) > 5:
            r = valid_sub[col].corr(valid_sub["daily_rate"])
            corrs.append((col.split("__", 1)[1], r, len(valid_sub)))

    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, r, n in corrs[:15]:
        print(f"  {name:30s}: r = {r:+.4f} (n={n})")

    print_subheader("6b. Correlation with Crowd Error (positive = crowd overestimates)")
    corrs2 = []
    for col in feat_cols:
        valid_sub = all_feat[[col, "crowd_error"]].dropna()
        if len(valid_sub) > 5:
            r = valid_sub[col].corr(valid_sub["crowd_error"])
            corrs2.append((col.split("__", 1)[1], r, len(valid_sub)))

    corrs2.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, r, n in corrs2[:15]:
        print(f"  {name:30s}: r = {r:+.4f} (n={n})")

    # ========================================================================
    # SECTION 7: PRE-MARKET BUCKET-LEVEL ANALYSIS (from prices.parquet)
    # ========================================================================
    print_header("SECTION 7: PRE-MARKET BUCKET-LEVEL ANALYSIS")

    # The metadata bucket prices are SETTLEMENT prices (1.0 for winner).
    # We need the EARLY prices from prices.parquet to assess crowd skill pre-resolution.
    print_subheader("7a. Early Crowd Prices on Winning Bucket (from parquet)")

    idx_raw = json.loads((BACKTEST_DIR / "backtest_index.json").read_text())
    gold_raw = [e for e in idx_raw["events"] if e["ground_truth_tier"] == "gold"]

    pre_market_results = []
    brier_results = []

    for e in gold_raw:
        out_dir = Path(e["output_dir"])
        prices_path = out_dir / "prices.parquet"
        meta_path = out_dir / "metadata.json"
        if not prices_path.exists() or not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        winning_bucket = meta.get("winning_bucket")
        if not winning_bucket:
            continue

        try:
            pdf = pd.read_parquet(prices_path)
        except Exception:
            continue

        if len(pdf) == 0:
            continue

        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        start_time = pdf["timestamp"].min()

        # Get early prices (first 24h of trading)
        early_window = start_time + pd.Timedelta(hours=24)
        early = pdf[pdf["timestamp"] <= early_window]
        if len(early) == 0:
            continue

        # Last price in first 24h for each bucket
        early_prices = early.sort_values("timestamp").groupby("bucket_label")["price"].last()

        # Also get "day before close" prices (48h before end of data)
        end_time = pdf["timestamp"].max()
        late_window = end_time - pd.Timedelta(hours=48)
        late = pdf[pdf["timestamp"] >= late_window]
        late_prices = late.sort_values("timestamp").groupby("bucket_label")["price"].last() if len(late) > 0 else pd.Series()

        # Find the winning bucket's early price
        win_early_price = early_prices.get(winning_bucket, 0)

        # Rank of winning bucket in early prices
        early_sorted = early_prices.sort_values(ascending=False)
        rank = list(early_sorted.index).index(winning_bucket) + 1 if winning_bucket in early_sorted.index else None

        # Compute Brier score of early crowd distribution
        all_buckets = list(meta.get("buckets", []))
        if all_buckets:
            brier = 0
            for b in all_buckets:
                lbl = b["bucket_label"]
                prob = early_prices.get(lbl, 0)
                is_winner = 1.0 if lbl == winning_bucket else 0.0
                brier += (prob - is_winner) ** 2
            brier /= len(all_buckets)

            # Also uniform Brier
            n_bk = len(all_buckets)
            uniform_prob = 1.0 / n_bk
            uniform_brier = ((1 - uniform_prob)**2 + (n_bk - 1) * uniform_prob**2) / n_bk

            brier_results.append({
                "slug": e["event_slug"],
                "market_type": e["market_type"],
                "brier_early_crowd": brier,
                "brier_uniform": uniform_brier,
                "n_buckets": n_bk,
            })

        pre_market_results.append({
            "slug": e["event_slug"],
            "market_type": e["market_type"],
            "actual_count": e["xtracker_count"],
            "winning_bucket": winning_bucket,
            "win_early_price": win_early_price,
            "win_late_price": late_prices.get(winning_bucket, None),
            "rank_early": rank,
            "n_buckets": len(early_prices),
            "top_bucket_early": early_sorted.index[0] if len(early_sorted) > 0 else None,
            "top_price_early": early_sorted.iloc[0] if len(early_sorted) > 0 else None,
        })

    if pre_market_results:
        pmdf = pd.DataFrame(pre_market_results)
        print(f"  Events analyzed: {len(pmdf)}")
        print(f"  Mean early crowd price on winning bucket: {pmdf['win_early_price'].mean():.4f}")
        print(f"  Median early crowd price on winning bucket: {pmdf['win_early_price'].median():.4f}")
        print(f"  Mean n_buckets: {pmdf['n_buckets'].mean():.1f}")
        uniform = 1.0 / pmdf["n_buckets"].mean()
        print(f"  Uniform baseline (1/n): {uniform:.4f}")
        print(f"  Early crowd lift over uniform: {pmdf['win_early_price'].mean() / uniform:.2f}x")

        print(f"\n  Early rank of winning bucket:")
        valid_rank = pmdf.dropna(subset=["rank_early"])
        print(f"    Mean rank: {valid_rank['rank_early'].mean():.1f}")
        print(f"    Median rank: {valid_rank['rank_early'].median():.1f}")
        print(f"    Rank 1 (crowd's top pick wins): {(valid_rank['rank_early'] == 1).sum()}/{len(valid_rank)} "
              f"({(valid_rank['rank_early'] == 1).mean()*100:.0f}%)")
        print(f"    Top 3: {(valid_rank['rank_early'] <= 3).sum()}/{len(valid_rank)} "
              f"({(valid_rank['rank_early'] <= 3).mean()*100:.0f}%)")
        print(f"    Top 5: {(valid_rank['rank_early'] <= 5).sum()}/{len(valid_rank)} "
              f"({(valid_rank['rank_early'] <= 5).mean()*100:.0f}%)")

        # By market type
        print("\n  By market type:")
        for mtype in pmdf["market_type"].unique():
            sub = pmdf[pmdf["market_type"] == mtype]
            print(f"    {mtype:8s}: n={len(sub)}, mean_win_price={sub['win_early_price'].mean():.4f}, "
                  f"mean_rank={sub['rank_early'].mean():.1f}, n_buckets={sub['n_buckets'].mean():.0f}")

        # How often does crowd's top pick actually win?
        print_subheader("7b. How Often Does Crowd's Top Pick Win?")
        top1_wins = (pmdf["rank_early"] == 1).sum()
        print(f"  Crowd top-1 accuracy: {top1_wins}/{len(pmdf)} ({top1_wins/len(pmdf)*100:.0f}%)")

        # When it misses, how far off is it?
        misses = pmdf[pmdf["rank_early"] > 1]
        if len(misses) > 0:
            print(f"  When crowd misses (rank>1), mean rank of winner: {misses['rank_early'].mean():.1f}")
            print(f"  Miss details:")
            for _, row in misses.iterrows():
                print(f"    {row['slug'][:50]:50s}: winner={row['winning_bucket']:12s}, "
                      f"rank={int(row['rank_early'])}, top_pick={row['top_bucket_early']}, "
                      f"win_price={row['win_early_price']:.3f}")

    # Brier score analysis
    if brier_results:
        print_subheader("7c. Brier Score: Early Crowd Distribution Quality")
        bdf = pd.DataFrame(brier_results)
        print(f"  Events: {len(bdf)}")
        print(f"  Mean Brier (early crowd): {bdf['brier_early_crowd'].mean():.4f}")
        print(f"  Mean Brier (uniform):     {bdf['brier_uniform'].mean():.4f}")
        print(f"  Crowd improvement over uniform: {(1 - bdf['brier_early_crowd'].mean()/bdf['brier_uniform'].mean())*100:.1f}%")
        print(f"\n  By market type:")
        for mtype in bdf["market_type"].unique():
            sub = bdf[bdf["market_type"] == mtype]
            print(f"    {mtype:8s}: n={len(sub)}, crowd_brier={sub['brier_early_crowd'].mean():.4f}, "
                  f"uniform_brier={sub['brier_uniform'].mean():.4f}")

    # ========================================================================
    # SECTION 7D: CROWD OVER/UNDER-PRICING IN TAILS
    # ========================================================================
    print_subheader("7d. Tail Analysis: Does Crowd Underprice Tails?")

    tail_results = []
    for e in gold_raw:
        out_dir = Path(e["output_dir"])
        prices_path = out_dir / "prices.parquet"
        meta_path = out_dir / "metadata.json"
        if not prices_path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        actual_count = e["xtracker_count"]
        winning_bucket = meta.get("winning_bucket")
        buckets = meta.get("buckets", [])
        if not buckets or not winning_bucket:
            continue

        try:
            pdf = pd.read_parquet(prices_path)
        except Exception:
            continue
        if len(pdf) == 0:
            continue

        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        start_time = pdf["timestamp"].min()
        early_window = start_time + pd.Timedelta(hours=24)
        early = pdf[pdf["timestamp"] <= early_window]
        if len(early) == 0:
            continue
        early_prices = early.sort_values("timestamp").groupby("bucket_label")["price"].last()

        # Compute crowd_ev from early prices
        crowd_ev_early = 0
        total_prob = 0
        for b in buckets:
            lbl = b["bucket_label"]
            p = early_prices.get(lbl, 0)
            mid = (b["lower_bound"] + min(b["upper_bound"], b["lower_bound"] + 100)) / 2
            crowd_ev_early += p * mid
            total_prob += p

        # Is the actual in the tail (top/bottom 20% of bucket range)?
        lower_bounds = sorted(set(b["lower_bound"] for b in buckets))
        upper_bounds = sorted(set(min(b["upper_bound"], 99999) for b in buckets))
        # Find percentile of actual within bucket range
        all_midpoints = [(b["lower_bound"] + min(b["upper_bound"], b["lower_bound"] + 100)) / 2 for b in buckets]
        if len(all_midpoints) > 2:
            max_mid = max(all_midpoints)
            min_mid = min(all_midpoints)
            if max_mid > min_mid:
                pctile = (actual_count - min_mid) / (max_mid - min_mid)
                is_tail = pctile < 0.2 or pctile > 0.8
                tail_label = "low_tail" if pctile < 0.2 else ("high_tail" if pctile > 0.8 else "center")

                win_price = early_prices.get(winning_bucket, 0)
                tail_results.append({
                    "slug": e["event_slug"],
                    "tail_label": tail_label,
                    "win_early_price": win_price,
                    "actual_pctile": pctile,
                    "n_buckets": len(buckets),
                })

    if tail_results:
        tdf = pd.DataFrame(tail_results)
        for region in ["low_tail", "center", "high_tail"]:
            sub = tdf[tdf["tail_label"] == region]
            if len(sub) > 0:
                print(f"  {region:10s}: n={len(sub)}, mean_win_price={sub['win_early_price'].mean():.4f}")
        print("  (If tail prices are lower, crowd underprices tail outcomes)")

    # ========================================================================
    # SECTION 8: TIME-SERIES OF WEEKLY COUNTS
    # ========================================================================
    print_header("SECTION 8: WEEKLY COUNT TIME SERIES")

    weekly = df[df["market_type"] == "weekly"].sort_values("start_date").copy()
    if len(weekly) > 3:
        weekly["daily_rate"] = weekly["xtracker_count"] / weekly["duration_days"]

        print_subheader("8a. Weekly Count Timeline")
        for _, row in weekly.iterrows():
            crowd_str = ""
            if pd.notna(row.get("market__crowd_implied_ev")):
                err = row["market__crowd_implied_ev"] - row["xtracker_count"]
                crowd_str = f"  crowd_ev={row['market__crowd_implied_ev']:6.0f}  err={err:+6.0f}"
            print(f"  {row['start_date'].date()} - {row['end_date'].date()} | "
                  f"count={row['xtracker_count']:6.0f} | "
                  f"daily_rate={row['daily_rate']:5.1f} | "
                  f"dur={row['duration_days']:2.0f}d{crowd_str}")

        # 8b. Volatility clustering
        print_subheader("8b. Volatility Clustering")
        rates = weekly["daily_rate"].values
        abs_changes = np.abs(np.diff(rates))
        if len(abs_changes) > 3:
            from scipy.stats import pearsonr
            r, p = pearsonr(abs_changes[:-1], abs_changes[1:])
            print(f"  Lag-1 autocorrelation of |change in daily rate|: r={r:.4f}, p={p:.4f}")
            print(f"  {'Volatility clustering detected!' if r > 0.2 and p < 0.1 else 'No strong volatility clustering.'}")

        # 8c. Day of week effect
        print_subheader("8c. Day of Week at Start")
        valid_dow = weekly.dropna(subset=["temporal__day_of_week"])
        if len(valid_dow) > 5:
            dow_rates = valid_dow.groupby("temporal__day_of_week")["daily_rate"].agg(["mean", "count"])
            print("  Day of week -> mean daily rate:")
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            for dow, row in dow_rates.iterrows():
                dow_int = int(dow)
                name = dow_names[dow_int] if dow_int < 7 else f"Day{dow_int}"
                print(f"    {name}: {row['mean']:.1f} tweets/day (n={int(row['count'])})")

    # ========================================================================
    # SECTION 9: DIRECTIONAL CROWD ERROR PATTERNS
    # ========================================================================
    print_header("SECTION 9: DIRECTIONAL CROWD ERROR PATTERNS")

    # Does the crowd consistently over- or under-estimate during specific conditions?
    print_subheader("9a. Crowd Error Direction by Trailing Daily Rate")

    valid_all = df.dropna(subset=["market__crowd_implied_ev", "temporal__rolling_avg_7d"]).copy()
    if len(valid_all) > 5:
        valid_all["crowd_error"] = valid_all["market__crowd_implied_ev"] - valid_all["xtracker_count"]
        valid_all["daily_rate"] = valid_all["xtracker_count"] / valid_all["duration_days"]
        valid_all["trailing_daily"] = valid_all["temporal__rolling_avg_7d"]

        # Split by trailing daily rate
        median_trail = valid_all["trailing_daily"].median()
        high_trail = valid_all[valid_all["trailing_daily"] > median_trail]
        low_trail = valid_all[valid_all["trailing_daily"] <= median_trail]

        print(f"  Median trailing 7d avg: {median_trail:.1f}")
        print(f"  HIGH trailing (>{median_trail:.0f}/day) events: n={len(high_trail)}")
        if len(high_trail) > 0:
            print(f"    Mean crowd error: {high_trail['crowd_error'].mean():+.1f} "
                  f"({high_trail['crowd_error'].mean()/high_trail['xtracker_count'].mean()*100:+.1f}%)")
            print(f"    Crowd over-estimates: {(high_trail['crowd_error'] > 0).sum()}/{len(high_trail)}")
        print(f"  LOW trailing (<={median_trail:.0f}/day) events: n={len(low_trail)}")
        if len(low_trail) > 0:
            print(f"    Mean crowd error: {low_trail['crowd_error'].mean():+.1f} "
                  f"({low_trail['crowd_error'].mean()/low_trail['xtracker_count'].mean()*100:+.1f}%)")
            print(f"    Crowd over-estimates: {(low_trail['crowd_error'] > 0).sum()}/{len(low_trail)}")

    # 9b. Largest crowd misses (worst 5 events)
    print_subheader("9b. Largest Crowd Misses (Top 5 by absolute error)")
    if len(valid_all) > 3:
        valid_all["abs_error"] = valid_all["crowd_error"].abs()
        worst = valid_all.nlargest(5, "abs_error")
        for _, row in worst.iterrows():
            print(f"  {row['start_date'].date()} {row['market_type']:8s}: "
                  f"crowd_ev={row['market__crowd_implied_ev']:.0f}, actual={row['xtracker_count']:.0f}, "
                  f"error={row['crowd_error']:+.0f} ({row['crowd_error']/row['xtracker_count']*100:+.1f}%), "
                  f"trail_7d={row['temporal__rolling_avg_7d']:.1f}/day")

    # 9c. Week-over-week crowd adjustment speed
    print_subheader("9c. Does the Crowd Adjust Fast Enough Week-to-Week?")
    weekly_valid = df[(df["market_type"] == "weekly") & df["has_market"]].sort_values("start_date").copy()
    weekly_valid = weekly_valid.dropna(subset=["market__crowd_implied_ev"])
    if len(weekly_valid) > 3:
        weekly_valid["daily_rate"] = weekly_valid["xtracker_count"] / weekly_valid["duration_days"]
        weekly_valid["crowd_daily_rate"] = weekly_valid["market__crowd_implied_ev"] / weekly_valid["duration_days"]

        # Lag actual rate and see how much crowd adjusts
        weekly_valid["prev_actual_rate"] = weekly_valid["daily_rate"].shift(1)
        weekly_valid["rate_change"] = weekly_valid["daily_rate"] - weekly_valid["prev_actual_rate"]
        weekly_valid["crowd_change"] = weekly_valid["crowd_daily_rate"] - weekly_valid["crowd_daily_rate"].shift(1)

        valid_changes = weekly_valid.dropna(subset=["rate_change", "crowd_change"])
        if len(valid_changes) > 3:
            # When actual rate increased, did crowd increase enough?
            up = valid_changes[valid_changes["rate_change"] > 5]
            down = valid_changes[valid_changes["rate_change"] < -5]
            print(f"  Weeks with rate INCREASE (>5/day): n={len(up)}")
            if len(up) > 0:
                print(f"    Mean actual increase: {up['rate_change'].mean():+.1f}/day")
                print(f"    Mean crowd increase:  {up['crowd_change'].mean():+.1f}/day")
                print(f"    Crowd captures {up['crowd_change'].mean()/up['rate_change'].mean()*100:.0f}% of upswing")
            print(f"  Weeks with rate DECREASE (<-5/day): n={len(down)}")
            if len(down) > 0:
                print(f"    Mean actual decrease: {down['rate_change'].mean():+.1f}/day")
                print(f"    Mean crowd decrease:  {down['crowd_change'].mean():+.1f}/day")
                print(f"    Crowd captures {down['crowd_change'].mean()/down['rate_change'].mean()*100:.0f}% of downswing")

    # ========================================================================
    # SECTION 10: ACTIONABLE EDGE SUMMARY
    # ========================================================================
    print_header("SECTION 10: ACTIONABLE EDGE OPPORTUNITIES SUMMARY")

    # Final: compute how often a simple "rolling_avg * duration" model
    # would beat the crowd EV
    print_subheader("10a. Head-to-Head: Simple Rolling Avg vs Crowd EV")

    head2head = df.dropna(subset=["temporal__rolling_avg_7d", "market__crowd_implied_ev"]).copy()
    if len(head2head) > 3:
        for avg_col, label in [("temporal__rolling_avg_7d", "7d"),
                                ("temporal__rolling_avg_14d", "14d")]:
            sub = head2head.dropna(subset=[avg_col])
            if len(sub) < 3:
                continue
            model_pred = sub[avg_col] * sub["duration_days"]
            crowd_pred = sub["market__crowd_implied_ev"]
            actual = sub["xtracker_count"]

            model_err = (model_pred - actual).abs()
            crowd_err = (crowd_pred - actual).abs()

            model_wins = (model_err < crowd_err).sum()
            crowd_wins = (model_err > crowd_err).sum()
            ties = (model_err == crowd_err).sum()

            print(f"  {label}_avg * duration vs crowd: model_wins={model_wins}, "
                  f"crowd_wins={crowd_wins}, ties={ties} "
                  f"({model_wins/(model_wins+crowd_wins)*100:.0f}% model win rate)")

    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
