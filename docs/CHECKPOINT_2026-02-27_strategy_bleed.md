# Checkpoint: Strategy P&L Bleed Investigation

**Date**: 2026-02-27
**Trigger**: Cumulative P&L dropped from peak +$860 (Feb 23) to +$200 (Feb 27), a -$660 drawdown in 4 days.
**Scope**: All 14 strategies with settled paper trading bets (77 total bets, 64 settled).

---

## Executive Summary

Only 2 of 14 strategies are profitable. The other 12 are correlated duplicates sharing the same crowd-EV anchor, which has regime-shifted. All models systematically overestimate tweet counts by +121 on average. Weekly markets are a complete loss (0W/35L). Duration models win only because they shrink the EV for short events, accidentally pushing mass toward the correct low-count buckets.

**Actions taken**: Restricted `duration_tail_robust` to daily/short markets. Retired 8 strategies. Marked 3 for iteration after recalibration. Updated strategy_registry.json.

---

## Finding 1: Systematic Upward Bias (+121 tweets)

Every model overestimates Elon's tweet counts. Average bet targets bucket midpoint ~203; actual average is ~82.

| Event | Actual | Avg Bet | Bias |
|-------|--------|---------|------|
| Feb 6-13 (weekly) | 160 | 400 | +240 |
| Feb 17-24 (weekly) | 124 | 341 | +217 |
| Feb 20-27 (weekly) | 107 | 291 | +184 |
| Feb 13-20 (weekly) | 109 | 290 | +180 |
| Feb 10-17 (weekly) | 170 | 280 | +110 |
| Feb 12-14 (short) | 17 | 64 | +48 |
| Feb 14-16 (daily) | 29 | 74 | +44 |
| Feb 23-25 (daily) | 32 | 62 | +30 |
| Feb 16-18 (daily) | 32 | 56 | +24 |
| Feb 21-23 (daily) | 62 | 71 | +9 |

**Root cause**: Models anchor to crowd EV which reflects historical tweet rates (~293/week). Current regime is ~107-170/week. The crowd hasn't fully recalibrated.

## Finding 2: Weekly Markets Are a Deathtrap

| Market Type | Bets | Wins | P&L |
|-------------|------|------|-----|
| Daily/Short | 29 | 3 | **+$683** |
| Weekly | 35 | 0 | **-$483** |

Zero wins on 35 weekly bets. The bias is worst on weekly markets (avg +186 tweets) because weekly crowd EVs have the most stale anchoring.

## Finding 3: All Losing Strategies Bet the Same Buckets

On Feb 23-25 daily (actual: 32, winning bucket: <40):
- **10 strategies** all bet `40-64` — off by 1 bucket
- `xgb_residual` and `regime_tail` bet `65-89` — off by 2 buckets

On Feb 20-27 weekly (actual: 107, winning bucket: 100-119):
- **11 bets** on `280-299` from 7 different strategies
- All strategies share the same crowd-EV-anchored tail redistribution

These aren't independent strategies. They're variations of the same model producing the same outputs.

## Finding 4: 26% of Losses Were Adjacent-Bucket Near-Misses

16 of 61 losses landed in the bucket immediately adjacent to the winning bucket. The models' distributional shape isn't wildly wrong — they're just centered too high.

## Finding 5: Why Duration Models Win

All 3 wins were extreme low-tail bets on daily/short markets:

| Event | Bucket | Price | Payout | Strategy |
|-------|--------|-------|--------|----------|
| Feb 14-16 | <40 | 0.074 | 12.5x | duration_tail_robust |
| Feb 16-18 | <40 | 0.040 | 24.0x | duration_tail_robust |
| Feb 21-23 | 40-64 | 0.033 | 29.3x | duration_shrink_simple |

Duration models work by **shrinking crowd EV for short-duration markets**, which pushes probability mass toward low buckets. In the current low-tweet regime, this accidentally targets the right region. On daily/short: **9 bets, 3W, +$1,101, +664% ROI**. On weekly: **9 bets, 0W, -$98**.

## Finding 6: XGBoost Residual Has Distribution Shift

Trained on historical data where crowd_ev ~293. Live crowd_ev ~105. The learned residual corrections are meaningless. All 8 bets targeted wrong ranges (65-114 and 260-319). The residual framing is sound (+28.4% walk-forward CV ROI) but needs retraining on current-regime data.

---

## Strategy Verdicts

### KEEP (2 strategies)

| Strategy | Live Record | P&L | Action |
|----------|------------|-----|--------|
| `duration_tail_robust` | 2W/13L | +$596 | **Restrict to daily/short** (removed weekly) |
| `duration_shrink_simple` | 1W/2L | +$406 | Keep as-is (already daily/short only) |

### RETIRE (8 strategies) — No salvageable signal

| Strategy | Live Record | P&L | Reason |
|----------|------------|-----|--------|
| `tail_boost_primary` | 0W/11L | -$125 | Strictly dominated by duration_tail_robust |
| `consensus_primary` | 0W/3L | -$37 | Ensemble of 3 broken models |
| `crowd_hedged_primary` | 0W/3L | -$48 | Subset of broken consensus, amplified sizing |
| `signal_enhanced_volume` (v3) | 0W/3L | -$40 | Correlated duplicate, same buckets as all others |
| `signal_enhanced_v4_volume` | 0W/3L | -$42 | No differentiation from v3 |
| `signal_enhanced_v5_volume` | 0W/3L | -$37 | No differentiation from v3/v4 |
| `regime_tail_primary` | 0W/3L | -$130 | Failed backtest (-80% ROI) AND live. Highest per-bet loss |
| `band_1` | 0W/1L | -$36 | Meta-strategy over correlated broken models |

### SUSPEND-ITERATE (3 strategies) — Salvageable signal, needs recalibration

| Strategy | Live Record | P&L | Iteration Path |
|----------|------------|-----|----------------|
| `xgb_residual_primary` | 0W/8L | -$176 | Retrain on Oct 2025+ data only. Restrict to daily/short. Residual framing is sound. |
| `price_dynamics_primary` | 0W/3L | -$39 | Recalibrate momentum on Oct 2025+ data. Restrict to daily/short. |
| `directional_signal_primary` | 0W/3L | -$45 | Raise min_edge to 5%. Restrict to daily/short. Unique signed-signal approach. |

### SUSPEND (1 strategy) — Insufficient data

| Strategy | Live Record | P&L | Notes |
|----------|------------|-----|-------|
| `signal_enhanced_selective` | 0W/2L | -$49 | 5% min_edge is correct idea. Too few bets to judge. Re-evaluate after recalibration. |

---

## Open Exposure (Sunk Cost)

10 open bets from suspended strategies (~$75 at risk) on Feb 24-Mar 3 weekly, mostly targeting `260-279`. These will likely settle as losses given the systematic upward bias on weekly markets. Accept as sunk cost.

---

## Lessons Learned

1. **Don't paper trade 12 variations of the same model.** If strategies share an anchor (crowd EV), they're correlated — not independent. One failure = twelve losses.
2. **Weekly markets need different treatment than daily.** The crowd-EV bias is 3x worse on weekly markets. All weekly bets lost.
3. **Duration-aware models are the real edge.** Shrinking EV for short events correctly captures the low-tweet regime. This is the only structural edge that survived live trading.
4. **Near-misses (26%) suggest the distributional shape isn't wrong, just the center.** Recalibrating the EV anchor downward could recover several losing strategies.
5. **XGBoost needs regime-aware retraining.** The residual framing works in principle but the training data is stale.

---

## Next Steps

1. Monitor `duration_tail_robust` (daily/short only) and `duration_shrink_simple` — the only active strategies
2. Recalibrate crowd EV baseline using Oct 2025+ data
3. Retrain XGBoost residual on current-regime data
4. Iterate directional_signal at 5% min_edge on daily/short
5. Do NOT reactivate any retired strategy
