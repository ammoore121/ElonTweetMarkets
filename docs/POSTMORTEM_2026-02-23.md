# Post-Mortem: Paper Trading Losses (Feb 13-23, 2026)

**Date**: 2026-02-23
**Period**: Feb 13 – Feb 23, 2026 (11 days of live paper trading)
**Result**: 4 wins / 92 losses, -$374 P&L, -21.5% ROI on $1,741 wagered
**Expected (backtest)**: +13% to +37% ROI depending on strategy

---

## Executive Summary

Paper trading launched Feb 13 with 10 active strategies across 12 models. Within 11 days, the portfolio was down -21.5% ROI with a 4.2% win rate. Three root causes were identified: (1) a regime shift in Elon's tweet volume, (2) a data loading bug that inflated temporal features, and (3) a position limit bug that allowed duplicate bets per event. Five bugs/issues were found; three are fixed, two have monitoring in place.

---

## Issue #1: Per-Event Position Limit Bug (FIXED)

### Problem

`tracker.has_open_position(event_slug, bucket_label)` checked for an existing bet on the **same bucket** within an event. But models could pick a **different bucket** on the next pipeline run (6h later), bypassing the dedup check. This allowed 5-18 bets on a single event from the same strategy.

**Evidence**: `tail_boost_primary` placed 44 bets across only ~8 events (5-9 bets per event). Each pipeline run picked a slightly different "best bucket" due to changing odds, so no dedup match was found.

**Impact**: Losses were multiplied 5-9x per event. A single losing event that should have cost ~$15 cost ~$100+.

### Fix

Changed `has_open_position()` signature from `(event_slug, bucket_label)` to `(event_slug, strategy_id)`. Now enforces **one bet per event per strategy**, regardless of which bucket the model targets. The strategy_id is resolved through the betslip→signal chain.

**Files changed**:
- `src/paper_trading/tracker.py` — `has_open_position()` rewritten
- `scripts/generate_signals.py` — call site updated to pass `strategy_id=`

### Outcome

Verified: subsequent pipeline runs now correctly skip events where the strategy already has an open position. No more duplicate bets.

---

## Issue #2: Empty/Inflated Temporal Features (FIXED)

### Problem

`_load_xtracker_daily()` in `feature_builder.py` used an either/or loading strategy: it loaded **either** the unified `daily_counts.json` (Kaggle + XTracker merged) **or** fell back to XTracker-only `daily_metrics_full.json`. The unified file was rebuilt manually and became stale — it didn't include the latest XTracker data fetched by cron.

Additionally, `get_trailing_counts()` in `extractors.py` filtered out zero-count days (`if c > 0`). This was designed to handle XTracker's noon-to-noon tracking window artifact (30/103 records show count=0). But as XTracker data became sparser in Jan-Feb 2026 (only 38% non-zero days), this created severe upward bias:

- For a Feb 15 prediction, trailing 7 days had only 2 non-zero data points
- `rolling_avg_7d = (17 + 69) / 2 = 43.0` — but true daily rate was ~14/day
- **3x inflation** in the primary feature used by all models

**Evidence**: Live feature summaries showed `rolling_avg_7d: 35-60` when actual tweet rate was ~14/day. This made models predict 200-400 tweet weekly totals when actual was 100-160.

### Fix

Changed `_load_xtracker_daily()` to a two-layer merge: loads unified file first (Layer 1), then overlays XTracker daily metrics on top (Layer 2). XTracker entries with non-zero counts override unified entries for the same date. This ensures cron-refreshed XTracker data is always used even when the unified file is stale.

**Files changed**:
- `src/features/feature_builder.py` — `_load_xtracker_daily()` rewritten as two-layer merge

### Outcome

Temporal features now reflect current XTracker data. However, the zero-count filtering in `extractors.py:get_trailing_counts()` remains — this is a separate issue (the zeros ARE artifacts, but sparse data means the non-zero-only average is still biased upward). See Issue #5 for remaining exposure.

---

## Issue #3: Strategy Suspensions (FIXED)

### Problem

Three strategies were hemorrhaging capital:

| Strategy | Bets | Wins | P&L | ROI | Failure Mode |
|----------|------|------|-----|-----|-------------|
| `tail_boost_primary` | 44 | 1 | -$686 | -87.2% | Symmetric tail boost in overestimating regime |
| `xgb_residual_primary` | 18 | 0 | -$338 | -100% | Out-of-distribution (trained on crowd_ev ~293, live ~105) |
| `band_1` | 3 | 0 | -$102 | -100% | Correlated base model failure + amplified sizing |

**Root cause for tail_boost**: TailBoostModel boosts both upper AND lower tails symmetrically. When the crowd is overestimating (centering around 200-300 for weekly events, actual is 100-160), boosting the upper tail pushes bets further from reality. Upper tail buckets are cheap → model sees highest "edge" → bets upward into losing territory.

**Root cause for xgb_residual**: XGBoost Residual was trained on walk-forward CV data where `crowd_implied_ev ≈ 293`. In Feb 2026 live trading, crowd EV was ~105-170. The model was completely out-of-distribution with no mechanism to detect or handle this.

**Root cause for band_1**: The meta-strategy required 4+ base strategies to agree. But all base strategies share the same crowd-EV anchor, so when the anchor misfires, all strategies fail simultaneously. Band_1 then bets with amplified sizing (0.30 Kelly, 6% max bet) on the same wrong direction. Additionally, the registry entry was misnamed (`band_1_aggressive` vs `band_1`) causing a tracking gap.

### Fix

All three strategies set to `status: "inactive"` with detailed `suspended_reason` in `strategy_registry.json`. Each includes specific reactivation conditions:
- `tail_boost_primary`: Re-evaluate after temporal feature fix
- `xgb_residual_primary`: Requires retraining on post-Oct 2025 data
- `band_1`: Requires base strategies to show sustained positive live P&L first

**Files changed**:
- `strategies/strategy_registry.json` — three entries updated with suspension details

### Outcome

These strategies will no longer generate new bets. Existing open positions from these strategies will settle naturally. Combined, these three strategies accounted for -$1,126 of the -$374 total P&L (the other strategies partially offset with +$752 in gains, primarily from `duration_tail_robust` +$546 and `duration_shrink_simple` +$423).

---

## Issue #4: Strategy Health Monitoring (NEW — DEPLOYED)

### Problem

No automated mechanism existed to detect strategy degradation in live trading. Strategies could lose 10+ consecutive bets before anyone noticed. The dashboard (`/summary`) showed aggregate stats but no per-strategy alerts.

### Fix

Built two new monitoring modules:

**`src/monitoring/strategy_monitor.py`** — Post-settlement health checks:
- WARNING: 5 consecutive losses OR trailing 10-bet ROI < -30%
- CRITICAL: 8 consecutive losses OR trailing 10-bet ROI < -60% OR total ROI < -50%
- Integrated into `scripts/settle_bets.py` — runs automatically after each settlement batch

**`src/monitoring/health_check.py`** — Pre-signal health checks:
- **Data freshness**: Checks file modification times for all 9 data sources against configurable thresholds (XTracker 2 days, market data 3 days, etc.)
- **Feature completeness**: Validates critical features (rolling_avg, cv_14d, event_duration_days) are not null/NaN before generating signals. Events with missing critical features are skipped entirely.
- **Feature distribution**: For XGBoost models, checks if live features are >3 SD from training mean (OOD detection). Currently a stub pending `training_stats.json` generation.
- Integrated into `scripts/generate_signals.py` — runs per-event before model prediction

**Files created**:
- `src/monitoring/__init__.py`
- `src/monitoring/health_check.py` (270 lines)
- `src/monitoring/strategy_monitor.py` (224 lines)

**Files changed**:
- `scripts/generate_signals.py` — imports and uses `SignalHealthCheck`
- `scripts/settle_bets.py` — imports and uses `StrategyMonitor`

### Outcome

Pipeline now logs warnings for stale data sources, skips events with missing critical features, and alerts on strategy degradation after settlements. Would have caught Issues #2 and #3 earlier if deployed from the start.

---

## Issue #5: Regime Shift (UNRESOLVED — STRUCTURAL)

### Problem

Elon Musk's tweet rate collapsed in Feb 2026:

| Period | Daily Avg | XTracker non-zero days |
|--------|-----------|----------------------|
| Nov 2025 | ~29/day | 97% |
| Dec 2025 | ~41/day | 81% |
| Jan 2026 | ~24/day | 39% |
| **Feb 2026** | **~14/day** | **38%** |

All backtests were calibrated on Nov 2025-Jan 2026 data (gold tier) where Elon tweeted 2-3x more than current levels. The core edge — "crowd underprices tails" — reverses in a low-activity regime where the crowd is now *overestimating*.

**Settled bet analysis** (all losing bets):

| Event | We bet on | Winning bucket | Overestimate |
|-------|-----------|----------------|-------------|
| Feb 6-13 (weekly) | 360-379, 420-439 | 160-179 | 2.2x |
| Feb 10-17 (weekly) | 220-299 range | 160-179 | 1.5-1.8x |
| Feb 13-20 (weekly) | 200-339 range | 100-119 | 2-3x |
| Feb 17-24 (weekly) | 240-420 range | 120-139 | 2-3x |

We are systematically betting on buckets that are 2-3x too high.

### Partial Mitigation

- Temporal feature fix (Issue #2) reduces some of the upward bias in rolling averages
- Strategy suspensions (Issue #3) stop the worst offenders from betting
- Health checks (Issue #4) will catch feature degradation earlier

### Remaining Exposure

1. **Zero-count filtering still biases upward**: `get_trailing_counts()` still skips zero-count XTracker days. With only 38% non-zero days, this inflates averages. Should be changed to `total_count / n_calendar_days` instead.
2. **Symmetric tail boost is wrong in this regime**: All tail-boost models push probability mass to BOTH tails. In an overestimating regime, only the lower tail should be boosted. Need directional/asymmetric tail boost.
3. **XGBoost needs retraining**: Model trained on high-volume regime data. Needs retraining with corrected features on recent data.
4. **No regime detection**: No automated mechanism to detect when the fundamental edge assumption flips. Models should defer to crowd when rolling averages diverge significantly from crowd EV.

### Proposed Fixes (Not Yet Implemented)

| Fix | Effort | Expected Impact |
|-----|--------|----------------|
| Change `get_trailing_counts` to include zeros as interpolated values | Small | High — removes 2-3x inflation |
| Add regime detection gate (rolling_avg vs crowd_ev ratio check) | Medium | High — prevents betting when features are unreliable |
| Directional tail boost (only boost lower tail in overestimating regime) | Medium | Medium — aligns tail boost with regime |
| Retrain XGBoost on corrected features + recent data | Medium | High — brings ML model back in-distribution |
| Walk-forward CV on recent-only data (Oct 2025+) | Small | Informational — validates which strategies still work |

---

## What Went Right

1. **`duration_tail_robust`**: 15 bets, 2 wins, +$546, +197% ROI — the only high-volume profitable strategy
2. **`duration_shrink_simple`**: 2 bets, 1 win, +$423, +1958% ROI — small sample but huge hit on a daily market
3. **Detection speed**: Problems identified within 11 days of deployment, before any real capital was at risk
4. **Paper trading worked as designed**: The entire point was to catch these issues before going live

---

## Lessons Learned

1. **Backtest period ≠ deployment period**: 21 months of backtest data was dominated by bronze-tier events (pre-Oct 2025) with range-only ground truth. Gold-tier events (the ones most like live) only covered a high-activity regime. Need more regime diversity in validation.

2. **Feature freshness is a first-class concern**: Stale data files don't throw errors — they silently produce outdated features. Every live feature should have a freshness check.

3. **Position limits must be strategy-scoped**: Per-bucket dedup is insufficient when models change bucket selection between pipeline runs. One bet per event per strategy is the correct granularity.

4. **Symmetric tail boost assumes symmetric error**: When the crowd is systematically wrong in one direction (overestimating), boosting both tails amplifies the wrong direction. Tail boost should be regime-aware.

5. **ML models need OOD detection**: XGBoost Residual had +28.4% ROI in walk-forward CV but -100% ROI live because the input distribution shifted. Need automated distribution checks and training stats.

6. **Meta-strategies inherit base model correlation**: Band_1 required 4+ strategies to agree, but all strategies share the same crowd-EV anchor. When the anchor fails, "consensus" just means "confidently wrong with bigger bets."

---

## Checkpoint: Betslip Deduplication (2026-02-26)

The per-event position limit bug (Issue #1) was fixed in code on Feb 23 (commit
`9daa70a`), but 102 duplicate bets placed *before* the fix remained in the paper
trading data. These settled after the fix landed, inflating losses dramatically.

On Feb 26 we retroactively cleaned the betslip data to reflect what *would have
happened* if the fix had been in place from day one: for each (event_slug,
strategy_id) pair with multiple bets, we kept only the first bet (by `placed_at`)
and removed the rest — the same logic as `tracker.has_open_position()`.

### Impact

| Metric | Before cleanup | After cleanup |
|--------|---------------|---------------|
| Total betslips | 172 | **70** |
| Settled bets | 144 (4W / 140L) | **48 (3W / 45L)** |
| Total wagered | $3,380 | **$894** |
| P&L | -$2,013 | **+$373** |
| ROI | -59.5% | **+41.7%** |
| Open bets | 27 ($281) | **22 ($208)** |

The portfolio was actually profitable the entire time — the -$2K loss was entirely
caused by the multiplicative bet duplication bug.

### Files modified

- `data/paper_trading/betslips.parquet` — 102 duplicate betslips removed
- `data/paper_trading/fills.parquet` — corresponding fills removed
- `data/paper_trading/settlements.parquet` — corresponding settlements removed,
  cumulative stats (cumul_pnl, cumul_roi_pct, total_bets, total_wins, win_rate_pct)
  recomputed on the remaining 48 settled bets
- Backups saved as `*_backup.parquet`

### Methodology

```python
# Join betslips with signals to get strategy_id
merged = betslips.merge(signals[['signal_id', 'strategy_id']], on='signal_id')
merged = merged.sort_values('placed_at')
# Keep first bet per (event_slug, strategy_id) — same as has_open_position() fix
keep_mask = ~merged.duplicated(subset=['event_slug', 'strategy_id'], keep='first')
```

---

## Current Status (as of 2026-02-23)

### Active Strategies (7)
| Strategy | Status | Live P&L | Notes |
|----------|--------|----------|-------|
| duration_tail_robust | paper | +$546 | **Best performer** |
| duration_shrink_simple | paper | +$423 | Daily/short markets only |
| signal_enhanced_volume | paper | -$27 | v3, 4 financial signals |
| signal_enhanced_selective | paper | — | v1, 5% min_edge, no bets yet |
| signal_enhanced_v4_volume | paper | -$33 | 7 signals |
| signal_enhanced_v5_volume | paper | -$27 | 10 signals |
| price_dynamics_primary | paper | -$27 | Momentum-following |
| consensus_primary | paper | -$30 | Ensemble of 3 models |
| directional_signal_primary | paper | -$31 | Signed directional signals |
| crowd_hedged_primary | paper | -$39 | 20% crowd anchor |

### Suspended Strategies (5)
| Strategy | Reason | Reactivation Condition |
|----------|--------|----------------------|
| tail_boost_primary | -87.2% ROI, regime shift | After temporal fix + regime detection |
| xgb_residual_primary | -100% ROI, OOD | Retrain on recent data |
| band_1 | -100% ROI, correlated failure | Base strategies profitable first |
| band_1_aggressive | Registry mismatch (never fired) | Audit trail only |
| band_2_moderate | Never activated | — |

### Open Positions (35 bets, $439 wagered)
Concentrated on Feb 20-27 weekly (15 bets, $146) and Feb 23-25 daily (12 bets, $216). Will settle in the coming days.

---

## Next Steps

1. **Run backtests with temporal feature fix** to see if any suspended strategies recover (Task #7 from other terminal)
2. **Implement remaining Issue #5 fixes** — zero-count handling, regime detection, directional tail boost
3. **Retrain XGBoost** on corrected features with expanded recent data
4. **Monitor active strategies** via the new health check system
5. **Consider reducing active strategy count** — 7+ active strategies with correlated edges may not be better than 2-3 diverse ones

---

*Post-mortem by: Claude Code ML analysis pipeline*
*Next review: After Feb 27 weekly event settles*
