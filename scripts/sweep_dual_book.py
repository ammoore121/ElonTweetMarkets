"""Dual book analysis: hedged + unhedged running simultaneously on GOLD tier.

Questions answered:
1. How much overlap in bets? (same event+bucket)
2. What's the combined portfolio P&L?
3. Does diversification reduce drawdowns?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.crowd_hedged_model import CrowdHedgedModel
from src.ml.consensus_model import ConsensusEnsembleModel
from src.backtesting.engine import BacktestEngine

BACKTEST_DIR = ROOT / "data" / "backtest"
INDEX_FILE = BACKTEST_DIR / "backtest_index.json"


def load_events(tier):
    with open(INDEX_FILE) as f:
        index = json.load(f)
    return [e for e in index["events"] if e.get("ground_truth_tier") == tier]


def run_book(model, events, kelly, min_edge=0.02):
    config = {
        "min_edge": min_edge,
        "kelly_fraction": kelly,
        "max_bet_pct": 0.10,
        "bankroll": 1000.0,
        "entry_hours_before_close": 24,
        "entry_window_hours": 6,
    }
    engine = BacktestEngine(config=config)
    return engine.run(model, events)


def analyze_overlap(r_unhedged, r_hedged):
    """Analyze bet overlap between the two books."""
    # Build sets of (event_slug, bucket_label) for each book
    unhedged_bets = set()
    hedged_bets = set()

    unhedged_by_event = defaultdict(list)
    hedged_by_event = defaultdict(list)

    for t in r_unhedged.trades:
        key = (t.event_slug, t.bucket_label)
        unhedged_bets.add(key)
        unhedged_by_event[t.event_slug].append(t)

    for t in r_hedged.trades:
        key = (t.event_slug, t.bucket_label)
        hedged_bets.add(key)
        hedged_by_event[t.event_slug].append(t)

    overlap = unhedged_bets & hedged_bets
    unhedged_only = unhedged_bets - hedged_bets
    hedged_only = hedged_bets - unhedged_bets

    # Events touched
    unhedged_events = set(unhedged_by_event.keys())
    hedged_events = set(hedged_by_event.keys())
    shared_events = unhedged_events & hedged_events
    any_event = unhedged_events | hedged_events

    return {
        "overlap_bets": len(overlap),
        "unhedged_only_bets": len(unhedged_only),
        "hedged_only_bets": len(hedged_only),
        "total_unique_bets": len(unhedged_bets | hedged_bets),
        "unhedged_events": len(unhedged_events),
        "hedged_events": len(hedged_events),
        "shared_events": len(shared_events),
        "total_events": len(any_event),
        "overlap_keys": overlap,
        "unhedged_by_event": unhedged_by_event,
        "hedged_by_event": hedged_by_event,
    }


def compute_combined_drawdown(r1, r2, w1=0.5, w2=0.5):
    """Compute combined portfolio drawdown from two books with allocation weights.

    w1, w2 = fraction of total bankroll allocated to each book.
    """
    # Build per-event P&L for each book
    pnl1 = defaultdict(float)
    pnl2 = defaultdict(float)
    for s in r1.settlements:
        pnl1[s.event_slug] += s.pnl
    for s in r2.settlements:
        pnl2[s.event_slug] += s.pnl

    # Get all events in chronological order (use event slug as proxy)
    all_events = sorted(set(list(pnl1.keys()) + list(pnl2.keys())))

    cum1 = cum2 = cum_combined = 0.0
    peak1 = peak2 = peak_combined = 0.0
    dd1 = dd2 = dd_combined = 0.0

    event_pnls = []
    for ev in all_events:
        p1 = pnl1.get(ev, 0.0) * w1
        p2 = pnl2.get(ev, 0.0) * w2
        p_comb = p1 + p2

        cum1 += pnl1.get(ev, 0.0)
        cum2 += pnl2.get(ev, 0.0)
        cum_combined += p_comb

        peak1 = max(peak1, cum1)
        peak2 = max(peak2, cum2)
        peak_combined = max(peak_combined, cum_combined)

        dd1 = max(dd1, peak1 - cum1)
        dd2 = max(dd2, peak2 - cum2)
        dd_combined = max(dd_combined, peak_combined - cum_combined)

        event_pnls.append({
            "event": ev, "pnl_book1": p1, "pnl_book2": p2,
            "pnl_combined": p_comb, "cum_combined": cum_combined,
        })

    return {
        "dd_book1": dd1,
        "dd_book2": dd2,
        "dd_combined": dd_combined,
        "event_pnls": event_pnls,
    }


def main():
    gold = load_events("gold")
    print(f"Gold events: {len(gold)}\n")

    # =====================================================================
    # Define book configurations to test
    # =====================================================================
    configs = [
        # (label, CE_kelly, CH_weight, CH_kelly, alloc_CE, alloc_CH)
        ("50/50: CE(0.25) + CH20(0.25)", 0.25, 0.20, 0.25, 0.5, 0.5),
        ("50/50: CE(0.25) + CH20(0.40)", 0.25, 0.20, 0.40, 0.5, 0.5),
        ("60/40: CE(0.25) + CH20(0.35)", 0.25, 0.20, 0.35, 0.6, 0.4),
        ("40/60: CE(0.25) + CH20(0.50)", 0.25, 0.20, 0.50, 0.4, 0.6),
        ("50/50: CE(0.35) + CH20(0.35)", 0.35, 0.20, 0.35, 0.5, 0.5),
    ]

    # Run baselines
    print("=" * 100)
    print("  INDIVIDUAL BOOKS (GOLD, min_edge=2%)")
    print("=" * 100)
    header = f"{'Book':<45} {'Bets':>4}  {'Wagered':>8}  {'P&L':>8}  {'ROI':>7}  {'MaxDD':>6}"
    print(header)
    print("-" * 100)

    r_ce_25 = run_book(ConsensusEnsembleModel(), gold, kelly=0.25)
    r_ce_35 = run_book(ConsensusEnsembleModel(), gold, kelly=0.35)
    r_ch20_25 = run_book(CrowdHedgedModel(crowd_weight=0.20), gold, kelly=0.25)
    r_ch20_35 = run_book(CrowdHedgedModel(crowd_weight=0.20), gold, kelly=0.35)
    r_ch20_40 = run_book(CrowdHedgedModel(crowd_weight=0.20), gold, kelly=0.40)
    r_ch20_50 = run_book(CrowdHedgedModel(crowd_weight=0.20), gold, kelly=0.50)

    for label, r in [
        ("ConsensusEnsemble  kelly=0.25", r_ce_25),
        ("ConsensusEnsemble  kelly=0.35", r_ce_35),
        ("CrowdHedged w=0.20 kelly=0.25", r_ch20_25),
        ("CrowdHedged w=0.20 kelly=0.35", r_ch20_35),
        ("CrowdHedged w=0.20 kelly=0.40", r_ch20_40),
        ("CrowdHedged w=0.20 kelly=0.50", r_ch20_50),
    ]:
        dd = compute_combined_drawdown(r, r, 1.0, 0.0)  # hack: just book1
        print(f"{label:<45} {r.n_bets:>4}  ${r.total_wagered:>7.0f}  ${r.total_pnl:>+7.0f}  {r.roi:>+6.1f}%  ${dd['dd_book1']:>5.0f}")

    # Overlap analysis
    print(f"\n{'='*100}")
    print("  BET OVERLAP ANALYSIS")
    print(f"{'='*100}")

    ov = analyze_overlap(r_ce_25, r_ch20_25)
    print(f"  ConsensusEnsemble bets: {r_ce_25.n_bets}")
    print(f"  CrowdHedged w=0.20 bets: {r_ch20_25.n_bets}")
    print(f"  Overlapping bets (same event+bucket): {ov['overlap_bets']}")
    print(f"  CE-only bets: {ov['unhedged_only_bets']}")
    print(f"  CH-only bets: {ov['hedged_only_bets']}")
    print(f"  Total unique bets: {ov['total_unique_bets']}")
    print(f"  Events with CE bets: {ov['unhedged_events']}")
    print(f"  Events with CH bets: {ov['hedged_events']}")
    print(f"  Events in both: {ov['shared_events']}")
    print(f"  Overlap ratio: {ov['overlap_bets'] / max(ov['total_unique_bets'], 1):.0%}")

    # P&L on overlapping vs non-overlapping
    overlap_pnl_ce = 0.0
    nonoverlap_pnl_ce = 0.0
    for s in r_ce_25.settlements:
        key = (s.event_slug, s.bucket_label)
        if key in ov["overlap_keys"]:
            overlap_pnl_ce += s.pnl
        else:
            nonoverlap_pnl_ce += s.pnl

    print(f"\n  CE P&L from overlapping bets: ${overlap_pnl_ce:+.0f}")
    print(f"  CE P&L from CE-only bets: ${nonoverlap_pnl_ce:+.0f}")

    # =====================================================================
    # COMBINED PORTFOLIO
    # =====================================================================
    print(f"\n{'='*100}")
    print("  COMBINED DUAL-BOOK PORTFOLIOS (GOLD, min_edge=2%)")
    print("  Allocation: fraction of $1000 bankroll to each book")
    print(f"{'='*100}")
    header = f"{'Portfolio':<45} {'Bets':>4}  {'Wagered':>8}  {'P&L':>8}  {'ROI':>7}  {'MaxDD':>6}"
    print(header)
    print("-" * 100)

    book_map = {
        0.25: {0.25: r_ce_25, 0.35: r_ce_35},
    }

    for label, ce_kelly, ch_w, ch_kelly, alloc_ce, alloc_ch in configs:
        r_ce = run_book(ConsensusEnsembleModel(), gold, kelly=ce_kelly)
        r_ch = run_book(CrowdHedgedModel(crowd_weight=ch_w), gold, kelly=ch_kelly)

        combined_bets = r_ce.n_bets + r_ch.n_bets
        combined_wagered = r_ce.total_wagered * alloc_ce + r_ch.total_wagered * alloc_ch
        combined_pnl = r_ce.total_pnl * alloc_ce + r_ch.total_pnl * alloc_ch
        combined_roi = (combined_pnl / combined_wagered * 100) if combined_wagered > 0 else 0

        dd = compute_combined_drawdown(r_ce, r_ch, alloc_ce, alloc_ch)

        print(f"{label:<45} {combined_bets:>4}  ${combined_wagered:>7.0f}  ${combined_pnl:>+7.0f}  {combined_roi:>+6.1f}%  ${dd['dd_combined']:>5.0f}")

    # =====================================================================
    # WHAT-IF: Size up the hedged book
    # =====================================================================
    print(f"\n{'='*100}")
    print("  WHAT-IF: Hedged book gets more capital (60-70% allocation)")
    print(f"{'='*100}")
    print(header)
    print("-" * 100)

    for alloc_ch in [0.5, 0.6, 0.7, 0.8]:
        alloc_ce = 1.0 - alloc_ch
        r_ce = r_ce_25
        r_ch = r_ch20_40  # sized-up hedged book

        combined_bets = r_ce.n_bets + r_ch.n_bets
        combined_wagered = r_ce.total_wagered * alloc_ce + r_ch.total_wagered * alloc_ch
        combined_pnl = r_ce.total_pnl * alloc_ce + r_ch.total_pnl * alloc_ch
        combined_roi = (combined_pnl / combined_wagered * 100) if combined_wagered > 0 else 0

        dd = compute_combined_drawdown(r_ce, r_ch, alloc_ce, alloc_ch)

        label = f"CE(0.25):{alloc_ce:.0%} + CH20(0.40):{alloc_ch:.0%}"
        print(f"{label:<45} {combined_bets:>4}  ${combined_wagered:>7.0f}  ${combined_pnl:>+7.0f}  {combined_roi:>+6.1f}%  ${dd['dd_combined']:>5.0f}")


if __name__ == "__main__":
    main()
