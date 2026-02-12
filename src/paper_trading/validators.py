"""
Paper Trading Validators for Elon Tweet Count Prediction Markets.

Validation rules to ensure data integrity throughout the paper trading pipeline.
Each validator returns a list of error strings (empty = valid).

Adapted from EsportsBetting validators for categorical (multi-bucket) markets.
"""

from __future__ import annotations

from typing import Optional

from .schemas import MarketOdds, Signal, Betslip, Settlement


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_odds(odds: MarketOdds) -> list[str]:
    """Validate a MarketOdds snapshot.

    Checks:
    - Required fields present
    - Bucket prices are valid (0-1)
    - Prices sum to approximately 1
    """
    errors = []

    if not odds.event_slug:
        errors.append("event_slug is required")

    if not odds.bucket_prices:
        errors.append("bucket_prices must not be empty")
    else:
        for label, price in odds.bucket_prices.items():
            if not (0.0 <= price <= 1.0):
                errors.append(
                    "bucket '{}' price must be 0-1, got {:.4f}".format(label, price)
                )

        # Bucket prices should sum to roughly 1.0 (allowing for market vig)
        total = sum(odds.bucket_prices.values())
        if not (0.85 <= total <= 1.20):
            errors.append(
                "bucket prices should sum to ~1.0, got {:.4f}".format(total)
            )

    if odds.n_buckets < 0:
        errors.append("n_buckets must be non-negative")

    if odds.bucket_prices and odds.n_buckets != len(odds.bucket_prices):
        errors.append(
            "n_buckets ({}) does not match len(bucket_prices) ({})".format(
                odds.n_buckets, len(odds.bucket_prices)
            )
        )

    return errors


def validate_signal(signal: Signal) -> list[str]:
    """Validate a Signal prediction.

    Checks:
    - Required foreign keys present
    - Predicted probabilities are valid and sum to ~1
    - Kelly sizing is reasonable
    - Best bucket fields are consistent
    """
    errors = []

    if not signal.odds_id:
        errors.append("odds_id is required")

    if not signal.model_id:
        errors.append("model_id is required")

    # Predicted probs validation
    if not signal.predicted_probs:
        errors.append("predicted_probs must not be empty")
    else:
        for label, prob in signal.predicted_probs.items():
            if not (0.0 <= prob <= 1.0):
                errors.append(
                    "predicted_probs['{}'] must be 0-1, got {:.4f}".format(
                        label, prob
                    )
                )

        total = sum(signal.predicted_probs.values())
        if not (0.95 <= total <= 1.05):
            errors.append(
                "predicted_probs should sum to ~1.0, got {:.4f}".format(total)
            )

    # Best bucket consistency
    if signal.best_bucket and signal.predicted_probs:
        if signal.best_bucket not in signal.predicted_probs:
            errors.append(
                "best_bucket '{}' not found in predicted_probs".format(
                    signal.best_bucket
                )
            )

    # Kelly fraction bounds
    if signal.kelly_fraction < 0 or signal.kelly_fraction > 1:
        errors.append(
            "kelly_fraction must be 0-1, got {:.4f}".format(signal.kelly_fraction)
        )

    # Wager must be non-negative
    if signal.recommended_wager < 0:
        errors.append(
            "recommended_wager must be non-negative, got {:.2f}".format(
                signal.recommended_wager
            )
        )

    return errors


def validate_betslip(betslip: Betslip, signal: Optional[Signal] = None) -> list[str]:
    """Validate a Betslip.

    Checks:
    - Required fields present
    - Pricing is consistent (edge = model_prob - price_paid)
    - Sizing is reasonable
    - Cross-references signal if provided
    """
    errors = []

    if not betslip.signal_id:
        errors.append("signal_id is required")

    if not betslip.event_slug:
        errors.append("event_slug is required")

    if not betslip.bucket_label:
        errors.append("bucket_label is required")

    if betslip.price_paid <= 0 or betslip.price_paid >= 1:
        errors.append(
            "price_paid must be between 0 and 1 exclusive, got {:.4f}".format(
                betslip.price_paid
            )
        )

    if betslip.wager < 0:
        errors.append("wager must be non-negative")

    if betslip.model_prob < 0 or betslip.model_prob > 1:
        errors.append(
            "model_prob must be 0-1, got {:.4f}".format(betslip.model_prob)
        )

    # Edge consistency check
    expected_edge = betslip.model_prob - betslip.price_paid
    if abs(betslip.edge_at_bet - expected_edge) > 0.01:
        errors.append(
            "edge_at_bet ({:.4f}) != model_prob - price_paid ({:.4f})".format(
                betslip.edge_at_bet, expected_edge
            )
        )

    # Shares consistency check
    if betslip.price_paid > 0 and betslip.wager > 0:
        expected_shares = betslip.wager / betslip.price_paid
        if abs(betslip.shares - expected_shares) > 0.01:
            errors.append(
                "shares ({:.4f}) != wager / price_paid ({:.4f})".format(
                    betslip.shares, expected_shares
                )
            )

    # Cross-reference with signal
    if signal is not None:
        if betslip.signal_id != signal.signal_id:
            errors.append("signal_id mismatch")
        if betslip.bucket_label != signal.best_bucket:
            errors.append(
                "bucket_label '{}' != signal.best_bucket '{}'".format(
                    betslip.bucket_label, signal.best_bucket
                )
            )

    return errors


def validate_settlement(
    settlement: Settlement, betslip: Optional[Betslip] = None
) -> list[str]:
    """Validate a Settlement.

    Checks:
    - Required fields present
    - P&L arithmetic is correct
    - Cross-references betslip if provided
    """
    errors = []

    if not settlement.betslip_id:
        errors.append("betslip_id is required")

    if not settlement.winning_bucket:
        errors.append("winning_bucket is required")

    if not settlement.bucket_bet:
        errors.append("bucket_bet is required")

    # Won flag consistency
    expected_won = settlement.bucket_bet == settlement.winning_bucket
    if settlement.won != expected_won:
        errors.append(
            "won ({}) doesn't match bucket_bet vs winning_bucket".format(
                settlement.won
            )
        )

    # P&L consistency
    expected_pnl = settlement.payout - settlement.wager
    if abs(settlement.pnl - expected_pnl) > 0.01:
        errors.append(
            "pnl ({:.2f}) != payout - wager ({:.2f})".format(
                settlement.pnl, expected_pnl
            )
        )

    # Cross-reference with betslip
    if betslip is not None:
        if settlement.betslip_id != betslip.betslip_id:
            errors.append("betslip_id mismatch")

        if settlement.bucket_bet != betslip.bucket_label:
            errors.append(
                "bucket_bet '{}' != betslip.bucket_label '{}'".format(
                    settlement.bucket_bet, betslip.bucket_label
                )
            )

        if abs(settlement.wager - betslip.wager) > 0.01:
            errors.append(
                "wager ({:.2f}) != betslip.wager ({:.2f})".format(
                    settlement.wager, betslip.wager
                )
            )

        # Payout check: if won, payout = shares; if lost, payout = 0
        expected_payout = betslip.shares if settlement.won else 0.0
        if abs(settlement.payout - expected_payout) > 0.01:
            errors.append(
                "payout ({:.2f}) != expected ({:.2f})".format(
                    settlement.payout, expected_payout
                )
            )

    return errors
