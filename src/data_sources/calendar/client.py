"""
Calendar event data sources for feature engineering.
- Tesla earnings dates (Yahoo Finance / SEC EDGAR)
- SpaceX launch schedule (Launch Library 2 API, free, 15 req/hr)
"""
import requests
from datetime import date, datetime
from dataclasses import dataclass
from typing import Optional

LAUNCH_LIBRARY_URL = "https://ll.thespacedevs.com/2.0.0"


class CalendarClient:
    """Fetch scheduled events that may affect Elon's tweeting patterns."""

    def __init__(self):
        self.session = requests.Session()

    def get_spacex_launches(self, start_date: str = None, end_date: str = None,
                            limit: int = 50) -> list[dict]:
        """
        Fetch SpaceX launches from Launch Library 2 API.
        Free tier: 15 req/hr.
        """
        params = {"lsp__name": "SpaceX", "limit": limit, "ordering": "net"}
        if start_date:
            params["net__gte"] = start_date
        if end_date:
            params["net__lte"] = end_date
        resp = self.session.get(f"{LAUNCH_LIBRARY_URL}/launch/", params=params)
        resp.raise_for_status()
        return resp.json().get("results", [])

    def get_upcoming_spacex_launches(self, limit: int = 10) -> list[dict]:
        resp = self.session.get(
            f"{LAUNCH_LIBRARY_URL}/launch/upcoming/",
            params={"lsp__name": "SpaceX", "limit": limit}
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    # Tesla earnings dates - known dates + extend as announced
    TESLA_EARNINGS = [
        ("2025-01-29", "Q4 2024", True),
        ("2025-04-22", "Q1 2025", True),
        ("2025-07-22", "Q2 2025", True),
        ("2025-10-21", "Q3 2025", True),
        ("2026-01-28", "Q4 2025", True),
        ("2026-04-28", "Q1 2026", True),
    ]

    def days_to_next_earnings(self, ref_date: date) -> int:
        for earnings_date_str, _, _ in self.TESLA_EARNINGS:
            earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
            if earnings_date >= ref_date:
                return (earnings_date - ref_date).days
        return 999

    def days_since_last_earnings(self, ref_date: date) -> int:
        for earnings_date_str, _, _ in reversed(self.TESLA_EARNINGS):
            earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
            if earnings_date <= ref_date:
                return (ref_date - earnings_date).days
        return 999
