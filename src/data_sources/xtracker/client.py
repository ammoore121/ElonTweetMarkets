"""
XTracker API Client - Polymarket's tweet counting oracle.
API docs: https://xtracker.polymarket.com/docs

This is the RESOLUTION SOURCE for Elon tweet markets.
All markets resolve based on XTracker data.

What XTracker counts:
- Original posts (YES)
- Reposts/retweets (YES)
- Quote tweets (YES)
- Replies (NO, unless on main feed)
- Deleted posts (YES, if captured within ~5 min)
"""
import requests
from datetime import datetime, date
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

BASE_URL = "https://xtracker.polymarket.com/api"


@dataclass
class DailyMetric:
    date: date
    post_count: int
    repost_count: int = 0
    quote_count: int = 0
    total_count: int = 0


class XTrackerClient:
    """Client for the XTracker API (Polymarket's tweet counting oracle)."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_user(self, handle: str = "elonmusk") -> dict:
        """Get user profile and current post count."""
        resp = self.session.get(f"{self.base_url}/users/{handle}")
        resp.raise_for_status()
        return resp.json()

    def get_daily_metrics(self, user_id: str, start_date: str, end_date: str) -> list[dict]:
        """
        Get daily tweet metrics for a user.

        Args:
            user_id: User ID (get from get_user())
            start_date: ISO date string (YYYY-MM-DD)
            end_date: ISO date string (YYYY-MM-DD)

        Returns:
            List of daily metric dicts with post counts
        """
        resp = self.session.get(
            f"{self.base_url}/metrics/{user_id}",
            params={"type": "daily", "startDate": start_date, "endDate": end_date}
        )
        resp.raise_for_status()
        return resp.json()

    def get_trackings(self, handle: str = "elonmusk") -> list[dict]:
        """Get tracking periods for a user."""
        resp = self.session.get(f"{self.base_url}/users/{handle}/trackings")
        resp.raise_for_status()
        return resp.json()

    def get_hourly_metrics(self, user_id: str, start_date: str, end_date: str) -> list[dict]:
        """Get hourly tweet metrics (if available)."""
        resp = self.session.get(
            f"{self.base_url}/metrics/{user_id}",
            params={"type": "hourly", "startDate": start_date, "endDate": end_date}
        )
        resp.raise_for_status()
        return resp.json()
