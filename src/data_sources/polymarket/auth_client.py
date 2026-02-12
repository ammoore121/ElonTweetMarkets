"""
Polymarket Authenticated CLOB Client

Copied as-is from EsportsBetting/BettingMarkets (fully game-agnostic).
Provides authenticated access for historical price data and trade history.
"""

import os
import time
import hmac
import hashlib
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds
    HAS_CLOB_CLIENT = True
except ImportError:
    HAS_CLOB_CLIENT = False
    ClobClient = None
    ApiCreds = None

CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137
MAX_INTERVAL_SECONDS = 7 * 24 * 60 * 60  # 7 days max per request


class PolymarketAuthClient:
    def __init__(self, private_key: str = None):
        if not HAS_CLOB_CLIENT:
            raise ImportError("py-clob-client not installed. Run: pip install py-clob-client")
        self.private_key = private_key or os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not self.private_key:
            raise ValueError("Private key required. Set POLYMARKET_PRIVATE_KEY env var.")
        self.client = ClobClient(host=CLOB_HOST, key=self.private_key, chain_id=POLYGON_CHAIN_ID)
        self._api_creds = None
        self._init_credentials()

    def _init_credentials(self):
        try:
            self._api_creds = self.client.create_or_derive_api_creds()
            print("Polymarket API credentials initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize API credentials: {e}")

    def _sign_request(self, method: str, path: str) -> dict:
        timestamp = str(int(time.time()))
        message = timestamp + method + path
        signature = hmac.new(
            base64.b64decode(self._api_creds.api_secret),
            message.encode('utf-8'), hashlib.sha256
        ).digest()
        signature_b64 = base64.b64encode(signature).decode('utf-8')
        return {
            'POLY_API_KEY': self._api_creds.api_key,
            'POLY_SIGNATURE': signature_b64,
            'POLY_TIMESTAMP': timestamp,
            'POLY_PASSPHRASE': self._api_creds.api_passphrase,
        }

    def get_price_history(self, token_id: str, start_ts: int = None,
                          end_ts: int = None, fidelity: int = 60) -> pd.DataFrame:
        if not start_ts or not end_ts:
            raise ValueError("start_ts and end_ts are required")
        if (end_ts - start_ts) > MAX_INTERVAL_SECONDS:
            start_ts = end_ts - MAX_INTERVAL_SECONDS
        try:
            path = f"/prices-history?market={token_id}&startTs={start_ts}&endTs={end_ts}&fidelity={fidelity}"
            headers = self._sign_request("GET", path)
            response = requests.get(f"{CLOB_HOST}{path}", headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            history = data.get("history", [])
            if not history:
                return pd.DataFrame()
            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["t"], unit="s")
            df["price"] = df["p"].astype(float)
            df["token_id"] = token_id
            return df[["token_id", "timestamp", "price"]]
        except Exception as e:
            print(f"Error fetching price history: {e}")
            return pd.DataFrame()

    def get_full_price_history(self, token_id: str, start_ts: int, end_ts: int,
                                fidelity: int = 60) -> pd.DataFrame:
        all_history = []
        current_start = start_ts
        while current_start < end_ts:
            current_end = min(current_start + MAX_INTERVAL_SECONDS, end_ts)
            chunk = self.get_price_history(token_id=token_id, start_ts=current_start,
                                           end_ts=current_end, fidelity=fidelity)
            if not chunk.empty:
                all_history.append(chunk)
            current_start = current_end
            time.sleep(0.1)
        if not all_history:
            return pd.DataFrame()
        return pd.concat(all_history, ignore_index=True).drop_duplicates()

    def get_trades(self, token_id: str = None, market_id: str = None,
                   limit: int = 1000) -> pd.DataFrame:
        try:
            all_trades = []
            next_cursor = None
            while len(all_trades) < limit:
                params = {"limit": min(100, limit - len(all_trades))}
                if token_id:
                    params["asset_id"] = token_id
                if market_id:
                    params["market"] = market_id
                if next_cursor:
                    params["cursor"] = next_cursor
                response = self.client.get_trades(**params)
                trades = response.get("trades", [])
                all_trades.extend(trades)
                next_cursor = response.get("next_cursor")
                if not next_cursor or not trades:
                    break
                time.sleep(0.1)
            if not all_trades:
                return pd.DataFrame()
            df = pd.DataFrame(all_trades)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df
        except Exception as e:
            print(f"Error fetching trades: {e}")
            return pd.DataFrame()
