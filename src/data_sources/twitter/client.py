"""
Tweet data loader for Elon Musk tweet history.
Primary source: Kaggle CSV datasets (free).
Backup: twikit/twscrape for recent data.

Kaggle datasets:
- "Elon Musk Tweets 2010-2025": https://www.kaggle.com/datasets/dadalyndell/elon-musk-tweets-2010-to-2025-march
- "Elon Musk Tweets (Daily Updated)": https://www.kaggle.com/datasets/aryansingh0909/elon-musk-tweets-updated-daily
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "tweets"


class TweetHistoryLoader:
    """Load and process historical tweet data from Kaggle CSV files."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def load_kaggle_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load Kaggle tweet dataset CSV.
        Expected columns: createdAt (or created_at), id, fullText (or text)

        Returns DataFrame with standardized columns: timestamp, tweet_id, text
        """
        df = pd.read_csv(filepath)

        # Standardize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if 'createdat' in lower or 'created_at' in lower or 'date' in lower:
                col_map[col] = 'timestamp'
            elif col.lower() == 'id':
                col_map[col] = 'tweet_id'
            elif 'text' in lower or 'fulltext' in lower or 'content' in lower:
                col_map[col] = 'text'

        df = df.rename(columns=col_map)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def compute_daily_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate tweets into daily counts.
        Returns DataFrame with columns: date, tweet_count
        """
        df = df.copy()
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date').agg(tweet_count=('tweet_id', 'count')).reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        return daily

    def compute_hourly_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get hourly tweet distribution (for feature engineering)."""
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        hourly = df.groupby(['date', 'hour']).agg(count=('tweet_id', 'count')).reset_index()
        return hourly
