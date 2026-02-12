"""
Feature engineering for Elon Musk tweet count prediction.

Feature groups:
1. Temporal: day-of-week, rolling averages, trend, volatility
2. Calendar: Tesla earnings, SpaceX launches
3. News: GDELT event volume/tone (future)
4. Flight: Jet tracking activity (future)
5. Market: Current Polymarket odds (future)
"""
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional


class TweetFeatureBuilder:
    """Build features for tweet count prediction."""

    def __init__(self, daily_counts: pd.DataFrame):
        """
        Args:
            daily_counts: DataFrame with columns [date, tweet_count]
                          sorted by date ascending.
        """
        self.daily_counts = daily_counts.copy()
        self.daily_counts['date'] = pd.to_datetime(self.daily_counts['date'])
        self.daily_counts = self.daily_counts.sort_values('date').reset_index(drop=True)

    def build_temporal_features(self, ref_date: date) -> dict:
        """
        Temporal pattern features based on historical tweet counts.
        All features use data BEFORE ref_date (no lookahead).
        """
        ref_dt = pd.Timestamp(ref_date)
        hist = self.daily_counts[self.daily_counts['date'] < ref_dt]

        if len(hist) < 7:
            return {}

        features = {}

        # Day of week (0=Monday, 6=Sunday)
        features['day_of_week'] = ref_date.weekday()
        features['is_weekend'] = 1 if ref_date.weekday() >= 5 else 0

        # Rolling averages
        for window in [3, 7, 14, 30]:
            recent = hist.tail(window)
            if len(recent) >= window:
                features[f'avg_tweets_{window}d'] = recent['tweet_count'].mean()
                features[f'std_tweets_{window}d'] = recent['tweet_count'].std()
                features[f'min_tweets_{window}d'] = recent['tweet_count'].min()
                features[f'max_tweets_{window}d'] = recent['tweet_count'].max()

        # Trend: slope of last 7 days
        if len(hist) >= 7:
            last_7 = hist.tail(7)['tweet_count'].values
            x = np.arange(7)
            slope = np.polyfit(x, last_7, 1)[0]
            features['trend_7d'] = slope

        # Yesterday's count
        if len(hist) >= 1:
            features['yesterday_count'] = hist.iloc[-1]['tweet_count']

        # 2 days ago
        if len(hist) >= 2:
            features['two_days_ago_count'] = hist.iloc[-2]['tweet_count']

        # Day-of-week historical average
        dow_hist = hist[hist['date'].dt.weekday == ref_date.weekday()]
        if len(dow_hist) >= 4:
            features['dow_avg'] = dow_hist['tweet_count'].mean()
            features['dow_std'] = dow_hist['tweet_count'].std()

        # Coefficient of variation (volatility indicator)
        if len(hist) >= 14:
            recent_14 = hist.tail(14)['tweet_count']
            features['cv_14d'] = recent_14.std() / recent_14.mean() if recent_14.mean() > 0 else 0

        # Ratio of yesterday to 7d average (regime change detection)
        if 'avg_tweets_7d' in features and features['avg_tweets_7d'] > 0:
            features['yesterday_vs_7d_ratio'] = features.get('yesterday_count', 0) / features['avg_tweets_7d']

        return features

    def build_all_features(self, ref_date: date, calendar_features: dict = None,
                           news_features: dict = None, flight_features: dict = None) -> dict:
        """Combine all feature groups."""
        features = self.build_temporal_features(ref_date)

        if calendar_features:
            features.update(calendar_features)
        if news_features:
            features.update(news_features)
        if flight_features:
            features.update(flight_features)

        return features
