"""
Module 2: Financial Machine Learning Pipeline
Implementing López de Prado's advanced methods for financial ML
Optimized for Hetzner Cloud (4 vCPU, 8GB RAM)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from statsmodels.tsa.stattools import adfuller
import sqlite3
import pickle
from pathlib import Path
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LabelingResult:
    """Container for labeling results"""
    labels: pd.Series
    events: pd.DataFrame
    sample_weights: pd.Series
    returns: pd.Series


class FractionalDifferentiator(BaseEstimator, TransformerMixin):
    """
    Fractional Differentiation following de Prado Chapter 5
    Achieves stationarity while preserving maximum memory
    """
    def __init__(self, d: float = 0.5, threshold: float = 1e-5):
        self.d = d
        self.threshold = threshold
        self.weights_: Optional[np.ndarray] = None

    def _get_weights(self, size: int) -> np.ndarray:
        """Generate weights for fractional differentiation"""
        w = [1.0]
        k = 1
        while k < size:
            w_k = -w[-1] / k * (self.d - k + 1)
            if abs(w_k) < self.threshold:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the fractional differentiator"""
        self.weights_ = self._get_weights(len(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fractional differentiation"""
        if self.weights_ is None:
            raise ValueError("Must fit before transform")

        width = len(self.weights_) - 1
        result = {}

        for name in X.columns:
            # Use ffill() instead of deprecated fillna(method='ffill')
            series = X[name].ffill().dropna()
            diff_series = pd.Series(index=series.index, dtype=np.float64)

            for iloc in range(width, len(series)):
                loc = series.index[iloc]
                if not np.isfinite(series.loc[loc]):
                    continue

                start_idx = iloc - width
                values = series.iloc[start_idx:iloc + 1].values
                diff_series.loc[loc] = np.dot(self.weights_.T, values.reshape(-1, 1))[0, 0]

            result[name] = diff_series

        return pd.DataFrame(result)

    def find_optimal_d(self, series: pd.Series, max_d: float = 1.0,
                       step: float = 0.1, p_value_threshold: float = 0.05) -> float:
        """Find minimum d value that achieves stationarity"""
        for d in np.arange(0, max_d + step, step):
            self.d = d
            self.fit(pd.DataFrame({'series': series}))
            diff_series = self.transform(pd.DataFrame({'series': series}))['series']

            clean_series = diff_series.dropna()
            if len(clean_series) > 10:
                try:
                    adf_stat, p_value, _, _, _, _ = adfuller(
                        clean_series, maxlag=1, regression='c', autolag=None
                    )
                    if p_value < p_value_threshold:
                        logger.debug(f"Found optimal d={d:.2f} with p-value={p_value:.4f}")
                        return d
                except Exception as e:
                    logger.warning(f"ADF test failed for d={d}: {e}")
                    continue

        return max_d


class VolatilityEstimator:
    """
    Daily volatility estimation for dynamic thresholds
    Following de Prado Chapter 3, Section 3.3
    """
    def __init__(self, span: int = 100):
        self.span = span

    def estimate_daily_vol(self, close: pd.Series) -> pd.Series:
        """Estimate daily volatility using exponentially weighted moving average"""
        # Get daily returns
        df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        
        if len(df0) == 0:
            # Fallback: use simple returns if not enough data
            returns = close.pct_change()
            return returns.ewm(span=self.span).std()
        
        df0 = pd.Series(
            close.index[df0 - 1],
            index=close.index[close.shape[0] - df0.shape[0]:]
        )

        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
        df0 = df0.ewm(span=self.span).std()
        return df0


class TripleBarrierLabeler:
    """
    Triple-barrier labeling method
    Following de Prado Chapter 3, Section 3.4
    """
    def __init__(self, pt_sl: List[float] = None, min_ret: float = 0.005):
        self.pt_sl = pt_sl or [1.0, 1.0]  # [profit_taking_factor, stop_loss_factor]
        self.min_ret = min_ret
        self.vol_estimator = VolatilityEstimator()

    def get_events(self, close: pd.Series, t_events: pd.DatetimeIndex,
                   num_days: int = 1) -> pd.DataFrame:
        """
        Get triple-barrier events
        """
        # Get target (daily volatility)
        trgt = self.vol_estimator.estimate_daily_vol(close)
        trgt = trgt.reindex(t_events)
        trgt = trgt[trgt > self.min_ret]

        if trgt.empty:
            return pd.DataFrame()

        # Get vertical barrier (max holding period)
        t1 = close.index.searchsorted(
            t_events + pd.Timedelta(days=num_days)
        )
        t1 = t1[t1 < close.shape[0]]
        t1 = pd.Series(
            close.index[t1],
            index=t_events[:t1.shape[0]]
        )

        # Combine into events DataFrame
        events = pd.concat({'t1': t1, 'trgt': trgt}, axis=1).dropna(subset=['trgt'])
        return events

    def apply_barriers(self, close: pd.Series, events: pd.DataFrame,
                       num_threads: int = None) -> pd.DataFrame:
        """Apply triple barriers to events"""
        if events.empty:
            return events
            
        events_ = events.copy()
        events_['side'] = 1.0  # Buy-only: always long

        # Use multiprocessing for large datasets
        if num_threads is None:
            num_threads = min(mp.cpu_count(), 4)

        barrier_results = []
        
        # Process in parallel for better performance
        if len(events_) > 100 and num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(self._apply_single_barrier, close, loc, event)
                    for loc, event in events_.iterrows()
                ]
                barrier_results = [f.result() for f in futures]
        else:
            for loc, event in events_.iterrows():
                result = self._apply_single_barrier(close, loc, event)
                barrier_results.append(result)

        barrier_df = pd.DataFrame(barrier_results, index=events_.index)
        
        # Update t1 with first barrier touch
        t1_update = barrier_df[['sl', 'pt']].apply(
            lambda x: x.dropna().min() if x.notna().any() else pd.NaT, axis=1
        )
        events_['t1'] = t1_update.combine_first(events_['t1'])

        return events_

    def _apply_single_barrier(self, close: pd.Series, start_time: pd.Timestamp,
                              event: pd.Series) -> Dict:
        """Apply barriers to single event"""
        try:
            end_time = event['t1']
            if pd.isna(end_time):
                return {'sl': None, 'pt': None}
                
            path = close[start_time:end_time]

            if len(path) < 2:
                return {'sl': None, 'pt': None}

            # Calculate returns
            returns = (path / close[start_time] - 1) * event['side']

            # Define barriers
            pt_barrier = self.pt_sl[0] * event['trgt']
            sl_barrier = -self.pt_sl[1] * event['trgt']

            # Find first barrier touch
            pt_touches = returns[returns > pt_barrier]
            sl_touches = returns[returns < sl_barrier]
            
            pt_touch = pt_touches.index.min() if not pt_touches.empty else None
            sl_touch = sl_touches.index.min() if not sl_touches.empty else None

            return {'sl': sl_touch, 'pt': pt_touch}

        except Exception as e:
            logger.warning(f"Error applying barriers at {start_time}: {e}")
            return {'sl': None, 'pt': None}

    def get_bins(self, events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Get labels from barrier touches"""
        events_ = events.dropna(subset=['t1'])

        if events_.empty:
            return pd.DataFrame()

        # Get prices at event times
        px = events_.index.union(events_['t1'].values).drop_duplicates()
        px = close.reindex(px, method='bfill')

        # Create output DataFrame
        out = pd.DataFrame(index=events_.index)
        out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

        if 'side' in events_:
            out['ret'] *= events_['side']
            out['bin'] = np.sign(out['ret'])
            out.loc[out['ret'] <= 0, 'bin'] = 0  # Meta-labeling: 0 for no bet
        else:
            out['bin'] = np.sign(out['ret'])

        return out


class SampleWeighter:
    """
    Advanced sample weighting for non-IID financial data
    Following de Prado Chapter 4
    """

    def get_sample_weights(self, events: pd.DataFrame, close: pd.Series,
                           num_co_events: Optional[pd.Series] = None) -> pd.Series:
        """Get sample weights considering uniqueness and return attribution"""
        if events.empty:
            return pd.Series(dtype=float)
            
        if num_co_events is None:
            num_co_events = self._get_num_co_events(events, close)

        # Calculate average uniqueness
        sample_weights = self._get_avg_uniqueness(events, num_co_events)

        # Apply return attribution weighting
        return_weights = self._get_return_attribution_weights(
            events, close, num_co_events
        )

        # Combine weights
        combined_weights = sample_weights * return_weights

        # Handle edge cases
        combined_weights = combined_weights.replace([np.inf, -np.inf], np.nan)
        combined_weights = combined_weights.fillna(combined_weights.median())

        # Normalize to sum to number of samples
        if combined_weights.sum() > 0:
            combined_weights *= len(combined_weights) / combined_weights.sum()

        return combined_weights

    def _get_num_co_events(self, events: pd.DataFrame, close: pd.Series) -> pd.Series:
        """Calculate number of concurrent events at each timestamp"""
        events_filled = events.copy()
        events_filled['t1'] = events_filled['t1'].fillna(close.index[-1])

        # Create series to count overlapping events
        count = pd.Series(0, index=close.index)

        for start_time, row in events_filled.iterrows():
            end_time = row['t1']
            mask = (close.index >= start_time) & (close.index <= end_time)
            count.loc[mask] += 1

        return count

    def _get_avg_uniqueness(self, events: pd.DataFrame,
                            num_co_events: pd.Series) -> pd.Series:
        """Calculate average uniqueness over event lifespan"""
        weights = pd.Series(index=events.index, dtype=float)

        for start_time, row in events.iterrows():
            end_time = row['t1']
            if pd.isna(end_time):
                continue

            event_mask = (num_co_events.index >= start_time) & \
                         (num_co_events.index <= end_time)
            event_concurrent = num_co_events[event_mask]

            if len(event_concurrent) > 0 and (event_concurrent > 0).any():
                uniqueness = (1.0 / event_concurrent.replace(0, 1)).mean()
                weights.loc[start_time] = uniqueness

        return weights.fillna(0)

    def _get_return_attribution_weights(self, events: pd.DataFrame,
                                        close: pd.Series,
                                        num_co_events: pd.Series) -> pd.Series:
        """Weight by attributed returns"""
        ret = np.log(close).diff()
        weights = pd.Series(index=events.index, dtype=float)

        for start_time, row in events.iterrows():
            end_time = row['t1']
            if pd.isna(end_time):
                continue

            event_mask = (ret.index >= start_time) & (ret.index <= end_time)
            event_returns = ret[event_mask]
            event_concurrent = num_co_events[event_mask]

            if len(event_returns) > 0 and len(event_concurrent) > 0:
                # Replace zeros to avoid division errors
                safe_concurrent = event_concurrent.replace(0, 1)
                attributed_ret = (event_returns / safe_concurrent).sum()
                weights.loc[start_time] = abs(attributed_ret)

        return weights.fillna(0)


class PurgedKFold:
    """
    Purged K-Fold cross-validation to prevent leakage
    Following de Prado Chapter 7, Section 7.4
    """

    def __init__(self, n_splits: int = 5, pct_embargo: float = 0.01):
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo

    def split(self, X: pd.DataFrame, events: Optional[pd.DataFrame] = None) -> List[Tuple]:
        """Generate train/test splits with purging and embargo"""
        indices = np.arange(X.shape[0])
        embargo_size = int(X.shape[0] * self.pct_embargo)

        # Create test splits
        test_splits = np.array_split(indices, self.n_splits)

        splits = []
        for test_indices in test_splits:
            test_indices = test_indices.tolist()
            
            if events is not None and not events.empty:
                # Get test times
                test_times = X.index[test_indices]
                min_test_time = test_times.min()
                max_test_time = test_times.max()

                # Purge training set
                train_indices = self._purge_train_set(
                    events, indices, min_test_time, max_test_time
                )
            else:
                # Simple split without purging
                train_indices = [i for i in indices if i not in test_indices]

            # Apply embargo
            if embargo_size > 0 and len(test_indices) > 0:
                max_test_idx = max(test_indices)
                embargo_end = min(max_test_idx + embargo_size + 1, len(X))
                embargo_indices = set(range(max_test_idx + 1, embargo_end))
                train_indices = [i for i in train_indices if i not in embargo_indices]

            splits.append((train_indices, test_indices))

        return splits

    def _purge_train_set(self, events: pd.DataFrame, indices: np.ndarray,
                         min_test_time: pd.Timestamp,
                         max_test_time: pd.Timestamp) -> List[int]:
        """Purge training set of overlapping observations"""
        train_indices = []

        for idx in indices:
            if idx >= len(events):
                continue
                
            event_start = events.index[idx]
            event_end = events.iloc[idx]['t1']

            if pd.isna(event_end):
                event_end = events.index[-1]

            # Check for overlap with test period
            no_overlap = (event_end < min_test_time) or (event_start > max_test_time)

            if no_overlap:
                train_indices.append(idx)

        return train_indices


class FeatureImportanceAnalyzer:
    """
    Feature importance analysis following de Prado Chapter 8
    """

    def mean_decrease_impurity(self, clf, feature_names: List[str]) -> pd.DataFrame:
        """Calculate Mean Decrease Impurity (MDI) feature importance"""
        if not hasattr(clf, 'estimators_'):
            raise ValueError("Classifier must be a tree-based ensemble")

        # Get feature importances from each tree
        importances = {}
        for i, estimator in enumerate(clf.estimators_):
            # Handle both direct trees and wrapped estimators
            if hasattr(estimator, 'feature_importances_'):
                importances[i] = estimator.feature_importances_
            elif hasattr(estimator, 'estimators_'):
                # Nested ensemble
                nested_imp = np.mean([e.feature_importances_ for e in estimator.estimators_], axis=0)
                importances[i] = nested_imp

        if not importances:
            return pd.DataFrame(columns=['mean', 'std'], index=feature_names)

        imp_df = pd.DataFrame.from_dict(importances, orient='index')
        imp_df.columns = feature_names[:imp_df.shape[1]]
        imp_df = imp_df.replace(0, np.nan)

        # Calculate mean and std
        result = pd.DataFrame({
            'mean': imp_df.mean(),
            'std': imp_df.std() * imp_df.shape[0] ** -0.5
        })

        # Normalize so importances sum to 1
        if result['mean'].sum() > 0:
            result['mean'] = result['mean'] / result['mean'].sum()

        return result

    def mean_decrease_accuracy(self, clf, X: pd.DataFrame, y: pd.Series,
                               sample_weight: pd.Series, cv_gen: PurgedKFold,
                               scoring: str = 'neg_log_loss',
                               n_jobs: int = -1) -> Tuple[pd.DataFrame, float]:
        """Calculate Mean Decrease Accuracy (MDA) feature importance"""
        
        base_scores = []
        permuted_scores = {col: [] for col in X.columns}

        for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X, None)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
                
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train = sample_weight.iloc[train_idx]
            w_test = sample_weight.iloc[test_idx]

            # Clone and fit classifier
            from sklearn.base import clone
            clf_fitted = clone(clf)
            clf_fitted.fit(X_train, y_train, sample_weight=w_train.values)

            # Get baseline score
            if scoring == 'neg_log_loss':
                try:
                    prob = clf_fitted.predict_proba(X_test)
                    base_score = -log_loss(y_test, prob, sample_weight=w_test.values)
                except Exception:
                    pred = clf_fitted.predict(X_test)
                    base_score = accuracy_score(y_test, pred, sample_weight=w_test.values)
            else:
                pred = clf_fitted.predict(X_test)
                base_score = accuracy_score(y_test, pred, sample_weight=w_test.values)

            base_scores.append(base_score)

            # Permute each feature and calculate score
            for feature in X.columns:
                X_test_perm = X_test.copy()
                X_test_perm[feature] = np.random.permutation(X_test_perm[feature].values)

                if scoring == 'neg_log_loss':
                    try:
                        prob_perm = clf_fitted.predict_proba(X_test_perm)
                        perm_score = -log_loss(y_test, prob_perm, sample_weight=w_test.values)
                    except Exception:
                        pred_perm = clf_fitted.predict(X_test_perm)
                        perm_score = accuracy_score(y_test, pred_perm, sample_weight=w_test.values)
                else:
                    pred_perm = clf_fitted.predict(X_test_perm)
                    perm_score = accuracy_score(y_test, pred_perm, sample_weight=w_test.values)

                permuted_scores[feature].append(perm_score)

        # Calculate importance as performance difference
        base_score_mean = np.mean(base_scores) if base_scores else 0
        importance = pd.DataFrame(index=X.columns, columns=['mean', 'std'], dtype=float)

        for feature in X.columns:
            perm_scores = np.array(permuted_scores[feature])
            if len(perm_scores) > 0:
                diff = base_score_mean - perm_scores.mean()

                if scoring == 'neg_log_loss':
                    importance.loc[feature, 'mean'] = diff / abs(base_score_mean) if base_score_mean != 0 else 0
                else:
                    denom = 1 - perm_scores.mean()
                    importance.loc[feature, 'mean'] = diff / denom if denom != 0 else 0

                importance.loc[feature, 'std'] = perm_scores.std() * len(perm_scores) ** -0.5
            else:
                importance.loc[feature, 'mean'] = 0
                importance.loc[feature, 'std'] = 0

        return importance, base_score_mean


class FinancialMLPipeline:
    """
    Main ML Pipeline integrating all de Prado methods
    Optimized for Hetzner Cloud
    """

    def __init__(self, db_path: str = "/opt/financial_ml/data/financial_data.db",
                 models_dir: str = "/opt/financial_ml/models",
                 n_jobs: int = -1):
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Set number of parallel jobs (use all available CPUs on Hetzner)
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()

        # Components
        self.frac_diff = FractionalDifferentiator()
        self.labeler = TripleBarrierLabeler()
        self.weighter = SampleWeighter()
        self.cv = PurgedKFold(n_splits=5, pct_embargo=0.02)
        self.feat_analyzer = FeatureImportanceAnalyzer()
        self.scaler = StandardScaler()

        # Storage for fitted components
        self.fitted_models: Dict[str, Dict] = {}
        self.feature_importance_results: Dict[str, Dict] = {}

    def load_data(self, symbol: str, bar_type: str = 'dollar',
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """Load bar data from database"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT timestamp, open, high, low, close, volume, vwap, dollar_volume
            FROM bars 
            WHERE symbol = ? AND bar_type = ?
        """
        params: List[Any] = [symbol, bar_type]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        return df

    def create_features(self, data: pd.DataFrame, 
                        apply_frac_diff: bool = True) -> pd.DataFrame:
        """Create features from bar data"""
        if data.empty:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=data.index)

        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close']).diff()
        features['volatility'] = features['returns'].rolling(20).std()

        # Technical features
        features['rsi'] = self._calculate_rsi(data['close'])
        features['bollinger_position'] = self._calculate_bollinger_position(data['close'])
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        # Microstructure features
        if 'vwap' in data.columns:
            features['vwap_ratio'] = data['close'] / data['vwap']
        features['price_range'] = (data['high'] - data['low']) / data['close']

        # Moving average features
        for window in [5, 10, 20, 50]:
            ma = data['close'].rolling(window).mean()
            features[f'ma_{window}_ratio'] = data['close'] / ma

        # Momentum features
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['close'].pct_change(period)

        # Apply fractional differentiation to price-based features
        if apply_frac_diff:
            price_features = ['returns', 'log_returns', 'rsi', 'bollinger_position']

            for feature in price_features:
                if feature in features.columns:
                    try:
                        # Find optimal d for stationarity
                        series = features[feature].dropna()
                        if len(series) > 50:
                            optimal_d = self.frac_diff.find_optimal_d(series)
                            self.frac_diff.d = optimal_d
                            frac_diff_feature = self.frac_diff.fit_transform(
                                features[[feature]]
                            )[feature]
                            features[f'{feature}_fracdiff'] = frac_diff_feature
                    except Exception as e:
                        logger.warning(f"Fractional diff failed for {feature}: {e}")

        return features.dropna()

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        band_width = upper - lower
        return (prices - lower) / band_width.replace(0, np.inf)

    def generate_labels(self, data: pd.DataFrame,
                        event_filter_threshold: float = 0.01,
                        max_holding_days: int = 1) -> LabelingResult:
        """Generate labels using triple-barrier method"""
        if data.empty:
            return LabelingResult(
                labels=pd.Series(dtype=float),
                events=pd.DataFrame(),
                sample_weights=pd.Series(dtype=float),
                returns=pd.Series(dtype=float)
            )

        close = data['close']

        # Generate events using regular sampling
        # In production, use CUSUM filter from data_ingestion module
        event_frequency = pd.Timedelta(hours=4)
        t_events = pd.date_range(
            start=close.index[0],
            end=close.index[-1],
            freq=event_frequency
        )
        t_events = pd.DatetimeIndex([t for t in t_events if t in close.index])

        if len(t_events) == 0:
            return LabelingResult(
                labels=pd.Series(dtype=float),
                events=pd.DataFrame(),
                sample_weights=pd.Series(dtype=float),
                returns=pd.Series(dtype=float)
            )

        # Get events DataFrame
        events = self.labeler.get_events(close, t_events, num_days=max_holding_days)

        if events.empty:
            return LabelingResult(
                labels=pd.Series(dtype=float),
                events=pd.DataFrame(),
                sample_weights=pd.Series(dtype=float),
                returns=pd.Series(dtype=float)
            )

        # Apply barriers
        events_with_barriers = self.labeler.apply_barriers(close, events)

        # Generate labels
        labels_df = self.labeler.get_bins(events_with_barriers, close)

        if labels_df.empty:
            return LabelingResult(
                labels=pd.Series(dtype=float),
                events=events_with_barriers,
                sample_weights=pd.Series(dtype=float),
                returns=pd.Series(dtype=float)
            )

        # Calculate sample weights
        sample_weights = self.weighter.get_sample_weights(
            events_with_barriers, close
        )

        return LabelingResult(
            labels=labels_df['bin'],
            events=events_with_barriers,
            sample_weights=sample_weights,
            returns=labels_df['ret']
        )

    def train_model(self, features: pd.DataFrame, labeling_result: LabelingResult,
                    model_params: Optional[Dict] = None) -> Dict:
        """Train ML model with proper cross-validation"""

        # Align features and labels
        common_index = features.index.intersection(labeling_result.labels.index)
        
        if len(common_index) < 50:
            logger.warning(f"Insufficient samples for training: {len(common_index)}")
            return {}

        X = features.loc[common_index]
        y = labeling_result.labels.loc[common_index]
        sample_weights = labeling_result.sample_weights.reindex(common_index).fillna(1.0)
        events = labeling_result.events.reindex(common_index)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )

        # Set up classifier (using 'estimator' instead of deprecated 'base_estimator')
        if model_params is None:
            model_params = {
                'n_estimators': 500,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': self.n_jobs
            }

        base_clf = DecisionTreeClassifier(
            criterion='entropy',
            max_features='sqrt',
            class_weight='balanced',
            max_depth=10
        )

        clf = BaggingClassifier(
            estimator=base_clf,  # Updated from deprecated 'base_estimator'
            n_estimators=model_params.get('n_estimators', 500),
            max_samples=min(0.8, sample_weights.mean()),
            max_features=1.0,
            oob_score=True,
            n_jobs=self.n_jobs,
            random_state=model_params.get('random_state', 42)
        )

        # Set up cross-validation with events
        events_for_cv = pd.DataFrame({'t1': events['t1']}, index=events.index)
        cv_gen = PurgedKFold(n_splits=5, pct_embargo=0.02)

        # Calculate cross-validated scores
        cv_scores = []
        for train_idx, test_idx in cv_gen.split(X_scaled, events_for_cv):
            if len(train_idx) < 20 or len(test_idx) < 10:
                continue

            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train, w_test = sample_weights.iloc[train_idx], sample_weights.iloc[test_idx]

            clf_fold = BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    criterion='entropy',
                    max_features='sqrt',
                    class_weight='balanced',
                    max_depth=8
                ),
                n_estimators=100,
                max_samples=min(0.8, w_train.mean()),
                max_features=1.0,
                n_jobs=self.n_jobs
            )

            clf_fold.fit(X_train, y_train, sample_weight=w_train.values)

            # Score
            if len(y.unique()) == 2:  # Binary classification
                pred = clf_fold.predict(X_test)
                score = f1_score(y_test, pred, sample_weight=w_test.values, average='weighted')
            else:
                try:
                    prob = clf_fold.predict_proba(X_test)
                    score = -log_loss(y_test, prob, sample_weight=w_test.values)
                except Exception:
                    pred = clf_fold.predict(X_test)
                    score = accuracy_score(y_test, pred, sample_weight=w_test.values)

            cv_scores.append(score)

        # Fit final model on all data
        clf.fit(X_scaled, y, sample_weight=sample_weights.values)

        # Calculate feature importance
        try:
            mdi_importance = self.feat_analyzer.mean_decrease_impurity(clf, X.columns.tolist())
        except Exception as e:
            logger.warning(f"MDI calculation failed: {e}")
            mdi_importance = pd.DataFrame(columns=['mean', 'std'], index=X.columns)

        try:
            mda_importance, base_score = self.feat_analyzer.mean_decrease_accuracy(
                clf, X_scaled, y, sample_weights, cv_gen
            )
        except Exception as e:
            logger.warning(f"MDA calculation failed: {e}")
            mda_importance = pd.DataFrame(columns=['mean', 'std'], index=X.columns)
            base_score = 0

        # Store results
        model_key = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        results = {
            'model': clf,
            'scaler': self.scaler,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores) if cv_scores else 0,
            'cv_std': np.std(cv_scores) if cv_scores else 0,
            'oob_score': clf.oob_score_ if hasattr(clf, 'oob_score_') else 0,
            'feature_importance_mdi': mdi_importance,
            'feature_importance_mda': mda_importance,
            'base_score': base_score,
            'features': X.columns.tolist(),
            'n_samples': len(X)
        }

        self.fitted_models[model_key] = results

        # Log results
        logger.info(f"Model trained successfully: {model_key}")
        logger.info(f"CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        logger.info("Top 5 Important Features (MDI):")
        top_features = mdi_importance.sort_values('mean', ascending=False).head()
        for feature, row in top_features.iterrows():
            logger.info(f"  {feature}: {row['mean']:.4f}")

        # Save model
        self._save_model(model_key, results)

        return results

    def _save_model(self, model_key: str, results: Dict):
        """Save model to disk"""
        model_path = self.models_dir / f"{model_key}.pkl"
        
        # Save only serializable parts
        save_data = {
            'model': results['model'],
            'scaler': results['scaler'],
            'features': results['features'],
            'cv_mean': results['cv_mean'],
            'cv_std': results['cv_std'],
            'oob_score': results['oob_score']
        }
        
        joblib.dump(save_data, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_key: str) -> Optional[Dict]:
        """Load model from disk"""
        model_path = self.models_dir / f"{model_key}.pkl"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        return joblib.load(model_path)

    def predict(self, features: pd.DataFrame, model_key: str) -> pd.DataFrame:
        """Make predictions using trained model"""
        if model_key not in self.fitted_models:
            # Try loading from disk
            loaded = self.load_model(model_key)
            if loaded is None:
                raise ValueError(f"Model {model_key} not found")
            self.fitted_models[model_key] = loaded

        model_info = self.fitted_models[model_key]
        clf = model_info['model']
        scaler = model_info.get('scaler')

        # Ensure features match training features
        model_features = model_info['features']
        missing_features = set(model_features) - set(features.columns)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feat in missing_features:
                features[feat] = 0

        X = features[model_features]

        # Scale features
        if scaler is not None:
            X_scaled = pd.DataFrame(
                scaler.transform(X),
                index=X.index,
                columns=X.columns
            )
        else:
            X_scaled = X

        # Make predictions
        predictions = clf.predict(X_scaled)
        probabilities = clf.predict_proba(X_scaled)

        # Create results DataFrame
        results = pd.DataFrame(index=X.index)
        results['prediction'] = predictions

        # Add probability columns
        for i, class_label in enumerate(clf.classes_):
            results[f'prob_{class_label}'] = probabilities[:, i]

        return results

    def analyze_feature_importance(self, model_key: str) -> Dict:
        """Comprehensive feature importance analysis"""
        if model_key not in self.fitted_models:
            raise ValueError(f"Model {model_key} not found")

        model_info = self.fitted_models[model_key]

        # Get importance rankings
        mdi_ranking = model_info['feature_importance_mdi'].sort_values(
            'mean', ascending=False
        )
        mda_ranking = model_info['feature_importance_mda'].sort_values(
            'mean', ascending=False
        )

        # Correlation between importance methods
        common_features = mdi_ranking.index.intersection(mda_ranking.index)
        importance_correlation = np.nan
        
        if len(common_features) > 1:
            from scipy.stats import spearmanr
            try:
                correlation = spearmanr(
                    mdi_ranking.loc[common_features, 'mean'],
                    mda_ranking.loc[common_features, 'mean']
                )
                importance_correlation = correlation.correlation
            except Exception:
                pass

        return {
            'mdi_ranking': mdi_ranking,
            'mda_ranking': mda_ranking,
            'importance_correlation': importance_correlation,
            'top_mdi_features': mdi_ranking.head().index.tolist(),
            'top_mda_features': mda_ranking.head().index.tolist()
        }


# Entry point for module execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Financial ML Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--symbol', default='BTC/USD')
    parser.add_argument('--bar-type', default='dollar')
    parser.add_argument('--db-path', default='/opt/financial_ml/data/financial_data.db')
    parser.add_argument('--models-dir', default='/opt/financial_ml/models')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FinancialMLPipeline(
        db_path=args.db_path,
        models_dir=args.models_dir
    )

    # Load data
    data = pipeline.load_data(args.symbol, args.bar_type)

    if data.empty:
        logger.error("No data available")
        exit(1)

    logger.info(f"Loaded {len(data)} bars of data")

    if args.mode == 'train':
        # Create features
        features = pipeline.create_features(data)
        logger.info(f"Created {len(features.columns)} features")

        # Generate labels
        labeling_result = pipeline.generate_labels(data)
        logger.info(f"Generated {len(labeling_result.labels)} labels")

        # Train model
        model_results = pipeline.train_model(features, labeling_result)

        if model_results:
            # Analyze feature importance
            model_key = list(pipeline.fitted_models.keys())[-1]
            importance_analysis = pipeline.analyze_feature_importance(model_key)

            print("\n=== Feature Importance Analysis ===")
            print("Top 5 MDI Features:")
            print(importance_analysis['mdi_ranking'].head())
            print("\nTop 5 MDA Features:")
            print(importance_analysis['mda_ranking'].head())
            print(f"\nImportance Method Correlation: {importance_analysis['importance_correlation']:.3f}")
