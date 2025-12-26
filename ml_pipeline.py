"""
Module 2: Financial Machine Learning Pipeline
Implementing López de Prado's advanced methods for financial ML
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from statsmodels.tsa.stattools import adfuller
import sqlite3

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
        self.weights_ = None
    
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
            series = X[name].fillna(method='ffill').dropna()
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
                       step: float = 0.1) -> float:
        """Find minimum d value that achieves stationarity"""
        for d in np.arange(0, max_d + step, step):
            self.d = d
            self.fit(pd.DataFrame({'series': series}))
            diff_series = self.transform(pd.DataFrame({'series': series}))['series']
            
            if len(diff_series.dropna()) > 10:
                adf_stat, p_value, _, _, _, _ = adfuller(
                    diff_series.dropna(), maxlag=1, regression='c', autolag=None
                )
                if p_value < 0.05:  # Stationary
                    return d
        
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
    def __init__(self, pt_sl: List[float] = [1, 1], min_ret: float = 0.005):
        self.pt_sl = pt_sl  # [profit_taking_factor, stop_loss_factor]
        self.min_ret = min_ret
        self.vol_estimator = VolatilityEstimator()
    
    def get_events(self, close: pd.Series, t_events: pd.DatetimeIndex,
                   num_days: int = 1) -> pd.DataFrame:
        """
        Get triple-barrier events
        
        Args:
            close: Price series
            t_events: Event timestamps (from CUSUM filter or other trigger)
            num_days: Maximum holding period in days
        
        Returns:
            DataFrame with columns: t1 (vertical barrier), trgt (target threshold)
        """
        # Get target (daily volatility)
        trgt = self.vol_estimator.estimate_daily_vol(close)
        trgt = trgt.loc[t_events]
        trgt = trgt[trgt > self.min_ret]  # Filter by minimum target
        
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
                       num_threads: int = 1) -> pd.DataFrame:
        """Apply triple barriers to events"""
        events_ = events.copy()
        
        # Add side column (1 for long positions)
        events_['side'] = 1.0
        
        # Apply barriers
        barrier_results = []
        for idx, (loc, event) in enumerate(events_.iterrows()):
            result = self._apply_single_barrier(close, loc, event)
            barrier_results.append(result)
        
        barrier_df = pd.DataFrame(barrier_results, index=events_.index)
        events_['t1'] = barrier_df[['sl', 'pt']].fillna(
            events_['t1']
        ).min(axis=1)
        
        return events_
    
    def _apply_single_barrier(self, close: pd.Series, start_time: pd.Timestamp,
                             event: pd.Series) -> Dict:
        """Apply barriers to single event"""
        try:
            # Get price path
            end_time = event['t1']
            path = close[start_time:end_time]
            
            if len(path) < 2:
                return {'sl': None, 'pt': None}
            
            # Calculate returns
            returns = (path / close[start_time] - 1) * event['side']
            
            # Define barriers
            pt_barrier = self.pt_sl[0] * event['trgt']  # Profit taking
            sl_barrier = -self.pt_sl[1] * event['trgt']  # Stop loss
            
            # Find first barrier touch
            pt_touch = returns[returns > pt_barrier].index.min()
            sl_touch = returns[returns < sl_barrier].index.min()
            
            return {'sl': sl_touch, 'pt': pt_touch}
            
        except Exception as e:
            logger.warning(f"Error applying barriers: {e}")
            return {'sl': None, 'pt': None}
    
    def get_bins(self, events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Get labels from barrier touches"""
        events_ = events.dropna(subset=['t1'])
        
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
    
    def __init__(self):
        pass
    
    def get_sample_weights(self, events: pd.DataFrame, close: pd.Series,
                          num_co_events: Optional[pd.Series] = None) -> pd.Series:
        """
        Get sample weights considering uniqueness and return attribution
        
        Args:
            events: Events DataFrame with t1 column
            close: Price series
            num_co_events: Number of concurrent events (pre-computed)
        
        Returns:
            Sample weights series
        """
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
        
        # Normalize to sum to number of samples
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
            # Increment count for all times in the event's lifespan
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
            
            # Get uniqueness over event lifespan
            event_mask = (num_co_events.index >= start_time) & \
                        (num_co_events.index <= end_time)
            event_concurrent = num_co_events[event_mask]
            
            if len(event_concurrent) > 0:
                # Average uniqueness = average of 1/concurrent_events
                uniqueness = (1.0 / event_concurrent).mean()
                weights.loc[start_time] = uniqueness
        
        return weights.fillna(0)
    
    def _get_return_attribution_weights(self, events: pd.DataFrame, 
                                       close: pd.Series,
                                       num_co_events: pd.Series) -> pd.Series:
        """Weight by attributed returns"""
        ret = np.log(close).diff()  # Log returns
        weights = pd.Series(index=events.index, dtype=float)
        
        for start_time, row in events.iterrows():
            end_time = row['t1']
            if pd.isna(end_time):
                continue
            
            # Get returns over event period
            event_mask = (ret.index >= start_time) & (ret.index <= end_time)
            event_returns = ret[event_mask]
            event_concurrent = num_co_events[event_mask]
            
            if len(event_returns) > 0 and len(event_concurrent) > 0:
                # Attributed return = sum(return / concurrent_events)
                attributed_ret = (event_returns / event_concurrent).sum()
                weights.loc[start_time] = abs(attributed_ret)
        
        return weights.fillna(0)

class PurgedKFold:
    """
    Purged K-Fold cross-validation to prevent leakage
    Following de Prado Chapter 7, Section 7.4
    """
    
    def __init__(self, n_splits: int = 3, pct_embargo: float = 0.01):
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
    
    def split(self, X: pd.DataFrame, events: pd.DataFrame) -> List[Tuple]:
        """Generate train/test splits with purging and embargo"""
        if (X.index != events.index).any():
            raise ValueError("X and events must have the same index")
        
        indices = np.arange(X.shape[0])
        embargo_size = int(X.shape[0] * self.pct_embargo)
        
        # Create test splits
        test_splits = np.array_split(indices, self.n_splits)
        
        splits = []
        for test_indices in test_splits:
            # Get test times
            test_times = X.index[test_indices]
            min_test_time = test_times.min()
            max_test_time = test_times.max()
            
            # Purge training set
            train_indices = self._purge_train_set(
                events, indices, min_test_time, max_test_time
            )
            
            # Apply embargo
            if embargo_size > 0:
                max_train_idx = events.index.get_loc(max_test_time)
                embargo_start = max_train_idx + 1
                embargo_end = min(embargo_start + embargo_size, len(events))
                embargo_times = events.index[embargo_start:embargo_end]
                
                # Remove embargoed times from training
                train_indices = [
                    idx for idx in train_indices 
                    if X.index[idx] not in embargo_times
                ]
            
            splits.append((train_indices, test_indices.tolist()))
        
        return splits
    
    def _purge_train_set(self, events: pd.DataFrame, indices: np.ndarray,
                        min_test_time: pd.Timestamp, 
                        max_test_time: pd.Timestamp) -> List[int]:
        """Purge training set of overlapping observations"""
        train_indices = []
        
        for idx in indices:
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
    
    def __init__(self):
        pass
    
    def mean_decrease_impurity(self, clf, feature_names: List[str]) -> pd.DataFrame:
        """Calculate Mean Decrease Impurity (MDI) feature importance"""
        if not hasattr(clf, 'estimators_'):
            raise ValueError("Classifier must be a tree-based ensemble")
        
        # Get feature importances from each tree
        importances = {}
        for i, tree in enumerate(clf.estimators_):
            importances[i] = tree.feature_importances_
        
        imp_df = pd.DataFrame.from_dict(importances, orient='index')
        imp_df.columns = feature_names
        imp_df = imp_df.replace(0, np.nan)  # Replace zeros with NaN
        
        # Calculate mean and std
        result = pd.DataFrame({
            'mean': imp_df.mean(),
            'std': imp_df.std() * imp_df.shape[0] ** -0.5
        })
        
        # Normalize so importances sum to 1
        result /= result['mean'].sum()
        
        return result
    
    def mean_decrease_accuracy(self, clf, X: pd.DataFrame, y: pd.Series,
                              sample_weight: pd.Series, cv_gen,
                              scoring: str = 'neg_log_loss') -> Tuple[pd.DataFrame, float]:
        """Calculate Mean Decrease Accuracy (MDA) feature importance"""
        
        # Get baseline scores
        base_scores = []
        permuted_scores = pd.DataFrame(columns=X.columns)
        
        for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X, None)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train, w_test = sample_weight.iloc[train_idx], sample_weight.iloc[test_idx]
            
            # Fit classifier
            clf_fitted = clf.fit(X_train, y_train, sample_weight=w_train.values)
            
            # Get baseline score
            if scoring == 'neg_log_loss':
                prob = clf_fitted.predict_proba(X_test)
                base_score = -log_loss(y_test, prob, sample_weight=w_test.values)
            else:  # accuracy
                pred = clf_fitted.predict(X_test)
                base_score = accuracy_score(y_test, pred, sample_weight=w_test.values)
            
            base_scores.append(base_score)
            
            # Permute each feature and calculate score
            for feature in X.columns:
                X_test_perm = X_test.copy()
                np.random.shuffle(X_test_perm[feature].values)
                
                if scoring == 'neg_log_loss':
                    prob_perm = clf_fitted.predict_proba(X_test_perm)
                    perm_score = -log_loss(y_test, prob_perm, sample_weight=w_test.values)
                else:
                    pred_perm = clf_fitted.predict(X_test_perm)
                    perm_score = accuracy_score(y_test, pred_perm, sample_weight=w_test.values)
                
                permuted_scores.loc[fold, feature] = perm_score
        
        # Calculate importance as performance difference
        base_score_mean = np.mean(base_scores)
        importance = pd.DataFrame(index=X.columns)
        
        for feature in X.columns:
            perm_scores = permuted_scores[feature].values
            diff = base_score_mean - perm_scores.mean()
            
            if scoring == 'neg_log_loss':
                importance.loc[feature, 'mean'] = diff / abs(base_score_mean) if base_score_mean != 0 else 0
            else:
                importance.loc[feature, 'mean'] = diff / (1 - perm_scores.mean()) if perm_scores.mean() != 1 else 0
            
            importance.loc[feature, 'std'] = perm_scores.std() * len(perm_scores) ** -0.5
        
        return importance, base_score_mean

class FinancialMLPipeline:
    """
    Main ML Pipeline integrating all de Prado methods
    """
    
    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = db_path
        
        # Components
        self.frac_diff = FractionalDifferentiator()
        self.labeler = TripleBarrierLabeler()
        self.weighter = SampleWeighter()
        self.cv = PurgedKFold()
        self.feat_analyzer = FeatureImportanceAnalyzer()
        
        # Storage for fitted components
        self.fitted_models = {}
        self.feature_importance_results = {}
        
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
        params = [symbol, bar_type]
        
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
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from bar data"""
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
        features['vwap_ratio'] = data['close'] / data['vwap']
        features['price_range'] = (data['high'] - data['low']) / data['close']
        
        # Apply fractional differentiation to price-based features
        price_features = ['returns', 'log_returns', 'rsi', 'bollinger_position']
        
        for feature in price_features:
            if feature in features.columns:
                # Find optimal d for stationarity
                optimal_d = self.frac_diff.find_optimal_d(features[feature].dropna())
                
                # Apply fractional differentiation
                self.frac_diff.d = optimal_d
                frac_diff_feature = self.frac_diff.fit_transform(
                    features[[feature]]
                )[feature]
                
                features[f'{feature}_fracdiff'] = frac_diff_feature
        
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return (prices - lower) / (upper - lower)
    
    def generate_labels(self, data: pd.DataFrame, 
                       event_filter_threshold: float = 0.01) -> LabelingResult:
        """Generate labels using triple-barrier method"""
        close = data['close']
        
        # Generate events using CUSUM filter or simple sampling
        # For simplicity, using regular sampling here
        # In practice, use CUSUM filter from Module 1
        event_frequency = pd.Timedelta(hours=4)  # Event every 4 hours
        t_events = pd.date_range(
            start=close.index[0], 
            end=close.index[-1], 
            freq=event_frequency
        )
        t_events = t_events[t_events.isin(close.index)]
        
        # Get events DataFrame
        events = self.labeler.get_events(close, t_events)
        
        # Apply barriers
        events_with_barriers = self.labeler.apply_barriers(close, events)
        
        # Generate labels
        labels_df = self.labeler.get_bins(events_with_barriers, close)
        
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
        X = features.loc[common_index]
        y = labeling_result.labels.loc[common_index]
        sample_weights = labeling_result.sample_weights.loc[common_index]
        events = labeling_result.events.loc[common_index]
        
        # Set up classifier
        if model_params is None:
            model_params = {
                'n_estimators': 1000,
                'max_features': 1,  # Prevent masking effects
                'class_weight': 'balanced',
                'random_state': 42
            }
        
        base_clf = DecisionTreeClassifier(
            criterion='entropy',
            max_features=1,
            class_weight='balanced'
        )
        
        clf = BaggingClassifier(
            base_estimator=base_clf,
            n_estimators=model_params['n_estimators'],
            max_samples=sample_weights.mean(),  # Control for redundancy
            max_features=1.0,
            oob_score=True,
            n_jobs=-1
        )
        
        # Set up cross-validation with events
        events_for_cv = pd.DataFrame({'t1': events['t1']}, index=events.index)
        cv_gen = PurgedKFold(n_splits=5)
        
        # Calculate cross-validated scores
        cv_scores = []
        for train_idx, test_idx in cv_gen.split(X, events_for_cv):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train, w_test = sample_weights.iloc[train_idx], sample_weights.iloc[test_idx]
            
            clf_fold = BaggingClassifier(
                base_estimator=base_clf,
                n_estimators=100,  # Smaller for CV
                max_samples=w_train.mean(),
                max_features=1.0,
                n_jobs=-1
            )
            
            clf_fold.fit(X_train, y_train, sample_weight=w_train.values)
            
            # Score
            if set(y.unique()) == {0, 1}:  # Meta-labeling
                from sklearn.metrics import f1_score
                pred = clf_fold.predict(X_test)
                score = f1_score(y_test, pred, sample_weight=w_test.values, average='weighted')
            else:
                prob = clf_fold.predict_proba(X_test)
                score = -log_loss(y_test, prob, sample_weight=w_test.values)
            
            cv_scores.append(score)
        
        # Fit final model on all data
        clf.fit(X, y, sample_weight=sample_weights.values)
        
        # Calculate feature importance
        mdi_importance = self.feat_analyzer.mean_decrease_impurity(clf, X.columns)
        mda_importance, base_score = self.feat_analyzer.mean_decrease_accuracy(
            clf, X, y, sample_weights, cv_gen
        )
        
        # Store results
        model_key = f"model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = {
            'model': clf,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'oob_score': clf.oob_score_,
            'feature_importance_mdi': mdi_importance,
            'feature_importance_mda': mda_importance,
            'base_score': base_score,
            'features': X.columns.tolist(),
            'n_samples': len(X)
        }
        
        self.fitted_models[model_key] = results
        
        # Log results following de Prado's emphasis on feature importance
        logger.info(f"Model trained successfully: {model_key}")
        logger.info(f"CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        logger.info("Top 5 Important Features (MDI):")
        top_features = mdi_importance.sort_values('mean', ascending=False).head()
        for feature, row in top_features.iterrows():
            logger.info(f"  {feature}: {row['mean']:.4f}")
        
        return results
    
    def predict(self, features: pd.DataFrame, model_key: str) -> pd.DataFrame:
        """Make predictions using trained model"""
        if model_key not in self.fitted_models:
            raise ValueError(f"Model {model_key} not found")
        
        model_info = self.fitted_models[model_key]
        clf = model_info['model']
        
        # Ensure features match training features
        model_features = model_info['features']
        X = features[model_features]
        
        # Make predictions
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)
        
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
        if len(common_features) > 1:
            from scipy.stats import spearmanr
            correlation = spearmanr(
                mdi_ranking.loc[common_features, 'mean'],
                mda_ranking.loc[common_features, 'mean']
            )
            importance_correlation = correlation.correlation
        else:
            importance_correlation = np.nan
        
        return {
            'mdi_ranking': mdi_ranking,
            'mda_ranking': mda_ranking,
            'importance_correlation': importance_correlation,
            'top_mdi_features': mdi_ranking.head().index.tolist(),
            'top_mda_features': mda_ranking.head().index.tolist()
        }

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = FinancialMLPipeline()
    
    # Load data
    data = pipeline.load_data("BTC/USD", "dollar")
    
    if not data.empty:
        logger.info(f"Loaded {len(data)} bars of data")
        
        # Create features
        features = pipeline.create_features(data)
        logger.info(f"Created {len(features.columns)} features")
        
        # Generate labels
        labeling_result = pipeline.generate_labels(data)
        logger.info(f"Generated {len(labeling_result.labels)} labels")
        
        # Train model
        model_results = pipeline.train_model(features, labeling_result)
        
        # Analyze feature importance
        model_key = list(pipeline.fitted_models.keys())[-1]
        importance_analysis = pipeline.analyze_feature_importance(model_key)
        
        print("=== Feature Importance Analysis ===")
        print("Top 5 MDI Features:")
        print(importance_analysis['mdi_ranking'].head())
        print("\nTop 5 MDA Features:")
        print(importance_analysis['mda_ranking'].head())
        print(f"\nImportance Method Correlation: {importance_analysis['importance_correlation']:.3f}")
