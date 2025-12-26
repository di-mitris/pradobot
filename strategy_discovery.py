"""
Module 3: Strategy Discovery Engine
Implementing López de Prado's approach to systematic strategy discovery
Buy-only strategies with proper risk management
Optimized for Hetzner Cloud (4 vCPU, 8GB RAM)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import yaml
import sqlite3
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import defaultdict
import json
import joblib

# Import our pipeline modules
from ml_pipeline import FinancialMLPipeline, LabelingResult

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    HYBRID = "hybrid"


class SignalType(Enum):
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"  # Exit position only (buy-only constraint)


@dataclass
class TradingSignal:
    """Trading signal with confidence and metadata"""
    timestamp: pd.Timestamp
    symbol: str
    signal_type: SignalType
    confidence: float
    features_used: List[str]
    feature_values: Dict[str, float]
    model_prediction: float
    model_probability: float
    strategy_type: StrategyType
    risk_score: float
    position_size: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'features_used': self.features_used,
            'feature_values': self.feature_values,
            'model_prediction': float(self.model_prediction),
            'model_probability': self.model_probability,
            'strategy_type': self.strategy_type.value,
            'risk_score': self.risk_score,
            'position_size': self.position_size,
            'metadata': self.metadata
        }


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    name: str
    strategy_type: StrategyType
    total_signals: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return_per_trade: float
    feature_importance_scores: Dict[str, float]
    cv_scores: List[float]
    created_timestamp: pd.Timestamp

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'strategy_type': self.strategy_type.value,
            'total_signals': self.total_signals,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'avg_return_per_trade': self.avg_return_per_trade,
            'feature_importance_scores': self.feature_importance_scores,
            'cv_scores': self.cv_scores,
            'created_timestamp': self.created_timestamp.isoformat()
        }


class BaseStrategy(ABC):
    """Abstract base class for all strategies"""

    def __init__(self, name: str, strategy_type: StrategyType, config: Dict):
        self.name = name
        self.strategy_type = strategy_type
        self.config = config
        self.is_trained = False
        self.feature_names: List[str] = []
        self.model = None
        self.scaler = None
        self.last_signal_time: Dict[str, pd.Timestamp] = {}
        
        # Performance tracking
        self.historical_win_rate = 0.55
        self.historical_avg_win = 0.025
        self.historical_avg_loss = -0.015

    @abstractmethod
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate strategy-specific features"""
        pass

    @abstractmethod
    def generate_signal(self, features: pd.Series, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal from features"""
        pass

    def can_generate_signal(self, symbol: str, current_time: pd.Timestamp) -> bool:
        """Check if enough time has passed since last signal"""
        if symbol not in self.last_signal_time:
            return True

        cooldown = timedelta(hours=self.config.get('cooldown_hours', 1))
        return current_time - self.last_signal_time[symbol] >= cooldown

    def calculate_position_size(self, confidence: float) -> float:
        """Calculate position size using Kelly criterion"""
        win_rate = self.historical_win_rate
        avg_win = self.historical_avg_win
        avg_loss = abs(self.historical_avg_loss)

        if avg_loss > 0 and avg_win > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.05

        # Scale by confidence and apply maximum position limit
        max_position = self.config.get('max_position_size', 0.1)
        return min(kelly_fraction * confidence, max_position)

    def update_performance_stats(self, win_rate: float, avg_win: float, avg_loss: float):
        """Update historical performance statistics"""
        # EWMA update
        alpha = 0.1
        self.historical_win_rate = (1 - alpha) * self.historical_win_rate + alpha * win_rate
        self.historical_avg_win = (1 - alpha) * self.historical_avg_win + alpha * avg_win
        self.historical_avg_loss = (1 - alpha) * self.historical_avg_loss + alpha * avg_loss


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using trend-following features"""

    def __init__(self, config: Dict):
        super().__init__("momentum", StrategyType.MOMENTUM, config)
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20])

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-specific features"""
        features = pd.DataFrame(index=data.index)

        # Price momentum features
        for period in self.lookback_periods:
            features[f'return_{period}'] = data['close'].pct_change(period)
            features[f'volume_momentum_{period}'] = (
                data['volume'] / data['volume'].rolling(period).mean() - 1
            )

        # Moving average features
        features['ma_5_20_ratio'] = data['close'].rolling(5).mean() / data['close'].rolling(20).mean()
        features['ma_10_50_ratio'] = data['close'].rolling(10).mean() / data['close'].rolling(50).mean()

        # Volume-price trend
        features['vpt'] = ((data['close'].pct_change() * data['volume']).cumsum())
        features['vpt_normalized'] = features['vpt'] / features['vpt'].rolling(20).std()

        # Rate of change
        features['roc_5'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
        features['roc_10'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)

        # Trend strength
        features['adx'] = self._calculate_adx(data, 14)

        return features

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high, low, close = data['high'], data['low'], data['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.inf)
        adx = dx.rolling(period).mean()
        
        return adx

    def generate_signal(self, features: pd.Series, symbol: str) -> Optional[TradingSignal]:
        """Generate momentum signal"""
        if not self.is_trained or self.model is None:
            return None

        current_time = features.name
        if not self.can_generate_signal(symbol, current_time):
            return None

        # Prepare feature vector
        available_features = [f for f in self.feature_names if f in features.index]
        if len(available_features) < len(self.feature_names) * 0.8:
            return None

        feature_vector = features[available_features].values.reshape(1, -1)

        if np.any(np.isnan(feature_vector)):
            return None

        # Scale features
        if self.scaler is not None:
            try:
                feature_vector = self.scaler.transform(feature_vector)
            except Exception:
                pass

        # Get prediction
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        confidence = max(probabilities)

        # Check confidence threshold
        min_confidence = self.config.get('confidence_threshold', 0.6)
        if confidence < min_confidence:
            return None

        # Only generate buy signals (buy-only constraint)
        if prediction != 1:
            return None

        # Calculate position size
        position_size = self.calculate_position_size(confidence)

        signal = TradingSignal(
            timestamp=current_time,
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=confidence,
            features_used=available_features,
            feature_values=dict(zip(available_features, feature_vector[0])),
            model_prediction=prediction,
            model_probability=confidence,
            strategy_type=self.strategy_type,
            risk_score=1.0 - confidence,
            position_size=position_size,
            metadata={
                'kelly_fraction': position_size / confidence if confidence > 0 else 0,
                'win_rate': self.historical_win_rate,
                'avg_win': self.historical_avg_win,
                'avg_loss': self.historical_avg_loss
            }
        )

        self.last_signal_time[symbol] = current_time
        return signal


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy for oversold/overbought conditions"""

    def __init__(self, config: Dict):
        super().__init__("mean_reversion", StrategyType.MEAN_REVERSION, config)
        self.bollinger_window = config.get('bollinger_window', 20)
        self.rsi_window = config.get('rsi_window', 14)

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion features"""
        features = pd.DataFrame(index=data.index)

        # Bollinger Bands
        sma = data['close'].rolling(self.bollinger_window).mean()
        std = data['close'].rolling(self.bollinger_window).std()
        features['bb_position'] = (data['close'] - sma) / (2 * std).replace(0, np.inf)
        features['bb_width'] = (2 * std) / sma.replace(0, np.inf)

        # RSI
        features['rsi'] = self._calculate_rsi(data['close'])
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)

        # Price distance from moving averages
        for window in [10, 20, 50]:
            ma = data['close'].rolling(window).mean()
            features[f'price_ma{window}_ratio'] = data['close'] / ma.replace(0, np.inf)

        # Volume patterns
        features['volume_spike'] = (
            data['volume'] / data['volume'].rolling(20).mean().replace(0, np.inf)
        )

        # Volatility regime
        features['volatility'] = data['close'].pct_change().rolling(20).std()
        vol_mean = features['volatility'].rolling(100).mean()
        features['vol_regime'] = (features['volatility'] > vol_mean).astype(int)

        # Z-score of price
        features['price_zscore'] = (data['close'] - sma) / std.replace(0, np.inf)

        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = None) -> pd.Series:
        """Calculate RSI"""
        if window is None:
            window = self.rsi_window

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))

    def generate_signal(self, features: pd.Series, symbol: str) -> Optional[TradingSignal]:
        """Generate mean reversion signal"""
        if not self.is_trained or self.model is None:
            return None

        current_time = features.name
        if not self.can_generate_signal(symbol, current_time):
            return None

        # Prepare feature vector
        available_features = [f for f in self.feature_names if f in features.index]
        if len(available_features) < len(self.feature_names) * 0.8:
            return None

        feature_vector = features[available_features].values.reshape(1, -1)

        if np.any(np.isnan(feature_vector)):
            return None

        # Scale features
        if self.scaler is not None:
            try:
                feature_vector = self.scaler.transform(feature_vector)
            except Exception:
                pass

        # Get prediction
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        confidence = max(probabilities)

        # Check confidence threshold
        min_confidence = self.config.get('confidence_threshold', 0.6)
        if confidence < min_confidence:
            return None

        # Only generate buy signals
        if prediction != 1:
            return None

        # Additional mean reversion conditions
        rsi = features.get('rsi', 50)
        bb_position = features.get('bb_position', 0)

        # Only buy when oversold
        if rsi > 40 or bb_position > -0.5:
            return None

        # Conservative position sizing for mean reversion
        position_size = min(0.05, confidence * 0.1)

        signal = TradingSignal(
            timestamp=current_time,
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=confidence,
            features_used=available_features,
            feature_values=dict(zip(available_features, feature_vector[0])),
            model_prediction=prediction,
            model_probability=confidence,
            strategy_type=self.strategy_type,
            risk_score=1.0 - confidence,
            position_size=position_size,
            metadata={
                'rsi': float(rsi),
                'bb_position': float(bb_position),
                'strategy_logic': 'oversold_condition'
            }
        )

        self.last_signal_time[symbol] = current_time
        return signal


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy for trend continuation after consolidation"""

    def __init__(self, config: Dict):
        super().__init__("breakout", StrategyType.BREAKOUT, config)
        self.volatility_window = config.get('volatility_window', 20)

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout-specific features"""
        features = pd.DataFrame(index=data.index)

        # Volatility features
        returns = data['close'].pct_change()
        features['volatility'] = returns.rolling(self.volatility_window).std()
        vol_mean = features['volatility'].rolling(100).mean()
        features['vol_regime'] = features['volatility'] / vol_mean.replace(0, np.inf)

        # True Range and ATR
        features['true_range'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        features['atr'] = features['true_range'].rolling(14).mean()
        features['atr_ratio'] = features['true_range'] / features['atr'].replace(0, np.inf)

        # Support/Resistance levels
        for window in [10, 20, 50]:
            features[f'high_{window}'] = data['high'].rolling(window).max()
            features[f'low_{window}'] = data['low'].rolling(window).min()
            features[f'resistance_break_{window}'] = (
                data['close'] > features[f'high_{window}'].shift(1)
            ).astype(int)
            features[f'support_break_{window}'] = (
                data['close'] < features[f'low_{window}'].shift(1)
            ).astype(int)

        # Volume confirmation
        features['volume_surge'] = (
            data['volume'] / data['volume'].rolling(10).mean().replace(0, np.inf)
        )

        # Consolidation detection
        vol_quantile = features['volatility'].rolling(50).quantile(0.3)
        features['consolidation'] = (features['volatility'] < vol_quantile).astype(int)

        # Breakout strength
        range_20 = features['high_20'] - features['low_20']
        features['breakout_strength'] = (data['close'] - features['low_20']) / range_20.replace(0, np.inf)

        return features

    def generate_signal(self, features: pd.Series, symbol: str) -> Optional[TradingSignal]:
        """Generate breakout signal"""
        if not self.is_trained or self.model is None:
            return None

        current_time = features.name
        if not self.can_generate_signal(symbol, current_time):
            return None

        # Prepare feature vector
        available_features = [f for f in self.feature_names if f in features.index]
        if len(available_features) < len(self.feature_names) * 0.8:
            return None

        feature_vector = features[available_features].values.reshape(1, -1)

        if np.any(np.isnan(feature_vector)):
            return None

        # Scale features
        if self.scaler is not None:
            try:
                feature_vector = self.scaler.transform(feature_vector)
            except Exception:
                pass

        # Get prediction
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        confidence = max(probabilities)

        # Check confidence threshold
        min_confidence = self.config.get('confidence_threshold', 0.6)
        if confidence < min_confidence:
            return None

        # Only generate buy signals
        if prediction != 1:
            return None

        # Additional breakout conditions
        resistance_break = features.get('resistance_break_20', 0)
        volume_surge = features.get('volume_surge', 1)

        # Require resistance break with volume confirmation
        if resistance_break != 1 or volume_surge < 1.5:
            return None

        position_size = self.calculate_position_size(confidence)

        signal = TradingSignal(
            timestamp=current_time,
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=confidence,
            features_used=available_features,
            feature_values=dict(zip(available_features, feature_vector[0])),
            model_prediction=prediction,
            model_probability=confidence,
            strategy_type=self.strategy_type,
            risk_score=1.0 - confidence,
            position_size=position_size,
            metadata={
                'resistance_break': int(resistance_break),
                'volume_surge': float(volume_surge),
                'strategy_logic': 'breakout_with_volume'
            }
        )

        self.last_signal_time[symbol] = current_time
        return signal


class StrategyDiscoveryEngine:
    """
    Main engine for discovering and managing trading strategies
    Following de Prado's scientific approach
    """

    def __init__(self, config_path: str = None, config: Dict = None):
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        # Initialize paths
        base_dir = Path(self.config.get('paths', {}).get('base_dir', '/opt/financial_ml'))
        self.db_path = self.config.get('database', {}).get('sqlite', {}).get(
            'path', str(base_dir / 'data' / 'financial_data.db')
        )
        self.models_dir = base_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ML pipeline
        self.ml_pipeline = FinancialMLPipeline(
            db_path=self.db_path,
            models_dir=str(self.models_dir)
        )

        # Initialize database
        self._initialize_database()

        # Strategy storage
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.recent_signals: List[TradingSignal] = []
        self.signal_history: Dict[str, List[TradingSignal]] = defaultdict(list)

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'paths': {
                'base_dir': '/opt/financial_ml',
                'data_dir': '/opt/financial_ml/data',
                'models_dir': '/opt/financial_ml/models'
            },
            'database': {
                'sqlite': {
                    'path': '/opt/financial_ml/data/financial_data.db'
                }
            },
            'data_ingestion': {
                'bars': {
                    'algorithmic': {
                        'dollar': 50000
                    }
                }
            },
            'strategy_discovery': {
                'minimum_requirements': {
                    'min_samples': 500,
                    'min_cv_score': 0.55
                },
                'strategy_types': ['momentum', 'mean_reversion', 'breakout'],
                'confidence_threshold': 0.6,
                'cooldown_hours': 1,
                'max_position_size': 0.1
            }
        }

    def _initialize_database(self):
        """Initialize database tables for strategy management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT UNIQUE NOT NULL,
                strategy_type TEXT NOT NULL,
                config_json TEXT,
                feature_names_json TEXT,
                created_timestamp TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                accuracy REAL,
                is_active INTEGER DEFAULT 1
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_prediction REAL,
                position_size REAL,
                risk_score REAL,
                features_json TEXT,
                metadata_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                pnl REAL NOT NULL,
                return_pct REAL NOT NULL,
                position_size REAL NOT NULL,
                hold_time_hours REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp 
            ON strategy_signals(symbol, timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_performance_strategy 
            ON strategy_performance(strategy_name, timestamp)
        ''')

        conn.commit()
        conn.close()

    def discover_strategies(self, symbols: List[str],
                            timeframe: str = 'algorithmic') -> Dict[str, StrategyMetrics]:
        """
        Discover trading strategies using feature importance analysis
        """
        logger.info(f"Starting strategy discovery for symbols: {symbols}")

        discovered_strategies = {}
        strategy_config = self.config.get('strategy_discovery', {})

        for symbol in symbols:
            logger.info(f"Discovering strategies for {symbol}")

            # Load data
            data = self.ml_pipeline.load_data(
                symbol,
                'dollar',
                start_date=(datetime.now() - timedelta(days=90)).isoformat()
            )

            min_samples = strategy_config.get('minimum_requirements', {}).get('min_samples', 500)
            if len(data) < min_samples:
                logger.warning(f"Insufficient data for {symbol}: {len(data)} bars")
                continue

            # Generate features
            features = self.ml_pipeline.create_features(data)

            # Generate labels
            labeling_result = self.ml_pipeline.generate_labels(data)

            if labeling_result.labels.empty:
                logger.warning(f"No labels generated for {symbol}")
                continue

            # Train models for different strategy types
            strategy_types = strategy_config.get('strategy_types', ['momentum', 'mean_reversion', 'breakout'])
            
            for strategy_type in strategy_types:
                logger.info(f"Training {strategy_type} strategy for {symbol}")

                try:
                    # Create strategy-specific features
                    strategy_features = self._create_strategy_features(data, strategy_type)

                    # Combine with general features
                    combined_features = pd.concat([features, strategy_features], axis=1)
                    combined_features = combined_features.dropna()

                    if len(combined_features) < min_samples:
                        logger.warning(f"Insufficient features for {strategy_type}/{symbol}")
                        continue

                    # Train model
                    model_results = self.ml_pipeline.train_model(
                        combined_features, labeling_result
                    )

                    if not model_results:
                        continue

                    # Check minimum performance requirements
                    cv_score = model_results['cv_mean']
                    min_cv_score = strategy_config.get('minimum_requirements', {}).get('min_cv_score', 0.55)

                    if cv_score < min_cv_score:
                        logger.info(f"Strategy {strategy_type} for {symbol} below minimum CV score: {cv_score:.3f}")
                        continue

                    # Analyze feature importance
                    model_key = list(self.ml_pipeline.fitted_models.keys())[-1]
                    importance_analysis = self.ml_pipeline.analyze_feature_importance(model_key)

                    # Create strategy
                    strategy = self._create_strategy(
                        strategy_type, symbol, model_results, importance_analysis, strategy_config
                    )

                    if strategy:
                        strategy_name = f"{strategy_type}_{symbol}"
                        self.strategies[strategy_name] = strategy

                        # Create metrics
                        metrics = StrategyMetrics(
                            name=strategy_name,
                            strategy_type=StrategyType(strategy_type),
                            total_signals=0,
                            accuracy=cv_score,
                            precision=0.0,
                            recall=0.0,
                            f1_score=0.0,
                            sharpe_ratio=0.0,
                            max_drawdown=0.0,
                            win_rate=0.0,
                            avg_return_per_trade=0.0,
                            feature_importance_scores=dict(
                                importance_analysis['mdi_ranking']['mean'].head(10)
                            ),
                            cv_scores=model_results['cv_scores'],
                            created_timestamp=pd.Timestamp.now()
                        )

                        self.strategy_metrics[strategy_name] = metrics
                        discovered_strategies[strategy_name] = metrics

                        # Save strategy
                        self._save_strategy_metadata(strategy, metrics)

                        logger.info(f"Created strategy {strategy_name} with CV score: {cv_score:.3f}")

                except Exception as e:
                    logger.error(f"Error creating strategy {strategy_type} for {symbol}: {e}")
                    continue

        logger.info(f"Strategy discovery complete. Created {len(discovered_strategies)} strategies")
        return discovered_strategies

    def _create_strategy_features(self, data: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
        """Create strategy-specific features"""
        config = self.config.get('strategy_discovery', {})
        
        if strategy_type == 'momentum':
            strategy = MomentumStrategy(config)
        elif strategy_type == 'mean_reversion':
            strategy = MeanReversionStrategy(config)
        elif strategy_type == 'breakout':
            strategy = BreakoutStrategy(config)
        else:
            strategy = MomentumStrategy(config)

        return strategy.generate_features(data)

    def _create_strategy(self, strategy_type: str, symbol: str,
                         model_results: Dict, importance_analysis: Dict,
                         config: Dict) -> Optional[BaseStrategy]:
        """Create strategy instance from model results"""
        try:
            # Get top features
            top_features = importance_analysis['top_mdi_features'][:10]

            # Create strategy instance
            if strategy_type == 'momentum':
                strategy = MomentumStrategy(config)
            elif strategy_type == 'mean_reversion':
                strategy = MeanReversionStrategy(config)
            elif strategy_type == 'breakout':
                strategy = BreakoutStrategy(config)
            else:
                return None

            # Set strategy attributes
            strategy.model = model_results['model']
            strategy.scaler = model_results.get('scaler')
            strategy.feature_names = top_features
            strategy.is_trained = True

            return strategy

        except Exception as e:
            logger.error(f"Error creating strategy: {e}")
            return None

    def _save_strategy_metadata(self, strategy: BaseStrategy, metrics: StrategyMetrics):
        """Save strategy metadata to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO strategy_metadata 
            (strategy_name, strategy_type, config_json, feature_names_json, 
             created_timestamp, last_updated, accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.name,
            metrics.strategy_type.value,
            json.dumps(strategy.config),
            json.dumps(strategy.feature_names),
            metrics.created_timestamp.isoformat(),
            pd.Timestamp.now().isoformat(),
            metrics.accuracy
        ))

        conn.commit()
        conn.close()

    def generate_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate trading signals from all strategies"""
        new_signals = []

        for symbol in symbols:
            # Load latest data
            data = self.ml_pipeline.load_data(
                symbol, 'dollar',
                start_date=(datetime.now() - timedelta(hours=24)).isoformat()
            )

            if data.empty:
                continue

            # Get features
            features = self.ml_pipeline.create_features(data)

            if features.empty:
                continue

            latest_features = features.iloc[-1]

            # Generate signals from all strategies for this symbol
            for strategy_name, strategy in self.strategies.items():
                if symbol not in strategy_name:
                    continue

                try:
                    # Generate strategy-specific features
                    strategy_data = self._create_strategy_features(
                        data, strategy.strategy_type.value
                    )

                    if strategy_data.empty:
                        continue

                    # Combine features
                    combined = pd.concat([latest_features, strategy_data.iloc[-1]])

                    # Generate signal
                    signal = strategy.generate_signal(combined, symbol)

                    if signal:
                        new_signals.append(signal)
                        self.recent_signals.append(signal)
                        self.signal_history[symbol].append(signal)

                        # Store signal
                        self._store_signal(signal)

                        logger.info(f"Generated {signal.signal_type.value} signal for {symbol} "
                                    f"from {strategy_name} with confidence {signal.confidence:.3f}")

                except Exception as e:
                    logger.error(f"Error generating signal from {strategy_name}: {e}")

        # Clean up old signals
        if len(self.recent_signals) > 1000:
            self.recent_signals = self.recent_signals[-1000:]

        return new_signals

    def _store_signal(self, signal: TradingSignal):
        """Store trading signal in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO strategy_signals 
            (timestamp, symbol, strategy_name, strategy_type, signal_type, 
             confidence, model_prediction, position_size, risk_score, 
             features_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp.isoformat(),
            signal.symbol,
            f"{signal.strategy_type.value}_{signal.symbol}",
            signal.strategy_type.value,
            signal.signal_type.value,
            signal.confidence,
            signal.model_prediction,
            signal.position_size,
            signal.risk_score,
            json.dumps(signal.feature_values),
            json.dumps(signal.metadata)
        ))

        conn.commit()
        conn.close()

    def get_strategy_summary(self) -> pd.DataFrame:
        """Get summary of all discovered strategies"""
        summary_data = []

        for name, metrics in self.strategy_metrics.items():
            summary_data.append({
                'strategy_name': metrics.name,
                'strategy_type': metrics.strategy_type.value,
                'accuracy': metrics.accuracy,
                'total_signals': metrics.total_signals,
                'sharpe_ratio': metrics.sharpe_ratio,
                'win_rate': metrics.win_rate,
                'created_timestamp': metrics.created_timestamp,
                'top_features': list(metrics.feature_importance_scores.keys())[:5]
            })

        return pd.DataFrame(summary_data)

    def get_recent_signals(self, hours: int = 24) -> List[TradingSignal]:
        """Get recent signals within specified hours"""
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        return [s for s in self.recent_signals if s.timestamp >= cutoff_time]


class PerformanceTracker:
    """Track strategy performance for continuous improvement"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def record_trade_result(self, signal: TradingSignal, actual_return: float,
                            hold_time_hours: float):
        """Record actual trade result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        pnl = actual_return * signal.position_size

        cursor.execute('''
            INSERT INTO strategy_performance 
            (strategy_name, symbol, timestamp, pnl, return_pct, position_size, hold_time_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"{signal.strategy_type.value}_{signal.symbol}",
            signal.symbol,
            signal.timestamp.isoformat(),
            pnl,
            actual_return,
            signal.position_size,
            hold_time_hours
        ))

        conn.commit()
        conn.close()

    def calculate_strategy_metrics(self, strategy_name: str) -> Dict:
        """Calculate updated strategy performance metrics"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query('''
            SELECT * FROM strategy_performance 
            WHERE strategy_name = ?
            ORDER BY timestamp
        ''', conn, params=[strategy_name])

        conn.close()

        if df.empty:
            return {}

        # Calculate metrics
        total_trades = len(df)
        win_rate = (df['return_pct'] > 0).mean()
        avg_return = df['return_pct'].mean()
        
        std_return = df['return_pct'].std()
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(df['return_pct'])

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()


# CLI Interface
def main():
    """Main CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Strategy Discovery Engine CLI')
    parser.add_argument('command', choices=['discover', 'signals', 'summary'],
                        help='Command to execute')
    parser.add_argument('--symbols', default='BTC/USD,ETH/USD',
                        help='Comma-separated symbols')
    parser.add_argument('--timeframe', default='algorithmic',
                        help='Timeframe: algorithmic or swing')
    parser.add_argument('--config', default='/opt/financial_ml/config.yaml',
                        help='Path to configuration file')

    args = parser.parse_args()

    # Initialize engine
    engine = StrategyDiscoveryEngine(config_path=args.config)
    symbol_list = args.symbols.split(',')

    if args.command == 'discover':
        strategies = engine.discover_strategies(symbol_list, args.timeframe)
        print(f"\nDiscovered {len(strategies)} strategies:")
        for name, metrics in strategies.items():
            print(f"  {name}: CV Score = {metrics.accuracy:.3f}")

    elif args.command == 'signals':
        signals = engine.generate_signals(symbol_list)
        print(f"\nGenerated {len(signals)} signals:")
        for signal in signals:
            print(f"  {signal.timestamp}: {signal.signal_type.value} {signal.symbol} "
                  f"(confidence: {signal.confidence:.3f}, size: {signal.position_size:.3f})")

    elif args.command == 'summary':
        summary = engine.get_strategy_summary()
        if not summary.empty:
            print("\nStrategy Summary:")
            print(summary.to_string(index=False))
        else:
            print("No strategies found. Run 'discover' first.")


if __name__ == "__main__":
    main()
