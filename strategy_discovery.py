"""
Module 3: Strategy Discovery Engine
Implementing López de Prado's approach to systematic strategy discovery
Buy-only strategies with proper risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
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
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict

# Import our previous modules
import sys
sys.path.append('/opt/financial_ml')
from data_ingestion import BarData
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
    SELL = "sell"  # Exit position (buy-only constraint)

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
    
class BaseStrategy(ABC):
    """Abstract base class for all strategies"""
    
    def __init__(self, name: str, strategy_type: StrategyType, config: Dict):
        self.name = name
        self.strategy_type = strategy_type
        self.config = config
        self.is_trained = False
        self.feature_names = []
        self.model = None
        self.scaler = None
        self.last_signal_time = {}  # Per symbol
        
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

class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using trend-following features
    Buys when upward momentum is detected
    """
    
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
        
        # Rate of change
        features['roc_5'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
        features['roc_10'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
        
        return features
    
    def generate_signal(self, features: pd.Series, symbol: str) -> Optional[TradingSignal]:
        """Generate momentum signal"""
        if not self.is_trained or self.model is None:
            return None
        
        current_time = features.name
        if not self.can_generate_signal(symbol, current_time):
            return None
        
        # Prepare feature vector
        feature_vector = features[self.feature_names].values.reshape(1, -1)
        
        if np.any(np.isnan(feature_vector)):
            return None
        
        # Scale features if scaler exists
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)
        
        # Get prediction and probability
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        confidence = max(probabilities)
        
        # Check confidence threshold
        min_confidence = self.config.get('confidence_threshold', 0.6)
        if confidence < min_confidence:
            return None
        
        # Determine signal type
        if prediction == 1:  # Buy signal
            signal_type = SignalType.BUY
        else:
            return None  # Buy-only strategy
        
        # Calculate position size using Kelly criterion
        win_rate = self.calculate_win_rate()
        avg_win = self.calculate_avg_win()
        avg_loss = abs(self.calculate_avg_loss())
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.05  # Conservative default
        
        position_size = kelly_fraction * confidence  # Scale by confidence
        
        # Create signal
        signal = TradingSignal(
            timestamp=current_time,
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            features_used=self.feature_names,
            feature_values=dict(zip(self.feature_names, feature_vector[0])),
            model_prediction=prediction,
            model_probability=confidence,
            strategy_type=self.strategy_type,
            risk_score=1.0 - confidence,  # Higher confidence = lower risk
            position_size=position_size,
            metadata={
                'kelly_fraction': kelly_fraction,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
        )
        
        self.last_signal_time[symbol] = current_time
        return signal
    
    def calculate_win_rate(self) -> float:
        """Calculate historical win rate - placeholder"""
        return 0.55  # Would be calculated from historical performance
    
    def calculate_avg_win(self) -> float:
        """Calculate average winning trade return - placeholder"""
        return 0.025  # Would be calculated from historical performance
    
    def calculate_avg_loss(self) -> float:
        """Calculate average losing trade return - placeholder"""
        return -0.015  # Would be calculated from historical performance

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy for oversold/overbought conditions
    """
    
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
        features['bb_position'] = (data['close'] - sma) / (2 * std)
        features['bb_width'] = (2 * std) / sma
        
        # RSI
        features['rsi'] = self._calculate_rsi(data['close'])
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        
        # Price distance from moving averages
        for window in [10, 20, 50]:
            ma = data['close'].rolling(window).mean()
            features[f'price_ma{window}_ratio'] = data['close'] / ma
        
        # Volume patterns
        features['volume_spike'] = (
            data['volume'] / data['volume'].rolling(20).mean()
        )
        
        # Volatility regime
        features['volatility'] = data['close'].pct_change().rolling(20).std()
        features['vol_regime'] = features['volatility'] > features['volatility'].rolling(100).mean()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = None) -> pd.Series:
        """Calculate RSI"""
        if window is None:
            window = self.rsi_window
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_signal(self, features: pd.Series, symbol: str) -> Optional[TradingSignal]:
        """Generate mean reversion signal"""
        if not self.is_trained or self.model is None:
            return None
        
        current_time = features.name
        if not self.can_generate_signal(symbol, current_time):
            return None
        
        # Prepare feature vector
        feature_vector = features[self.feature_names].values.reshape(1, -1)
        
        if np.any(np.isnan(feature_vector)):
            return None
        
        # Scale features
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)
        
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
        
        # Check mean reversion conditions
        rsi = features.get('rsi', 50)
        bb_position = features.get('bb_position', 0)
        
        # Only buy when oversold (mean reversion logic)
        if rsi > 40 or bb_position > -0.5:  # Not oversold enough
            return None
        
        # Conservative position sizing for mean reversion
        position_size = min(0.05, confidence * 0.1)  # Max 5% position
        
        signal = TradingSignal(
            timestamp=current_time,
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=confidence,
            features_used=self.feature_names,
            feature_values=dict(zip(self.feature_names, feature_vector[0])),
            model_prediction=prediction,
            model_probability=confidence,
            strategy_type=self.strategy_type,
            risk_score=1.0 - confidence,
            position_size=position_size,
            metadata={
                'rsi': rsi,
                'bb_position': bb_position,
                'strategy_logic': 'oversold_condition'
            }
        )
        
        self.last_signal_time[symbol] = current_time
        return signal

class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy for trend continuation after consolidation
    """
    
    def __init__(self, config: Dict):
        super().__init__("breakout", StrategyType.BREAKOUT, config)
        self.volatility_window = config.get('volatility_window', 20)
        
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout-specific features"""
        features = pd.DataFrame(index=data.index)
        
        # Volatility features
        returns = data['close'].pct_change()
        features['volatility'] = returns.rolling(self.volatility_window).std()
        features['vol_regime'] = features['volatility'] / features['volatility'].rolling(100).mean()
        
        # Range features
        features['true_range'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        
        features['atr'] = features['true_range'].rolling(14).mean()
        features['atr_ratio'] = features['true_range'] / features['atr']
        
        # Support/Resistance levels
        for window in [10, 20, 50]:
            features[f'high_{window}'] = data['high'].rolling(window).max()
            features[f'low_{window}'] = data['low'].rolling(window).min()
            features[f'resistance_break_{window}'] = (
                data['close'] > features[f'high_{window}'].shift(1)
            ).astype(int)
        
        # Volume confirmation
        features['volume_surge'] = (
            data['volume'] / data['volume'].rolling(10).mean()
        )
        
        # Price consolidation detection
        features['consolidation'] = (
            features['volatility'] < features['volatility'].rolling(50).quantile(0.3)
        ).astype(int)
        
        return features
    
    def generate_signal(self, features: pd.Series, symbol: str) -> Optional[TradingSignal]:
        """Generate breakout signal"""
        if not self.is_trained or self.model is None:
            return None
        
        current_time = features.name
        if not self.can_generate_signal(symbol, current_time):
            return None
        
        # Prepare and validate features
        feature_vector = features[self.feature_names].values.reshape(1, -1)
        if np.any(np.isnan(feature_vector)):
            return None
        
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)
        
        # Get model prediction
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        confidence = max(probabilities)
        
        if confidence < self.config.get('confidence_threshold', 0.6):
            return None
        
        if prediction != 1:  # Buy-only constraint
            return None
        
        # Validate breakout conditions
        volume_surge = features.get('volume_surge', 1.0)
        resistance_break_20 = features.get('resistance_break_20', 0)
        
        # Require volume confirmation for breakout
        if volume_surge < 1.5 or resistance_break_20 != 1:
            return None
        
        # Higher position size for confirmed breakouts
        position_size = min(0.08, confidence * 0.15)
        
        signal = TradingSignal(
            timestamp=current_time,
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=confidence,
            features_used=self.feature_names,
            feature_values=dict(zip(self.feature_names, feature_vector[0])),
            model_prediction=prediction,
            model_probability=confidence,
            strategy_type=self.strategy_type,
            risk_score=1.0 - confidence,
            position_size=position_size,
            metadata={
                'volume_surge': volume_surge,
                'resistance_break': resistance_break_20,
                'strategy_logic': 'volume_confirmed_breakout'
            }
        )
        
        self.last_signal_time[symbol] = current_time
        return signal

class StrategyDiscoveryEngine:
    """
    Main engine for discovering and managing trading strategies
    Following de Prado's scientific approach to strategy development
    """
    
    def __init__(self, config_path: str = "/opt/financial_ml/config.yaml"):
        self.config = self._load_config(config_path)
        self.db_path = self.config['database']['sqlite']['path']
        self.ml_pipeline = FinancialMLPipeline(self.db_path)
        
        # Strategy registry
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        
        # Signal storage
        self.recent_signals: List[TradingSignal] = []
        self.signal_history: Dict[str, List[TradingSignal]] = defaultdict(list)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker(self.db_path)
        
        # Initialize database tables
        self._initialize_strategy_tables()
        
        logger.info("Strategy Discovery Engine initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_strategy_tables(self):
        """Initialize database tables for strategy storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Strategy signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                strategy_name TEXT,
                strategy_type TEXT,
                signal_type TEXT,
                confidence REAL,
                model_prediction REAL,
                position_size REAL,
                risk_score REAL,
                features_json TEXT,
                metadata_json TEXT
            )
        ''')
        
        # Strategy performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                symbol TEXT,
                timestamp TEXT,
                pnl REAL,
                return_pct REAL,
                position_size REAL,
                hold_time_hours REAL
            )
        ''')
        
        # Strategy metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_metadata (
                strategy_name TEXT PRIMARY KEY,
                strategy_type TEXT,
                config_json TEXT,
                feature_names_json TEXT,
                created_timestamp TEXT,
                last_updated TEXT,
                total_signals INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def discover_strategies(self, symbols: List[str], 
                          timeframe: str = 'algorithmic') -> Dict[str, StrategyMetrics]:
        """
        Discover trading strategies using feature importance analysis
        
        This is the core method following de Prado's approach:
        1. Load data and create features
        2. Train ML models for different strategy types
        3. Analyze feature importance
        4. Create strategies based on most important features
        5. Validate strategies using purged CV
        """
        logger.info(f"Starting strategy discovery for symbols: {symbols}")
        
        discovered_strategies = {}
        
        for symbol in symbols:
            logger.info(f"Discovering strategies for {symbol}")
            
            # Load data
            bar_type = self.config['data_ingestion']['bars'][timeframe]['dollar']
            data = self.ml_pipeline.load_data(
                symbol, 
                'dollar',
                start_date=(datetime.now() - timedelta(days=90)).isoformat()
            )
            
            if len(data) < self.config['strategy_discovery']['minimum_requirements']['min_samples']:
                logger.warning(f"Insufficient data for {symbol}: {len(data)} bars")
                continue
            
            # Generate features
            features = self.ml_pipeline.create_features(data)
            
            # Generate labels
            labeling_result = self.ml_pipeline.generate_labels(data)
            
            # Train models for different strategy types
            for strategy_type in self.config['strategy_discovery']['strategy_types']:
                logger.info(f"Training {strategy_type} strategy for {symbol}")
                
                try:
                    # Create strategy-specific features
                    strategy_features = self._create_strategy_features(
                        data, strategy_type
                    )
                    
                    # Combine with general features
                    combined_features = pd.concat([features, strategy_features], axis=1)
                    combined_features = combined_features.dropna()
                    
                    # Train model
                    model_results = self.ml_pipeline.train_model(
                        combined_features, labeling_result
                    )
                    
                    # Check minimum performance requirements
                    cv_score = model_results['cv_mean']
                    min_cv_score = self.config['strategy_discovery']['minimum_requirements']['min_cv_score']
                    
                    if cv_score < min_cv_score:
                        logger.info(f"Strategy {strategy_type} for {symbol} below minimum CV score: {cv_score:.3f}")
                        continue
                    
                    # Analyze feature importance
                    importance_analysis = self.ml_pipeline.analyze_feature_importance(
                        list(self.ml_pipeline.fitted_models.keys())[-1]
                    )
                    
                    # Create strategy based on top features
                    strategy = self._create_strategy(
                        strategy_type, symbol, model_results, importance_analysis
                    )
                    
                    if strategy:
                        self.strategies[f"{strategy_type}_{symbol}"] = strategy
                        
                        # Create strategy metrics
                        metrics = StrategyMetrics(
                            name=f"{strategy_type}_{symbol}",
                            strategy_type=StrategyType(strategy_type),
                            total_signals=0,
                            accuracy=cv_score,
                            precision=0.0,  # Will be calculated from live performance
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
                        
                        self.strategy_metrics[f"{strategy_type}_{symbol}"] = metrics
                        discovered_strategies[f"{strategy_type}_{symbol}"] = metrics
                        
                        # Save strategy to database
                        self._save_strategy_metadata(strategy, metrics)
                        
                        logger.info(f"Created strategy {strategy_type}_{symbol} with CV score: {cv_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error creating strategy {strategy_type} for {symbol}: {e}")
                    continue
        
        logger.info(f"Strategy discovery complete. Created {len(discovered_strategies)} strategies")
        return discovered_strategies
    
    def _create_strategy_features(self, data: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
        """Create strategy-specific features"""
        if strategy_type == 'momentum':
            strategy = MomentumStrategy(self.config['strategy_discovery'])
        elif strategy_type == 'mean_reversion':
            strategy = MeanReversionStrategy(self.config['strategy_discovery'])
        elif strategy_type == 'breakout':
            strategy = BreakoutStrategy(self.config['strategy_discovery'])
        else:
            # Default to momentum
            strategy = MomentumStrategy(self.config['strategy_discovery'])
        
        return strategy.generate_features(data)
    
    def _create_strategy(self, strategy_type: str, symbol: str, 
                        model_results: Dict, importance_analysis: Dict) -> Optional[BaseStrategy]:
        """Create strategy instance from model results"""
        try:
            # Get top features
            top_features = importance_analysis['top_mdi_features'][:10]  # Top 10 features
            
            # Create strategy instance
            if strategy_type == 'momentum':
                strategy = MomentumStrategy(self.config['strategy_discovery'])
            elif strategy_type == 'mean_reversion':
                strategy = MeanReversionStrategy(self.config['strategy_discovery'])
            elif strategy_type == 'breakout':
                strategy = BreakoutStrategy(self.config['strategy_discovery'])
            else:
                return None
            
            # Set strategy attributes
            strategy.model = model_results['model']
            strategy.feature_names = top_features
            strategy.is_trained = True
            
            # Add feature scaler if needed
            from sklearn.preprocessing import StandardScaler
            strategy.scaler = StandardScaler()
            # This would be fitted on training data in real implementation
            
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
            str(strategy.config),  # Would use JSON in production
            str(strategy.feature_names),
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
            
            # Get latest features
            features = self.ml_pipeline.create_features(data)
            
            if features.empty:
                continue
            
            latest_features = features.iloc[-1]
            
            # Generate signals from all strategies for this symbol
            for strategy_name, strategy in self.strategies.items():
                if symbol in strategy_name:
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
                            
                            # Store signal in database
                            self._store_signal(signal)
                            
                            logger.info(f"Generated {signal.signal_type.value} signal for {symbol} "
                                      f"from {strategy_name} with confidence {signal.confidence:.3f}")
                    
                    except Exception as e:
                        logger.error(f"Error generating signal from {strategy_name}: {e}")
        
        # Clean up old signals (keep last 1000)
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
            str(signal.feature_values),  # Would use JSON in production
            str(signal.metadata)
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
        """Record actual trade result for strategy evaluation"""
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
        sharpe_ratio = df['return_pct'].mean() / df['return_pct'].std() if df['return_pct'].std() > 0 else 0
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

# CLI Interface for strategy discovery
def main():
    """Main CLI interface"""
    import click
    
    @click.group()
    def cli():
        """Strategy Discovery Engine CLI"""
        pass
    
    @cli.command()
    @click.option('--symbols', default='BTC/USD,ETH/USD', help='Comma-separated symbols')
    @click.option('--timeframe', default='algorithmic', help='Timeframe: algorithmic or swing')
    def discover(symbols, timeframe):
        """Discover new trading strategies"""
        engine = StrategyDiscoveryEngine()
        symbol_list = symbols.split(',')
        
        strategies = engine.discover_strategies(symbol_list, timeframe)
        
        print(f"\nDiscovered {len(strategies)} strategies:")
        for name, metrics in strategies.items():
            print(f"  {name}: CV Score = {metrics.accuracy:.3f}")
    
    @cli.command()
    @click.option('--symbols', default='BTC/USD,ETH/USD', help='Comma-separated symbols')
    def signals(symbols):
        """Generate trading signals"""
        engine = StrategyDiscoveryEngine()
        symbol_list = symbols.split(',')
        
        signals = engine.generate_signals(symbol_list)
        
        print(f"\nGenerated {len(signals)} signals:")
        for signal in signals:
            print(f"  {signal.timestamp}: {signal.signal_type.value} {signal.symbol} "
                  f"(confidence: {signal.confidence:.3f}, size: {signal.position_size:.3f})")
    
    @cli.command()
    def summary():
        """Show strategy summary"""
        engine = StrategyDiscoveryEngine()
        summary = engine.get_strategy_summary()
        
        if not summary.empty:
            print("\nStrategy Summary:")
            print(summary.to_string(index=False))
        else:
            print("No strategies found. Run 'discover' first.")
    
    cli()

if __name__ == "__main__":
    main()
