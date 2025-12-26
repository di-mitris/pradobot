"""
Module 1: Financial Data Ingestion Pipeline
Following López de Prado's recommendations for proper financial data handling
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
from abc import ABC, abstractmethod
import sqlite3
import threading
from queue import Queue
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade data structure"""
    timestamp: pd.Timestamp
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    symbol: str

@dataclass
class BarData:
    """Bar data structure following de Prado's recommendations"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    dollar_volume: float
    tick_count: int
    symbol: str
    bar_type: str  # 'tick', 'volume', 'dollar', 'imbalance'

class TickClassifier:
    """
    Implements tick rule for trade classification
    Following de Prado Chapter 2, Section 2.3.2.1
    """
    def __init__(self):
        self.last_price = None
        self.last_side = 1  # boundary condition
    
    def classify_tick(self, price: float) -> int:
        """
        Returns 1 for buy tick, -1 for sell tick
        """
        if self.last_price is None:
            self.last_price = price
            return self.last_side
        
        if price > self.last_price:
            side = 1
        elif price < self.last_price:
            side = -1
        else:
            side = self.last_side
        
        self.last_price = price
        self.last_side = side
        return side

class CUSUMFilter:
    """
    CUSUM filter for event-based sampling
    Following de Prado Chapter 2, Section 2.5.2.1
    """
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.last_price = None
    
    def update(self, price: float) -> bool:
        """
        Returns True if threshold is breached (event detected)
        """
        if self.last_price is None:
            self.last_price = price
            return False
        
        diff = price - self.last_price
        self.s_pos = max(0, self.s_pos + diff)
        self.s_neg = min(0, self.s_neg + diff)
        
        if self.s_neg < -self.threshold:
            self.s_neg = 0
            return True
        elif self.s_pos > self.threshold:
            self.s_pos = 0
            return True
        
        return False

class InformationDrivenBars:
    """
    Implements information-driven bar construction
    Following de Prado Chapter 2, Section 2.3.2
    """
    def __init__(self, symbol: str, bar_type: str, threshold: float):
        self.symbol = symbol
        self.bar_type = bar_type
        self.threshold = threshold
        self.tick_classifier = TickClassifier()
        
        # Accumulated data for current bar
        self.reset_bar_data()
        
        # Expected values for imbalance bars
        self.ewm_alpha = 0.05
        self.expected_ticks = 1000  # Initial estimate
        self.expected_imbalance = 0.0
        
    def reset_bar_data(self):
        """Reset accumulated bar data"""
        self.trades = []
        self.tick_imbalance = 0.0
        self.volume_imbalance = 0.0
        self.dollar_imbalance = 0.0
        self.tick_runs = 0.0
        self.volume_runs = 0.0
        self.dollar_runs = 0.0
        
    def add_trade(self, trade: Trade) -> Optional[BarData]:
        """
        Add trade and check if bar should be formed
        Returns BarData if bar is complete, None otherwise
        """
        self.trades.append(trade)
        side = self.tick_classifier.classify_tick(trade.price)
        
        # Update imbalances
        self.tick_imbalance += side
        self.volume_imbalance += side * trade.volume
        self.dollar_imbalance += side * trade.volume * trade.price
        
        # Update runs (max of buy or sell activity)
        if side == 1:
            self.tick_runs += 1
            self.volume_runs += trade.volume
            self.dollar_runs += trade.volume * trade.price
        
        # Check bar completion condition
        if self._check_bar_completion():
            bar = self._create_bar()
            self._update_expectations()
            self.reset_bar_data()
            return bar
        
        return None
    
    def _check_bar_completion(self) -> bool:
        """Check if current bar should be completed"""
        if self.bar_type == 'tick_imbalance':
            return abs(self.tick_imbalance) >= self.expected_ticks * abs(2 * 0.5 - 1)  # Simplified
        elif self.bar_type == 'volume_imbalance':
            return abs(self.volume_imbalance) >= self.threshold
        elif self.bar_type == 'dollar_imbalance':
            return abs(self.dollar_imbalance) >= self.threshold
        elif self.bar_type == 'volume':
            return sum(t.volume for t in self.trades) >= self.threshold
        elif self.bar_type == 'dollar':
            return sum(t.volume * t.price for t in self.trades) >= self.threshold
        else:  # tick bars
            return len(self.trades) >= self.threshold
    
    def _create_bar(self) -> BarData:
        """Create bar from accumulated trades"""
        if not self.trades:
            return None
        
        prices = [t.price for t in self.trades]
        volumes = [t.volume for t in self.trades]
        dollar_volumes = [t.volume * t.price for t in self.trades]
        
        # Calculate VWAP
        total_dollar_volume = sum(dollar_volumes)
        total_volume = sum(volumes)
        vwap = total_dollar_volume / total_volume if total_volume > 0 else prices[-1]
        
        return BarData(
            timestamp=self.trades[-1].timestamp,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=total_volume,
            vwap=vwap,
            dollar_volume=total_dollar_volume,
            tick_count=len(self.trades),
            symbol=self.symbol,
            bar_type=self.bar_type
        )
    
    def _update_expectations(self):
        """Update expected values using EWMA"""
        # Simplified expectation updates
        self.expected_ticks = (1 - self.ewm_alpha) * self.expected_ticks + \
                             self.ewm_alpha * len(self.trades)

class KrakenDataIngestion:
    """
    Main data ingestion class for Kraken WebSocket
    """
    def __init__(self, symbols: List[str], bar_configs: Dict):
        self.symbols = symbols
        self.bar_configs = bar_configs
        self.websocket = None
        self.running = False
        
        # Data storage
        self.data_queue = Queue()
        self.bar_constructors = {}
        self.cusum_filters = {}
        
        # Initialize bar constructors for each symbol and type
        self._initialize_bar_constructors()
        
        # Database connection
        self.db_path = "financial_data.db"
        self._initialize_database()
        
        # Event callbacks
        self.bar_callbacks: List[Callable] = []
        self.event_callbacks: List[Callable] = []
        
    def _initialize_bar_constructors(self):
        """Initialize bar constructors for each symbol"""
        for symbol in self.symbols:
            self.bar_constructors[symbol] = {}
            self.cusum_filters[symbol] = CUSUMFilter(threshold=0.01)  # 1% threshold
            
            for bar_type, threshold in self.bar_configs.items():
                self.bar_constructors[symbol][bar_type] = \
                    InformationDrivenBars(symbol, bar_type, threshold)
    
    def _initialize_database(self):
        """Initialize SQLite database for data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                price REAL,
                volume REAL,
                side TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                bar_type TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                vwap REAL,
                dollar_volume REAL,
                tick_count INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                event_type TEXT,
                price REAL,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_bar_callback(self, callback: Callable):
        """Add callback function for new bars"""
        self.bar_callbacks.append(callback)
    
    def add_event_callback(self, callback: Callable):
        """Add callback function for events"""
        self.event_callbacks.append(callback)
    
    async def _connect_websocket(self):
        """Connect to Kraken WebSocket"""
        uri = "wss://ws.kraken.com/v2"
        
        try:
            self.websocket = await websockets.connect(uri)
            logger.info("Connected to Kraken WebSocket")
            
            # Subscribe to trade feeds
            for symbol in self.symbols:
                subscribe_message = {
                    "method": "subscribe",
                    "params": {
                        "channel": "trade",
                        "symbol": [symbol],
                        "snapshot": False
                    }
                }
                await self.websocket.send(json.dumps(subscribe_message))
                logger.info(f"Subscribed to {symbol} trades")
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def _listen_websocket(self):
        """Listen for WebSocket messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._process_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
    
    async def _process_message(self, data: Dict):
        """Process incoming WebSocket message"""
        if data.get("channel") == "trade" and "data" in data:
            for trade_data in data["data"]:
                trade = Trade(
                    timestamp=pd.Timestamp(trade_data["timestamp"]),
                    price=float(trade_data["price"]),
                    volume=float(trade_data["qty"]),
                    side=trade_data["side"],
                    symbol=trade_data["symbol"]
                )
                
                await self._process_trade(trade)
    
    async def _process_trade(self, trade: Trade):
        """Process individual trade"""
        # Store raw trade
        self._store_trade(trade)
        
        # Check for CUSUM events
        if self.cusum_filters[trade.symbol].update(trade.price):
            self._trigger_event_callbacks("cusum_event", trade)
        
        # Process through bar constructors
        for bar_type, constructor in self.bar_constructors[trade.symbol].items():
            bar = constructor.add_trade(trade)
            if bar:
                self._store_bar(bar)
                self._trigger_bar_callbacks(bar)
    
    def _store_trade(self, trade: Trade):
        """Store trade in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, price, volume, side)
            VALUES (?, ?, ?, ?, ?)
        ''', (trade.timestamp.isoformat(), trade.symbol, trade.price, 
              trade.volume, trade.side))
        
        conn.commit()
        conn.close()
    
    def _store_bar(self, bar: BarData):
        """Store bar in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bars (timestamp, symbol, bar_type, open, high, low, 
                            close, volume, vwap, dollar_volume, tick_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (bar.timestamp.isoformat(), bar.symbol, bar.bar_type, bar.open,
              bar.high, bar.low, bar.close, bar.volume, bar.vwap,
              bar.dollar_volume, bar.tick_count))
        
        conn.commit()
        conn.close()
    
    def _trigger_bar_callbacks(self, bar: BarData):
        """Trigger all registered bar callbacks"""
        for callback in self.bar_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Error in bar callback: {e}")
    
    def _trigger_event_callbacks(self, event_type: str, trade: Trade):
        """Trigger all registered event callbacks"""
        for callback in self.event_callbacks:
            try:
                callback(event_type, trade)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    async def start(self):
        """Start the data ingestion pipeline"""
        self.running = True
        await self._connect_websocket()
        logger.info("Starting data ingestion pipeline")
        
        try:
            await self._listen_websocket()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            self.running = False
    
    def stop(self):
        """Stop the data ingestion pipeline"""
        self.running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
    
    def get_bars(self, symbol: str, bar_type: str, 
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve bars from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM bars 
            WHERE symbol = ? AND bar_type = ?
        '''
        params = [symbol, bar_type]
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time.isoformat())
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time.isoformat())
        
        query += ' ORDER BY timestamp'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df

# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    symbols = ["BTC/USD", "ETH/USD"]
    bar_configs = {
        "tick": 1000,
        "volume": 10.0,
        "dollar": 100000.0,
        "dollar_imbalance": 50000.0
    }
    
    # Initialize ingestion pipeline
    pipeline = KrakenDataIngestion(symbols, bar_configs)
    
    # Add callback for new bars
    def on_new_bar(bar: BarData):
        logger.info(f"New {bar.bar_type} bar for {bar.symbol}: "
                   f"OHLC={bar.open:.2f}/{bar.high:.2f}/{bar.low:.2f}/{bar.close:.2f}")
    
    pipeline.add_bar_callback(on_new_bar)
    
    # Start pipeline
    try:
        asyncio.run(pipeline.start())
    except KeyboardInterrupt:
        logger.info("Shutting down pipeline")
        pipeline.stop()
