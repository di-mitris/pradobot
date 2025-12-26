"""
Module 1: Financial Data Ingestion Pipeline
Following López de Prado's recommendations for proper financial data handling
Optimized for Hetzner Cloud (4 vCPU, 8GB RAM)
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import logging
from abc import ABC, abstractmethod
import aiosqlite
import signal
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        self.last_price: Optional[float] = None
        self.last_side: int = 1  # boundary condition

    def classify_tick(self, price: float) -> int:
        """Returns 1 for buy tick, -1 for sell tick"""
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
        self.s_pos: float = 0.0
        self.s_neg: float = 0.0
        self.last_price: Optional[float] = None

    def update(self, price: float) -> bool:
        """Returns True if threshold is breached (event detected)"""
        if self.last_price is None:
            self.last_price = price
            return False

        diff = price - self.last_price
        self.last_price = price
        
        self.s_pos = max(0, self.s_pos + diff)
        self.s_neg = min(0, self.s_neg + diff)

        if self.s_neg < -self.threshold:
            self.s_neg = 0
            return True
        elif self.s_pos > self.threshold:
            self.s_pos = 0
            return True

        return False

    def reset(self):
        """Reset filter state"""
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.last_price = None


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

        # Expected values for imbalance bars (EWMA)
        self.ewm_alpha = 0.05
        self.expected_ticks = 1000.0
        self.expected_imbalance = 0.0

    def reset_bar_data(self):
        """Reset accumulated bar data"""
        self.trades: List[Trade] = []
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
        if not self.trades:
            return False
            
        if self.bar_type == 'tick_imbalance':
            expected_runs = self.expected_ticks * max(0.5, abs(2 * 0.5 - 1))
            return abs(self.tick_imbalance) >= expected_runs
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

    def _create_bar(self) -> Optional[BarData]:
        """Create bar from accumulated trades"""
        if not self.trades:
            return None

        prices = [t.price for t in self.trades]
        volumes = [t.volume for t in self.trades]
        dollar_volumes = [t.volume * t.price for t in self.trades]

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
        self.expected_ticks = (1 - self.ewm_alpha) * self.expected_ticks + \
                              self.ewm_alpha * len(self.trades)


class AsyncDatabaseManager:
    """
    Async database manager with connection pooling for high-throughput operations
    """
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._initialized = False
        self._write_lock = asyncio.Lock()
        
        # Buffer for batch writes
        self._trade_buffer: List[Trade] = []
        self._bar_buffer: List[BarData] = []
        self._event_buffer: List[Dict] = []
        self._buffer_lock = asyncio.Lock()
        self.buffer_size = 100
        self.flush_interval = 30.0

    async def initialize(self):
        """Initialize database and connection pool"""
        if self._initialized:
            return
            
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('PRAGMA journal_mode=WAL')
            await db.execute('PRAGMA synchronous=NORMAL')
            await db.execute('PRAGMA cache_size=200000')  # ~200MB cache
            await db.execute('PRAGMA temp_store=MEMORY')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    side TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    bar_type TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    vwap REAL NOT NULL,
                    dollar_volume REAL NOT NULL,
                    tick_count INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for query performance
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp 
                ON trades(symbol, timestamp)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_bars_symbol_type_timestamp 
                ON bars(symbol, bar_type, timestamp)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_symbol_timestamp 
                ON events(symbol, timestamp)
            ''')
            
            await db.commit()
        
        # Initialize connection pool
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute('PRAGMA journal_mode=WAL')
            await self._pool.put(conn)
        
        self._initialized = True
        logger.info(f"Database initialized with pool size {self.pool_size}")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        conn = await self._pool.get()
        try:
            yield conn
        finally:
            await self._pool.put(conn)

    async def buffer_trade(self, trade: Trade):
        """Add trade to buffer for batch insert"""
        async with self._buffer_lock:
            self._trade_buffer.append(trade)
            if len(self._trade_buffer) >= self.buffer_size:
                await self._flush_trades()

    async def buffer_bar(self, bar: BarData):
        """Add bar to buffer for batch insert"""
        async with self._buffer_lock:
            self._bar_buffer.append(bar)
            if len(self._bar_buffer) >= self.buffer_size:
                await self._flush_bars()

    async def buffer_event(self, event: Dict):
        """Add event to buffer for batch insert"""
        async with self._buffer_lock:
            self._event_buffer.append(event)
            if len(self._event_buffer) >= self.buffer_size:
                await self._flush_events()

    async def _flush_trades(self):
        """Flush trade buffer to database"""
        if not self._trade_buffer:
            return
            
        trades_to_insert = self._trade_buffer.copy()
        self._trade_buffer.clear()
        
        async with self._write_lock:
            async with self.get_connection() as conn:
                await conn.executemany(
                    '''INSERT INTO trades (timestamp, symbol, price, volume, side)
                       VALUES (?, ?, ?, ?, ?)''',
                    [(t.timestamp.isoformat(), t.symbol, t.price, t.volume, t.side)
                     for t in trades_to_insert]
                )
                await conn.commit()
        
        logger.debug(f"Flushed {len(trades_to_insert)} trades to database")

    async def _flush_bars(self):
        """Flush bar buffer to database"""
        if not self._bar_buffer:
            return
            
        bars_to_insert = self._bar_buffer.copy()
        self._bar_buffer.clear()
        
        async with self._write_lock:
            async with self.get_connection() as conn:
                await conn.executemany(
                    '''INSERT INTO bars (timestamp, symbol, bar_type, open, high, low, 
                                        close, volume, vwap, dollar_volume, tick_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    [(b.timestamp.isoformat(), b.symbol, b.bar_type, b.open, b.high,
                      b.low, b.close, b.volume, b.vwap, b.dollar_volume, b.tick_count)
                     for b in bars_to_insert]
                )
                await conn.commit()
        
        logger.debug(f"Flushed {len(bars_to_insert)} bars to database")

    async def _flush_events(self):
        """Flush event buffer to database"""
        if not self._event_buffer:
            return
            
        events_to_insert = self._event_buffer.copy()
        self._event_buffer.clear()
        
        async with self._write_lock:
            async with self.get_connection() as conn:
                await conn.executemany(
                    '''INSERT INTO events (timestamp, symbol, event_type, price, details)
                       VALUES (?, ?, ?, ?, ?)''',
                    [(e['timestamp'], e['symbol'], e['event_type'], e['price'], e.get('details', ''))
                     for e in events_to_insert]
                )
                await conn.commit()

    async def flush_all(self):
        """Flush all buffers"""
        async with self._buffer_lock:
            await self._flush_trades()
            await self._flush_bars()
            await self._flush_events()

    async def get_bars(self, symbol: str, bar_type: str,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve bars from database"""
        query = '''
            SELECT timestamp, open, high, low, close, volume, vwap, dollar_volume, tick_count
            FROM bars WHERE symbol = ? AND bar_type = ?
        '''
        params: List[Any] = [symbol, bar_type]
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time.isoformat())
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time.isoformat())
        
        query += ' ORDER BY timestamp'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        async with self.get_connection() as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame([dict(row) for row in rows])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df

    async def close(self):
        """Close all connections and flush buffers"""
        await self.flush_all()
        
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
        
        self._initialized = False
        logger.info("Database connections closed")


class KrakenDataIngestion:
    """
    Main data ingestion class for Kraken WebSocket
    Optimized for Hetzner Cloud deployment
    """
    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config.get('symbols', ['BTC/USD', 'ETH/USD'])
        self.bar_configs = config.get('bar_configs', {
            'tick': 1000,
            'volume': 10.0,
            'dollar': 100000.0,
            'dollar_imbalance': 50000.0
        })
        
        # WebSocket settings
        self.ws_uri = config.get('websocket_uri', 'wss://ws.kraken.com/v2')
        self.reconnect_delay = config.get('reconnect_delay', 5.0)
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self.ping_interval = config.get('ping_interval', 30)
        self.ping_timeout = config.get('ping_timeout', 10)
        
        # State
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self._shutdown_event = asyncio.Event()
        self._reconnect_count = 0
        
        # Data storage
        self.bar_constructors: Dict[str, Dict[str, InformationDrivenBars]] = {}
        self.cusum_filters: Dict[str, CUSUMFilter] = {}
        
        # Initialize bar constructors
        self._initialize_bar_constructors()
        
        # Database manager
        db_path = config.get('db_path', '/opt/financial_ml/data/financial_data.db')
        pool_size = config.get('db_pool_size', 5)
        self.db = AsyncDatabaseManager(db_path, pool_size)
        
        # Callbacks
        self.bar_callbacks: List[Callable] = []
        self.event_callbacks: List[Callable] = []
        
        # Metrics
        self.metrics = {
            'trades_processed': 0,
            'bars_created': 0,
            'events_detected': 0,
            'reconnections': 0,
            'last_trade_time': None,
            'start_time': None
        }

    def _initialize_bar_constructors(self):
        """Initialize bar constructors for each symbol"""
        cusum_threshold = self.config.get('cusum_threshold', 0.01)
        
        for symbol in self.symbols:
            self.bar_constructors[symbol] = {}
            self.cusum_filters[symbol] = CUSUMFilter(threshold=cusum_threshold)
            
            for bar_type, threshold in self.bar_configs.items():
                self.bar_constructors[symbol][bar_type] = \
                    InformationDrivenBars(symbol, bar_type, threshold)

    def add_bar_callback(self, callback: Callable):
        """Add callback function for new bars"""
        self.bar_callbacks.append(callback)

    def add_event_callback(self, callback: Callable):
        """Add callback function for events"""
        self.event_callbacks.append(callback)

    async def _connect_websocket(self) -> bool:
        """Connect to Kraken WebSocket with retry logic"""
        try:
            self.websocket = await websockets.connect(
                self.ws_uri,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                close_timeout=10
            )
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
            
            self._reconnect_count = 0
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    async def _reconnect_with_backoff(self):
        """Reconnect with exponential backoff"""
        while self.running and self._reconnect_count < self.max_reconnect_attempts:
            self._reconnect_count += 1
            delay = min(self.reconnect_delay * (2 ** (self._reconnect_count - 1)), 300)
            
            logger.info(f"Reconnection attempt {self._reconnect_count}/{self.max_reconnect_attempts} "
                       f"in {delay:.1f}s")
            
            await asyncio.sleep(delay)
            
            if await self._connect_websocket():
                self.metrics['reconnections'] += 1
                return True
        
        logger.error("Max reconnection attempts reached")
        return False

    async def _listen_websocket(self):
        """Listen for WebSocket messages with error handling"""
        try:
            async for message in self.websocket:
                if self._shutdown_event.is_set():
                    break
                    
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            if self.running:
                await self._reconnect_with_backoff()
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
            if self.running:
                await self._reconnect_with_backoff()

    async def _process_message(self, data: Dict):
        """Process incoming WebSocket message"""
        if data.get("channel") == "trade" and "data" in data:
            for trade_data in data["data"]:
                try:
                    trade = Trade(
                        timestamp=pd.Timestamp(trade_data["timestamp"]),
                        price=float(trade_data["price"]),
                        volume=float(trade_data["qty"]),
                        side=trade_data["side"],
                        symbol=trade_data["symbol"]
                    )
                    await self._process_trade(trade)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Invalid trade data: {e}")

    async def _process_trade(self, trade: Trade):
        """Process individual trade"""
        self.metrics['trades_processed'] += 1
        self.metrics['last_trade_time'] = trade.timestamp
        
        # Buffer raw trade
        await self.db.buffer_trade(trade)
        
        # Check for CUSUM events
        if trade.symbol in self.cusum_filters:
            if self.cusum_filters[trade.symbol].update(trade.price):
                self.metrics['events_detected'] += 1
                await self._trigger_event("cusum_event", trade)
        
        # Process through bar constructors
        if trade.symbol in self.bar_constructors:
            for bar_type, constructor in self.bar_constructors[trade.symbol].items():
                bar = constructor.add_trade(trade)
                if bar:
                    self.metrics['bars_created'] += 1
                    await self.db.buffer_bar(bar)
                    await self._trigger_bar_callbacks(bar)

    async def _trigger_bar_callbacks(self, bar: BarData):
        """Trigger all registered bar callbacks"""
        for callback in self.bar_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(bar)
                else:
                    callback(bar)
            except Exception as e:
                logger.error(f"Error in bar callback: {e}")

    async def _trigger_event(self, event_type: str, trade: Trade):
        """Trigger event callbacks and store event"""
        event = {
            'timestamp': trade.timestamp.isoformat(),
            'symbol': trade.symbol,
            'event_type': event_type,
            'price': trade.price,
            'details': json.dumps({'volume': trade.volume, 'side': trade.side})
        }
        
        await self.db.buffer_event(event)
        
        for callback in self.event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, trade)
                else:
                    callback(event_type, trade)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    async def _periodic_flush(self):
        """Periodically flush database buffers"""
        while self.running:
            try:
                await asyncio.sleep(self.db.flush_interval)
                await self.db.flush_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")

    async def _health_check(self):
        """Periodic health check and metrics logging"""
        while self.running:
            try:
                await asyncio.sleep(60)
                
                uptime = (datetime.now() - self.metrics['start_time']).total_seconds() \
                    if self.metrics['start_time'] else 0
                trades_per_sec = self.metrics['trades_processed'] / uptime if uptime > 0 else 0
                
                logger.info(
                    f"Health Check - Trades: {self.metrics['trades_processed']}, "
                    f"Bars: {self.metrics['bars_created']}, "
                    f"Events: {self.metrics['events_detected']}, "
                    f"Rate: {trades_per_sec:.2f}/s, "
                    f"Reconnections: {self.metrics['reconnections']}"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check: {e}")

    async def start(self):
        """Start the data ingestion pipeline"""
        logger.info("Starting data ingestion pipeline")
        
        # Initialize database
        await self.db.initialize()
        
        self.running = True
        self.metrics['start_time'] = datetime.now()
        
        # Connect to WebSocket
        if not await self._connect_websocket():
            if not await self._reconnect_with_backoff():
                logger.error("Failed to establish WebSocket connection")
                return
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._listen_websocket()),
            asyncio.create_task(self._periodic_flush()),
            asyncio.create_task(self._health_check())
        ]
        
        try:
            # Wait for shutdown signal or task completion
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        """Stop the data ingestion pipeline gracefully"""
        logger.info("Stopping data ingestion pipeline")
        self.running = False
        self._shutdown_event.set()
        
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
        
        # Flush and close database
        await self.db.close()
        
        logger.info("Data ingestion pipeline stopped")

    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return self.metrics.copy()

    async def get_bars(self, symbol: str, bar_type: str,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve bars from database"""
        return await self.db.get_bars(symbol, bar_type, start_time, end_time)


async def main():
    """Main entry point with signal handling"""
    import yaml
    
    # Load configuration
    config_path = os.environ.get('CONFIG_PATH', '/opt/financial_ml/config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'symbols': ['BTC/USD', 'ETH/USD'],
            'bar_configs': {
                'tick': 1000,
                'volume': 10.0,
                'dollar': 100000.0,
                'dollar_imbalance': 50000.0
            },
            'db_path': '/opt/financial_ml/data/financial_data.db',
            'db_pool_size': 5,
            'websocket_uri': 'wss://ws.kraken.com/v2',
            'cusum_threshold': 0.01
        }
    
    # Extract data ingestion config if nested
    if 'data_ingestion' in config:
        ingestion_config = {
            'symbols': config['data_ingestion'].get('symbols', {}).get('algorithmic', ['BTC/USD', 'ETH/USD']),
            'bar_configs': config['data_ingestion'].get('bars', {}).get('algorithmic', {}),
            'db_path': config.get('database', {}).get('sqlite', {}).get('path', '/opt/financial_ml/data/financial_data.db'),
            'websocket_uri': config['data_ingestion'].get('websocket', {}).get('uri', 'wss://ws.kraken.com/v2'),
            'cusum_threshold': config['data_ingestion'].get('cusum_filter', {}).get('algorithmic_threshold', 0.01)
        }
    else:
        ingestion_config = config
    
    # Initialize pipeline
    pipeline = KrakenDataIngestion(ingestion_config)
    
    # Set up signal handlers
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(pipeline.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    # Add sample callback
    async def on_new_bar(bar: BarData):
        logger.info(f"New {bar.bar_type} bar for {bar.symbol}: "
                   f"OHLC={bar.open:.2f}/{bar.high:.2f}/{bar.low:.2f}/{bar.close:.2f}")
    
    pipeline.add_bar_callback(on_new_bar)
    
    # Start pipeline
    try:
        await pipeline.start()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
