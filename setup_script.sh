#!/bin/bash
# Financial ML Pipeline Setup Script for Raspberry Pi 5
# Run as: curl -sSL https://raw.githubusercontent.com/your-repo/setup.sh | bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/financial_ml"
DATA_DIR="/opt/financial_ml/data"
LOG_DIR="/var/log/financial_ml"
USER="pi"
GROUP="pi"

echo -e "${GREEN}Financial ML Pipeline Setup for Raspberry Pi 5${NC}"
echo "=================================================="

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo -e "${YELLOW}Warning: This script is optimized for Raspberry Pi 5${NC}"
fi

# Check available memory
TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
if [ $TOTAL_MEM -lt 7000000 ]; then
    echo -e "${YELLOW}Warning: Less than 7GB RAM detected. Some features may be limited.${NC}"
fi

# Update system
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo -e "${GREEN}Installing system dependencies...${NC}"
sudo apt install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    git \
    sqlite3 \
    libsqlite3-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    libffi-dev \
    libssl-dev \
    curl \
    htop \
    iotop \
    systemd-container

# Install TA-Lib from source (ARM64 compatible)
echo -e "${GREEN}Installing TA-Lib...${NC}"
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr/local
make
sudo make install
sudo ldconfig
cd /

# Create directories
echo -e "${GREEN}Creating directories...${NC}"
sudo mkdir -p $INSTALL_DIR
sudo mkdir -p $DATA_DIR
sudo mkdir -p $LOG_DIR
sudo mkdir -p /etc/financial_ml

# Set permissions
sudo chown -R $USER:$GROUP $INSTALL_DIR
sudo chown -R $USER:$GROUP $LOG_DIR

# Create Python virtual environment
echo -e "${GREEN}Creating Python virtual environment...${NC}"
cd $INSTALL_DIR
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

# Install Python packages optimized for ARM64
echo -e "${GREEN}Installing Python packages (this may take 15-30 minutes)...${NC}"

# Install numpy/scipy from piwheels for faster installation
pip install --extra-index-url https://www.piwheels.org/simple/ numpy scipy

# Install other core packages
pip install pandas scikit-learn matplotlib seaborn

# Install remaining requirements
cat > requirements_pi.txt << EOF
# Raspberry Pi optimized requirements
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
aiosqlite>=0.19.0
websockets>=11.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
python-dotenv>=1.0.0
loguru>=0.7.0
psutil>=5.9.0
tqdm>=4.65.0
click>=8.1.0
schedule>=1.2.0
python-dateutil>=2.8.0
joblib>=1.3.0
ta-lib>=0.4.25
pytest>=7.4.0
black>=23.0.0
systemd-python>=235
EOF

pip install -r requirements_pi.txt

# Download and setup project files
echo -e "${GREEN}Setting up project files...${NC}"
# In a real deployment, you would clone from git:
# git clone https://github.com/your-repo/financial-ml-pipeline.git .

# For now, create the module structure
mkdir -p src
touch src/__init__.py
touch src/data_ingestion.py
touch src/ml_pipeline.py
touch src/strategy_discovery.py

# Create configuration file
echo -e "${GREEN}Creating configuration...${NC}"
cp config.yaml $INSTALL_DIR/

# Create systemd service files
echo -e "${GREEN}Installing systemd services...${NC}"
sudo tee /etc/systemd/system/financial-ml-ingestion.service > /dev/null << 'EOF'
[Unit]
Description=Financial ML Data Ingestion Service
Documentation=https://github.com/your-repo/financial-ml-pipeline
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/opt/financial_ml
ExecStart=/opt/financial_ml/venv/bin/python -m src.data_ingestion --config /opt/financial_ml/config.yaml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=financial-ml-ingestion

NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/financial_ml/data
ReadWritePaths=/var/log/financial_ml
PrivateTmp=true

MemoryLimit=2G
CPUQuota=75%
TasksMax=100

Environment=PYTHONPATH=/opt/financial_ml
Environment=PYTHONUNBUFFERED=1

CPUAffinity=1 2 3
IOSchedulingClass=2
IOSchedulingPriority=4
Nice=5

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/financial-ml-training.service > /dev/null << 'EOF'
[Unit]
Description=Financial ML Training Service
After=financial-ml-ingestion.service

[Service]
Type=oneshot
User=pi
Group=pi
WorkingDirectory=/opt/financial_ml
ExecStart=/opt/financial_ml/venv/bin/python -m src.ml_pipeline --mode train
StandardOutput=journal
StandardError=journal
TimeoutSec=3600
MemoryLimit=4G
EOF

sudo tee /etc/systemd/system/financial-ml-training.timer > /dev/null << 'EOF'
[Unit]
Description=Run Financial ML Training Daily
Requires=financial-ml-training.service

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

# Create monitoring script
echo -e "${GREEN}Creating monitoring scripts...${NC}"
tee $INSTALL_DIR/monitor.sh > /dev/null << 'EOF'
#!/bin/bash
# System monitoring script for Financial ML Pipeline

LOGFILE="/var/log/financial_ml/monitor.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Function to log with timestamp
log() {
    echo "[$TIMESTAMP] $1" >> $LOGFILE
}

# Check memory usage
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
if (( $(echo "$MEMORY_USAGE > 85.0" | bc -l) )); then
    log "WARNING: High memory usage: ${MEMORY_USAGE}%"
fi

# Check CPU temperature
CPU_TEMP=$(vcgencmd measure_temp | cut -d= -f2 | cut -d\' -f1)
if (( $(echo "$CPU_TEMP > 75.0" | bc -l) )); then
    log "WARNING: High CPU temperature: ${CPU_TEMP}°C"
fi

# Check disk space
DISK_USAGE=$(df /opt/financial_ml | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    log "WARNING: High disk usage: ${DISK_USAGE}%"
fi

# Check service status
if ! systemctl is-active --quiet financial-ml-ingestion; then
    log "ERROR: Data ingestion service is not running"
    systemctl restart financial-ml-ingestion
fi

# Log system stats
log "System Stats - Memory: ${MEMORY_USAGE}%, CPU Temp: ${CPU_TEMP}°C, Disk: ${DISK_USAGE}%"
EOF

chmod +x $INSTALL_DIR/monitor.sh

# Create logrotate configuration
sudo tee /etc/logrotate.d/financial-ml > /dev/null << 'EOF'
/var/log/financial_ml/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 pi pi
}
EOF

# Setup cron job for monitoring
echo -e "${GREEN}Setting up monitoring cron job...${NC}"
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/financial_ml/monitor.sh") | crontab -

# Enable and start services
echo -e "${GREEN}Enabling systemd services...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable financial-ml-ingestion.service
sudo systemctl enable financial-ml-training.timer

# Create helper scripts
echo -e "${GREEN}Creating helper scripts...${NC}"
tee $INSTALL_DIR/financial-ml-cli > /dev/null << 'EOF'
#!/bin/bash
# Financial ML CLI Helper

VENV_PYTHON="/opt/financial_ml/venv/bin/python"
PROJECT_DIR="/opt/financial_ml"

case "$1" in
    "start")
        echo "Starting data ingestion service..."
        sudo systemctl start financial-ml-ingestion
        ;;
    "stop")
        echo "Stopping data ingestion service..."
        sudo systemctl stop financial-ml-ingestion
        ;;
    "status")
        echo "=== Service Status ==="
        systemctl status financial-ml-ingestion --no-pager
        echo ""
        echo "=== Recent Logs ==="
        journalctl -u financial-ml-ingestion -n 10 --no-pager
        ;;
    "logs")
        journalctl -u financial-ml-ingestion -f
        ;;
    "train")
        echo "Starting model training..."
        cd $PROJECT_DIR && $VENV_PYTHON -m src.ml_pipeline --mode train
        ;;
    "discover")
        echo "Starting strategy discovery..."
        cd $PROJECT_DIR && $VENV_PYTHON -m src.strategy_discovery discover --symbols BTC/USD,ETH/USD
        ;;
    "signals")
        echo "Generating trading signals..."
        cd $PROJECT_DIR && $VENV_PYTHON -m src.strategy_discovery signals --symbols BTC/USD,ETH/USD
        ;;
    "monitor")
        echo "=== System Resources ==="
        free -h
        echo ""
        echo "=== CPU Temperature ==="
        vcgencmd measure_temp
        echo ""
        echo "=== Disk Usage ==="
        df -h /opt/financial_ml
        echo ""
        echo "=== Service Status ==="
        systemctl is-active financial-ml-ingestion
        ;;
    *)
        echo "Financial ML Pipeline CLI"
        echo "Usage: $0 {start|stop|status|logs|train|discover|signals|monitor}"
        echo ""
        echo "Commands:"
        echo "  start     - Start data ingestion service"
        echo "  stop      - Stop data ingestion service"
        echo "  status    - Show service status and recent logs"
        echo "  logs      - Follow service logs"
        echo "  train     - Run model training"
        echo "  discover  - Run strategy discovery"
        echo "  signals   - Generate trading signals"
        echo "  monitor   - Show system monitoring info"
        ;;
esac
EOF

chmod +x $INSTALL_DIR/financial-ml-cli
sudo ln -sf $INSTALL_DIR/financial-ml-cli /usr/local/bin/financial-ml

# Create initial database
echo -e "${GREEN}Initializing database...${NC}"
cd $INSTALL_DIR
$INSTALL_DIR/venv/bin/python -c "
import sqlite3
import os

db_path = '/opt/financial_ml/data/financial_data.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Enable WAL mode for better performance
cursor.execute('PRAGMA journal_mode=WAL;')
cursor.execute('PRAGMA synchronous=NORMAL;')
cursor.execute('PRAGMA cache_size=100000;')

print('Database initialized successfully')
conn.close()
"

# Set up Raspberry Pi specific optimizations
echo -e "${GREEN}Applying Raspberry Pi optimizations...${NC}"

# Increase GPU memory split for better performance
if ! grep -q "gpu_mem=128" /boot/config.txt; then
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt
fi

# Disable swap to protect SD card
sudo dphys-swapfile swapoff
sudo dphys-swapfile uninstall
sudo update-rc.d dphys-swapfile remove

# Optimize I/O scheduler for SD card
echo 'ACTION=="add|change", KERNEL=="mmcblk[0-9]*", ATTR{queue/scheduler}="mq-deadline"' | sudo tee /etc/udev/rules.d/60-ioschedulers.rules

# Create performance monitoring dashboard
echo -e "${GREEN}Creating performance dashboard...${NC}"
tee $INSTALL_DIR/dashboard.py > /dev/null << 'EOF'
#!/usr/bin/env python3
"""
Simple performance dashboard for Financial ML Pipeline
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

def show_dashboard():
    db_path = "/opt/financial_ml/data/financial_data.db"
    
    if not os.path.exists(db_path):
        print("Database not found. Run data ingestion first.")
        return
    
    conn = sqlite3.connect(db_path)
    
    print("=" * 60)
    print("FINANCIAL ML PIPELINE DASHBOARD")
    print("=" * 60)
    
    # Recent bars
    try:
        bars_df = pd.read_sql_query("""
            SELECT symbol, bar_type, COUNT(*) as count, 
                   MAX(timestamp) as latest
            FROM bars 
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY symbol, bar_type
        """, conn)
        
        print("\n📊 RECENT DATA (Last 24 hours):")
        if not bars_df.empty:
            print(bars_df.to_string(index=False))
        else:
            print("No recent data found")
    except:
        print("No bars table found")
    
    # Recent signals
    try:
        signals_df = pd.read_sql_query("""
            SELECT strategy_name, symbol, signal_type, 
                   confidence, timestamp
            FROM strategy_signals 
            WHERE timestamp > datetime('now', '-24 hours')
            ORDER BY timestamp DESC
            LIMIT 10
        """, conn)
        
        print("\n🎯 RECENT SIGNALS (Last 24 hours):")
        if not signals_df.empty:
            print(signals_df.to_string(index=False))
        else:
            print("No recent signals found")
    except:
        print("No signals table found")
    
    # Strategy performance
    try:
        perf_df = pd.read_sql_query("""
            SELECT strategy_name, COUNT(*) as trades,
                   AVG(return_pct) as avg_return,
                   SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
            FROM strategy_performance 
            GROUP BY strategy_name
        """, conn)
        
        print("\n📈 STRATEGY PERFORMANCE:")
        if not perf_df.empty:
            print(perf_df.to_string(index=False))
        else:
            print("No performance data found")
    except:
        print("No performance table found")
    
    print("\n" + "=" * 60)
    
    conn.close()

if __name__ == "__main__":
    show_dashboard()
EOF

chmod +x $INSTALL_DIR/dashboard.py

# Final setup message
echo -e "${GREEN}Setup completed successfully!${NC}"
echo ""
echo "🎉 Financial ML Pipeline installed on Raspberry Pi 5"
echo ""
echo "Next steps:"
echo "1. Review configuration: nano $INSTALL_DIR/config.yaml"
echo "2. Start data ingestion: financial-ml start"
echo "3. Check status: financial-ml status"
echo "4. View dashboard: python3 $INSTALL_DIR/dashboard.py"
echo ""
echo "Useful commands:"
echo "  financial-ml status    - Check service status"
echo "  financial-ml logs      - View live logs"  
echo "  financial-ml train     - Train ML models"
echo "  financial-ml discover  - Discover strategies"
echo "  financial-ml monitor   - System monitoring"
echo ""
echo "Data will be stored in: $DATA_DIR"
echo "Logs available at: $LOG_DIR"
echo ""
echo -e "${YELLOW}Note: The system will start collecting data immediately."
echo "Allow 2-4 weeks of data collection before running strategy discovery."
echo "Monitor CPU temperature and memory usage during initial setup.${NC}"

# Option to start service immediately
read -p "Start data ingestion service now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Starting data ingestion service...${NC}"
    sudo systemctl start financial-ml-ingestion
    sleep 3
    financial-ml status
fi

echo -e "${GREEN}Setup complete! 🚀${NC}"
