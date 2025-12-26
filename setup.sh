#!/bin/bash
# Financial ML Pipeline Setup Script for Hetzner Cloud
# Optimized for CPX21 (4 vCPU, 8GB RAM, Ubuntu 22.04+)
# Run as: curl -sSL https://raw.githubusercontent.com/your-repo/setup.sh | bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
INSTALL_DIR="/opt/financial_ml"
DATA_DIR="/opt/financial_ml/data"
MODELS_DIR="/opt/financial_ml/models"
LOG_DIR="/var/log/financial_ml"
BACKUP_DIR="/opt/financial_ml/backups"
SERVICE_USER="financial_ml"
SERVICE_GROUP="financial_ml"
PYTHON_VERSION="3.11"

echo -e "${GREEN}Financial ML Pipeline Setup for Hetzner Cloud${NC}"
echo "=================================================="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi

# Detect OS
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
else
    echo -e "${RED}Cannot detect OS${NC}"
    exit 1
fi

echo -e "${GREEN}Detected OS: $OS $VERSION${NC}"

# Check system resources
TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
CPU_CORES=$(nproc)

echo -e "${GREEN}System Resources:${NC}"
echo "  CPU Cores: $CPU_CORES"
echo "  Memory: $((TOTAL_MEM / 1024)) MB"

if [ $TOTAL_MEM -lt 7000000 ]; then
    echo -e "${YELLOW}Warning: Less than 7GB RAM detected. Some features may be limited.${NC}"
fi

# Update system
echo -e "${GREEN}Updating system packages...${NC}"
apt update && apt upgrade -y

# Install system dependencies
echo -e "${GREEN}Installing system dependencies...${NC}"
apt install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
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
    wget \
    htop \
    iotop \
    jq \
    unzip \
    nginx \
    certbot \
    python3-certbot-nginx

# Install TA-Lib from source
echo -e "${GREEN}Installing TA-Lib...${NC}"
if [ ! -f /usr/local/lib/libta_lib.so ]; then
    cd /tmp
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr/local
    make -j$(nproc)
    make install
    ldconfig
    cd /
    rm -rf /tmp/ta-lib*
fi

# Create service user
echo -e "${GREEN}Creating service user...${NC}"
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -s /bin/false -d $INSTALL_DIR $SERVICE_USER
fi

# Create directories
echo -e "${GREEN}Creating directories...${NC}"
mkdir -p $INSTALL_DIR
mkdir -p $DATA_DIR
mkdir -p $MODELS_DIR
mkdir -p $LOG_DIR
mkdir -p $BACKUP_DIR
mkdir -p $INSTALL_DIR/scripts
mkdir -p /etc/financial_ml

# Set permissions
chown -R $SERVICE_USER:$SERVICE_GROUP $INSTALL_DIR
chown -R $SERVICE_USER:$SERVICE_GROUP $LOG_DIR
chmod 755 $INSTALL_DIR
chmod 755 $LOG_DIR

# Create Python virtual environment
echo -e "${GREEN}Creating Python virtual environment...${NC}"
cd $INSTALL_DIR
python${PYTHON_VERSION} -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install Python packages
echo -e "${GREEN}Installing Python packages...${NC}"
cat > requirements.txt << 'EOF'
# Core
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
xgboost>=2.0.0
lightgbm>=4.0.0

# Async
aiosqlite>=0.19.0
websockets>=12.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Config and logging
pyyaml>=6.0
python-dotenv>=1.0.0
loguru>=0.7.0

# System
psutil>=5.9.0
joblib>=1.3.0

# Analysis
arch>=6.2.0
ta-lib>=0.4.25

# Utils
tqdm>=4.65.0
click>=8.1.0
schedule>=1.2.0
python-dateutil>=2.8.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Optional
numba>=0.58.0
httpx>=0.25.0
EOF

pip install -r requirements.txt

# Download project files (replace with your actual repo)
echo -e "${GREEN}Setting up project files...${NC}"
# git clone https://github.com/your-repo/financial-ml-pipeline.git .

# Create configuration file
echo -e "${GREEN}Creating configuration...${NC}"
cat > config.yaml << 'EOF'
# Auto-generated configuration for Hetzner Cloud
system:
  max_memory_usage: 7.0
  max_cpu_cores: 4
  log_level: "INFO"
  log_file: "/var/log/financial_ml/pipeline.log"

database:
  type: "sqlite"
  sqlite:
    path: "/opt/financial_ml/data/financial_data.db"
    journal_mode: "WAL"
    synchronous: "NORMAL"
    cache_size: 200000

data_ingestion:
  websocket:
    uri: "wss://ws.kraken.com/v2"
    reconnect_delay: 5.0
    max_reconnect_attempts: 20
  symbols:
    algorithmic:
      - "BTC/USD"
      - "ETH/USD"
  bars:
    algorithmic:
      tick: 500
      volume: 5.0
      dollar: 50000
      dollar_imbalance: 25000
  cusum_filter:
    algorithmic_threshold: 0.005

paths:
  base_dir: "/opt/financial_ml"
  data_dir: "/opt/financial_ml/data"
  models_dir: "/opt/financial_ml/models"
  logs_dir: "/var/log/financial_ml"
EOF

cp config.yaml /etc/financial_ml/config.yaml

# Initialize database
echo -e "${GREEN}Initializing database...${NC}"
$INSTALL_DIR/venv/bin/python -c "
import sqlite3
import os

db_path = '/opt/financial_ml/data/financial_data.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('PRAGMA journal_mode=WAL;')
cursor.execute('PRAGMA synchronous=NORMAL;')
cursor.execute('PRAGMA cache_size=200000;')
cursor.execute('PRAGMA temp_store=MEMORY;')

print('Database initialized successfully')
conn.close()
"

# Set database permissions
chown $SERVICE_USER:$SERVICE_GROUP $DATA_DIR/financial_data.db

# Create systemd services
echo -e "${GREEN}Setting up systemd services...${NC}"

# Data ingestion service
cat > /etc/systemd/system/financial-ml-ingestion.service << EOF
[Unit]
Description=Financial ML Data Ingestion Service
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python -m data_ingestion
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=financial-ml-ingestion
MemoryLimit=4G
CPUQuota=200%
Environment=PYTHONPATH=$INSTALL_DIR
Environment=PYTHONUNBUFFERED=1
Environment=CONFIG_PATH=/etc/financial_ml/config.yaml

[Install]
WantedBy=multi-user.target
EOF

# Training service
cat > /etc/systemd/system/financial-ml-training.service << EOF
[Unit]
Description=Financial ML Training Service
After=financial-ml-ingestion.service

[Service]
Type=oneshot
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python -m ml_pipeline --mode train
StandardOutput=journal
StandardError=journal
TimeoutSec=7200
MemoryLimit=6G
Environment=PYTHONPATH=$INSTALL_DIR
Environment=CONFIG_PATH=/etc/financial_ml/config.yaml
EOF

# Training timer
cat > /etc/systemd/system/financial-ml-training.timer << EOF
[Unit]
Description=Run Financial ML Training Daily

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

# Create CLI wrapper
cat > $INSTALL_DIR/scripts/financial-ml << 'EOFCLI'
#!/bin/bash
PROJECT_DIR="/opt/financial_ml"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
export PYTHONPATH=$PROJECT_DIR
export CONFIG_PATH=/etc/financial_ml/config.yaml

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
        journalctl -u financial-ml-ingestion -n 20 --no-pager
        ;;
    "logs")
        journalctl -u financial-ml-ingestion -f
        ;;
    "train")
        echo "Starting model training..."
        cd $PROJECT_DIR && $VENV_PYTHON -m ml_pipeline --mode train "$@"
        ;;
    "discover")
        echo "Starting strategy discovery..."
        cd $PROJECT_DIR && $VENV_PYTHON -m strategy_discovery discover "${@:2}"
        ;;
    "signals")
        echo "Generating trading signals..."
        cd $PROJECT_DIR && $VENV_PYTHON -m strategy_discovery signals "${@:2}"
        ;;
    "monitor")
        echo "=== System Resources ==="
        free -h
        echo ""
        echo "=== CPU Usage ==="
        top -bn1 | head -5
        echo ""
        echo "=== Disk Usage ==="
        df -h $PROJECT_DIR
        echo ""
        echo "=== Service Status ==="
        systemctl is-active financial-ml-ingestion
        ;;
    "backup")
        echo "Running backup..."
        $PROJECT_DIR/scripts/backup.sh
        ;;
    *)
        echo "Financial ML Pipeline CLI"
        echo "Usage: $0 {start|stop|status|logs|train|discover|signals|monitor|backup}"
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
        echo "  backup    - Run database backup"
        ;;
esac
EOFCLI

chmod +x $INSTALL_DIR/scripts/financial-ml
ln -sf $INSTALL_DIR/scripts/financial-ml /usr/local/bin/financial-ml

# Create backup script
cat > $INSTALL_DIR/scripts/backup.sh << 'EOFBACKUP'
#!/bin/bash
BACKUP_DIR="/opt/financial_ml/backups"
DATA_DIR="/opt/financial_ml/data"
RETENTION_DAYS=7

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/financial_data_$DATE.db.gz"

# Create backup
sqlite3 $DATA_DIR/financial_data.db ".backup '/tmp/backup.db'"
gzip -c /tmp/backup.db > $BACKUP_FILE
rm /tmp/backup.db

# Remove old backups
find $BACKUP_DIR -name "*.db.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup created: $BACKUP_FILE"
EOFBACKUP

chmod +x $INSTALL_DIR/scripts/backup.sh

# Set final permissions
chown -R $SERVICE_USER:$SERVICE_GROUP $INSTALL_DIR
chown -R $SERVICE_USER:$SERVICE_GROUP $LOG_DIR

# Reload systemd
systemctl daemon-reload

# Enable services
echo -e "${GREEN}Enabling services...${NC}"
systemctl enable financial-ml-ingestion
systemctl enable financial-ml-training.timer

# Setup log rotation
cat > /etc/logrotate.d/financial-ml << EOF
/var/log/financial_ml/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 $SERVICE_USER $SERVICE_GROUP
}
EOF

# Setup firewall (if ufw is installed)
if command -v ufw &> /dev/null; then
    echo -e "${GREEN}Configuring firewall...${NC}"
    ufw allow 22/tcp
    ufw --force enable
fi

# Print summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Directory structure:"
echo "  Base:    $INSTALL_DIR"
echo "  Data:    $DATA_DIR"
echo "  Models:  $MODELS_DIR"
echo "  Logs:    $LOG_DIR"
echo "  Backups: $BACKUP_DIR"
echo ""
echo "Commands:"
echo "  financial-ml start    - Start data ingestion"
echo "  financial-ml stop     - Stop data ingestion"
echo "  financial-ml status   - Check status"
echo "  financial-ml train    - Train models"
echo "  financial-ml discover - Discover strategies"
echo "  financial-ml signals  - Generate signals"
echo ""
echo "To start the service:"
echo "  sudo systemctl start financial-ml-ingestion"
echo ""
echo "To view logs:"
echo "  journalctl -u financial-ml-ingestion -f"
echo ""
echo -e "${YELLOW}Note: Copy your Python code to $INSTALL_DIR before starting${NC}"
