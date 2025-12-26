#!/bin/bash
# =============================================================================
# Financial ML Pipeline - Deployment Script
# =============================================================================
# 
# This script deploys the Financial ML Pipeline from a GitHub repository
# to a Hetzner Cloud VPS (or any Ubuntu 22.04+ server).
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/deploy.sh | sudo bash -s -- --repo YOUR_USER/YOUR_REPO
#
# Or download and run:
#   chmod +x deploy.sh
#   sudo ./deploy.sh --repo YOUR_USER/YOUR_REPO
#
# Options:
#   --repo OWNER/REPO    GitHub repository (required)
#   --branch BRANCH      Git branch to deploy (default: main)
#   --skip-setup         Skip system setup (if already done)
#   --skip-services      Skip systemd service installation
#   --dry-run            Show what would be done without making changes
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/financial_ml"
CONFIG_DIR="/etc/financial_ml"
LOG_DIR="/var/log/financial_ml"
DATA_DIR="/opt/financial_ml/data"
MODELS_DIR="/opt/financial_ml/models"
SCRIPTS_DIR="/opt/financial_ml/scripts"
BACKUP_DIR="/opt/financial_ml/backups"
SERVICE_USER="financial_ml"
SERVICE_GROUP="financial_ml"
SYSTEMD_DIR="/etc/systemd/system"

# Default options
GITHUB_REPO=""
GIT_BRANCH="main"
SKIP_SETUP=false
SKIP_SERVICES=false
DRY_RUN=false

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}==>${NC} ${GREEN}$1${NC}"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --repo)
                GITHUB_REPO="$2"
                shift 2
                ;;
            --branch)
                GIT_BRANCH="$2"
                shift 2
                ;;
            --skip-setup)
                SKIP_SETUP=true
                shift
                ;;
            --skip-services)
                SKIP_SERVICES=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$GITHUB_REPO" ]]; then
        log_error "GitHub repository is required. Use --repo OWNER/REPO"
        show_help
        exit 1
    fi
}

show_help() {
    cat << EOF
Financial ML Pipeline - Deployment Script

Usage: $0 --repo OWNER/REPO [OPTIONS]

Options:
    --repo OWNER/REPO    GitHub repository (required)
    --branch BRANCH      Git branch to deploy (default: main)
    --skip-setup         Skip system setup (if already done)
    --skip-services      Skip systemd service installation
    --dry-run            Show what would be done without making changes
    -h, --help           Show this help message

Examples:
    $0 --repo myuser/financial-ml-pipeline
    $0 --repo myuser/financial-ml-pipeline --branch develop
    $0 --repo myuser/financial-ml-pipeline --skip-setup
EOF
}

run_cmd() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

# =============================================================================
# System Setup Functions
# =============================================================================

check_system() {
    log_step "Checking system requirements"
    
    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        log_error "Cannot determine OS. This script requires Ubuntu 22.04+"
        exit 1
    fi
    
    source /etc/os-release
    if [[ "$ID" != "ubuntu" ]] && [[ "$ID" != "debian" ]]; then
        log_warn "This script is designed for Ubuntu/Debian. Proceeding anyway..."
    fi
    
    # Check memory
    TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_MEM_GB=$((TOTAL_MEM / 1024 / 1024))
    log_info "Available memory: ${TOTAL_MEM_GB}GB"
    
    if [[ $TOTAL_MEM -lt 7000000 ]]; then
        log_warn "Less than 7GB RAM detected. Some features may be limited."
    fi
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    log_info "CPU cores: $CPU_CORES"
    
    # Check disk space
    DISK_FREE=$(df -BG /opt 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    log_info "Free disk space in /opt: ${DISK_FREE}GB"
    
    if [[ ${DISK_FREE:-0} -lt 10 ]]; then
        log_warn "Less than 10GB free disk space. Consider freeing up space."
    fi
}

install_dependencies() {
    log_step "Installing system dependencies"
    
    # Add deadsnakes PPA for Python 3.11 (not in default Ubuntu 22.04 repos)
    log_info "Adding deadsnakes PPA for Python 3.11..."
    run_cmd apt update
    run_cmd apt install -y software-properties-common
    run_cmd add-apt-repository -y ppa:deadsnakes/ppa
    run_cmd apt update
    
    run_cmd apt install -y \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
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
        jq
    
    # Ensure pip is available for Python 3.11
    if ! command -v /usr/bin/python3.11 &> /dev/null; then
        log_error "Python 3.11 installation failed"
        exit 1
    fi
    
    log_info "Python 3.11 installed successfully"
}

install_talib() {
    log_step "Installing TA-Lib"
    
    if [[ -f /usr/local/lib/libta_lib.so ]]; then
        log_info "TA-Lib already installed, skipping..."
        return
    fi
    
    cd /tmp
    run_cmd wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    run_cmd tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    run_cmd ./configure --prefix=/usr/local
    run_cmd make
    run_cmd make install
    run_cmd ldconfig
    cd /
    rm -rf /tmp/ta-lib*
}

create_user() {
    log_step "Creating service user"
    
    if id "$SERVICE_USER" &>/dev/null; then
        log_info "User $SERVICE_USER already exists"
    else
        run_cmd useradd --system --no-create-home --shell /bin/false "$SERVICE_USER"
        log_info "Created user $SERVICE_USER"
    fi
}

create_directories() {
    log_step "Creating directory structure"
    
    local dirs=(
        "$INSTALL_DIR"
        "$DATA_DIR"
        "$MODELS_DIR"
        "$SCRIPTS_DIR"
        "$BACKUP_DIR"
        "$CONFIG_DIR"
        "$LOG_DIR"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            run_cmd mkdir -p "$dir"
            log_info "Created $dir"
        fi
    done
    
    # Set ownership
    run_cmd chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
    run_cmd chown -R "$SERVICE_USER:$SERVICE_GROUP" "$LOG_DIR"
}

# =============================================================================
# Deployment Functions
# =============================================================================

clone_repository() {
    log_step "Cloning repository from GitHub"
    
    local CLONE_DIR="/tmp/financial-ml-deploy-$$"
    
    # Clean up any existing clone
    rm -rf "$CLONE_DIR"
    
    log_info "Cloning https://github.com/${GITHUB_REPO}.git (branch: $GIT_BRANCH)"
    run_cmd git clone --depth 1 --branch "$GIT_BRANCH" \
        "https://github.com/${GITHUB_REPO}.git" "$CLONE_DIR"
    
    # Debug: Show repository structure
    log_info "Repository contents:"
    find "$CLONE_DIR" -type f -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.conf" -o -name "*.sh" 2>/dev/null | head -20
    
    echo "$CLONE_DIR"
}

find_file() {
    # Find a file anywhere in the clone directory
    local CLONE_DIR="$1"
    local FILENAME="$2"
    
    # First try exact path
    if [[ -f "$CLONE_DIR/$FILENAME" ]]; then
        echo "$CLONE_DIR/$FILENAME"
        return 0
    fi
    
    # Search recursively
    local FOUND=$(find "$CLONE_DIR" -type f -name "$FILENAME" 2>/dev/null | head -1)
    if [[ -n "$FOUND" ]]; then
        echo "$FOUND"
        return 0
    fi
    
    return 1
}

find_directory() {
    # Find a directory anywhere in the clone directory
    local CLONE_DIR="$1"
    local DIRNAME="$2"
    
    # First try exact path
    if [[ -d "$CLONE_DIR/$DIRNAME" ]]; then
        echo "$CLONE_DIR/$DIRNAME"
        return 0
    fi
    
    # Search recursively
    local FOUND=$(find "$CLONE_DIR" -type d -name "$DIRNAME" 2>/dev/null | head -1)
    if [[ -n "$FOUND" ]]; then
        echo "$FOUND"
        return 0
    fi
    
    return 1
}

setup_virtualenv() {
    log_step "Setting up Python virtual environment"
    
    if [[ ! -d "$INSTALL_DIR/venv" ]]; then
        run_cmd python3.11 -m venv "$INSTALL_DIR/venv"
        log_info "Created virtual environment"
    fi
    
    # Upgrade pip
    run_cmd "$INSTALL_DIR/venv/bin/pip" install --upgrade pip wheel setuptools
}

install_python_packages() {
    log_step "Installing Python packages"
    
    local CLONE_DIR="$1"
    
    # Find requirements.txt anywhere in the repo
    local REQUIREMENTS=$(find_file "$CLONE_DIR" "requirements.txt")
    
    # Also check for requirements_txt.txt (alternative name)
    if [[ -z "$REQUIREMENTS" ]]; then
        REQUIREMENTS=$(find_file "$CLONE_DIR" "requirements_txt.txt")
    fi
    
    if [[ -n "$REQUIREMENTS" ]] && [[ -f "$REQUIREMENTS" ]]; then
        log_info "Found requirements at: $REQUIREMENTS"
        run_cmd "$INSTALL_DIR/venv/bin/pip" install -r "$REQUIREMENTS"
    else
        log_warn "requirements.txt not found, installing core packages..."
        run_cmd "$INSTALL_DIR/venv/bin/pip" install \
            numpy pandas scipy scikit-learn statsmodels \
            aiosqlite websockets matplotlib seaborn \
            pyyaml python-dotenv loguru psutil tqdm click \
            joblib ta-lib xgboost lightgbm
    fi
}

deploy_python_modules() {
    log_step "Deploying Python modules"
    
    local CLONE_DIR="$1"
    
    # List of Python modules to deploy
    local MODULES=(
        "data_ingestion.py"
        "ml_pipeline.py"
        "strategy_discovery.py"
    )
    
    for module in "${MODULES[@]}"; do
        local FOUND_PATH=$(find_file "$CLONE_DIR" "$module")
        if [[ -n "$FOUND_PATH" ]]; then
            run_cmd cp "$FOUND_PATH" "$INSTALL_DIR/"
            log_info "Deployed $module (from $FOUND_PATH)"
        else
            log_warn "Module $module not found in repository"
        fi
    done
    
    # Create __init__.py if it doesn't exist
    if [[ ! -f "$INSTALL_DIR/__init__.py" ]]; then
        run_cmd touch "$INSTALL_DIR/__init__.py"
    fi
    
    # Set ownership
    run_cmd chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
}

deploy_config() {
    log_step "Deploying configuration"
    
    local CLONE_DIR="$1"
    
    # Look for config file with various names
    local CONFIG_FILES=("config.yaml" "config.yml" "config_yaml.txt")
    local CONFIG_FOUND=false
    
    for config_name in "${CONFIG_FILES[@]}"; do
        local FOUND_PATH=$(find_file "$CLONE_DIR" "$config_name")
        if [[ -n "$FOUND_PATH" ]]; then
            run_cmd cp "$FOUND_PATH" "$CONFIG_DIR/config.yaml"
            log_info "Deployed configuration from $FOUND_PATH"
            CONFIG_FOUND=true
            break
        fi
    done
    
    if [[ "$CONFIG_FOUND" == "false" ]]; then
        log_warn "No config file found, creating default configuration..."
        create_default_config
    fi
    
    run_cmd chown "$SERVICE_USER:$SERVICE_GROUP" "$CONFIG_DIR/config.yaml"
}

create_default_config() {
    cat > "$CONFIG_DIR/config.yaml" << 'EOFCONFIG'
# Financial ML Pipeline Configuration
# Generated by deploy.sh

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
    ping_interval: 30

  symbols:
    algorithmic:
      - "BTC/USD"
      - "ETH/USD"
    swing:
      - "BTC/USD"
      - "ETH/USD"
      - "ADA/USD"
      - "DOT/USD"

  bars:
    algorithmic:
      tick: 500
      volume: 5.0
      dollar: 50000
      dollar_imbalance: 25000
    swing:
      tick: 5000
      volume: 50.0
      dollar: 500000
      dollar_imbalance: 250000

  cusum_filter:
    algorithmic_threshold: 0.005
    swing_threshold: 0.02

machine_learning:
  models:
    random_forest:
      n_estimators: 1000
      max_depth: 10
      min_samples_leaf: 50
      n_jobs: -1

cross_validation:
  method: "purged_kfold"
  n_splits: 5
  embargo_pct: 0.02

paths:
  base_dir: "/opt/financial_ml"
  data_dir: "/opt/financial_ml/data"
  models_dir: "/opt/financial_ml/models"
  logs_dir: "/var/log/financial_ml"
EOFCONFIG
}

# =============================================================================
# Systemd Service Functions
# =============================================================================

parse_and_install_services() {
    log_step "Installing systemd services"
    
    local CLONE_DIR="$1"
    
    # Find services.conf anywhere in the repo
    local SERVICES_CONF=$(find_file "$CLONE_DIR" "services.conf")
    
    if [[ -z "$SERVICES_CONF" ]]; then
        # Also try looking in systemd directory
        local SYSTEMD_DIR_FOUND=$(find_directory "$CLONE_DIR" "systemd")
        if [[ -n "$SYSTEMD_DIR_FOUND" ]] && [[ -f "$SYSTEMD_DIR_FOUND/services.conf" ]]; then
            SERVICES_CONF="$SYSTEMD_DIR_FOUND/services.conf"
        fi
    fi
    
    if [[ -z "$SERVICES_CONF" ]] || [[ ! -f "$SERVICES_CONF" ]]; then
        log_error "services.conf not found in repository"
        log_info "Searched in: $CLONE_DIR"
        log_info "Repository contents:"
        find "$CLONE_DIR" -type f -name "*.conf" 2>/dev/null || echo "  No .conf files found"
        return 1
    fi
    
    log_info "Found services.conf at: $SERVICES_CONF"
    
    # Parse services.conf and create individual unit files
    local current_file=""
    local content=""
    local in_file=false
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Check for file marker
        if [[ "$line" =~ ^###\ FILE:\ (.+)\ ###$ ]]; then
            # Save previous file if exists
            if [[ -n "$current_file" ]] && [[ -n "$content" ]]; then
                echo "$content" > "$SYSTEMD_DIR/$current_file"
                log_info "Created $current_file"
            fi
            
            # Start new file
            current_file="${BASH_REMATCH[1]}"
            content=""
            in_file=true
        elif [[ "$in_file" == "true" ]]; then
            # Skip comment lines at the start of sections
            if [[ "$line" =~ ^#\ ===|^#\ $ ]] && [[ -z "$content" ]]; then
                continue
            fi
            # Add line to content
            content+="$line"$'\n'
        fi
    done < "$SERVICES_CONF"
    
    # Save last file
    if [[ -n "$current_file" ]] && [[ -n "$content" ]]; then
        echo "$content" > "$SYSTEMD_DIR/$current_file"
        log_info "Created $current_file"
    fi
    
    # Reload systemd
    run_cmd systemctl daemon-reload
}

create_utility_scripts() {
    log_step "Creating utility scripts"
    
    # Create backup script
    cat > "$SCRIPTS_DIR/backup.sh" << 'EOFBACKUP'
#!/bin/bash
BACKUP_DIR="/opt/financial_ml/backups"
DATA_DIR="/opt/financial_ml/data"
RETENTION_DAYS=7

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/financial_data_$DATE.db.gz"

# Create backup
sqlite3 "$DATA_DIR/financial_data.db" ".backup '/tmp/backup.db'"
gzip -c /tmp/backup.db > "$BACKUP_FILE"
rm /tmp/backup.db

# Remove old backups
find "$BACKUP_DIR" -name "*.db.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup created: $BACKUP_FILE"
EOFBACKUP
    
    # Create health check script
    cat > "$SCRIPTS_DIR/health-check.sh" << 'EOFHEALTH'
#!/bin/bash
# Health check script for Financial ML Pipeline

LOG_FILE="/var/log/financial_ml/health.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

check_service() {
    local service=$1
    if systemctl is-active --quiet "$service"; then
        echo "[$TIMESTAMP] $service: OK" >> "$LOG_FILE"
        return 0
    else
        echo "[$TIMESTAMP] $service: FAILED" >> "$LOG_FILE"
        return 1
    fi
}

check_database() {
    local db_path="/opt/financial_ml/data/financial_data.db"
    if [[ -f "$db_path" ]]; then
        local result=$(sqlite3 "$db_path" "SELECT 1" 2>&1)
        if [[ "$result" == "1" ]]; then
            echo "[$TIMESTAMP] Database: OK" >> "$LOG_FILE"
            return 0
        fi
    fi
    echo "[$TIMESTAMP] Database: FAILED" >> "$LOG_FILE"
    return 1
}

check_disk_space() {
    local threshold=90
    local usage=$(df /opt/financial_ml | tail -1 | awk '{print $5}' | tr -d '%')
    if [[ $usage -lt $threshold ]]; then
        echo "[$TIMESTAMP] Disk space: OK ($usage%)" >> "$LOG_FILE"
        return 0
    else
        echo "[$TIMESTAMP] Disk space: WARNING ($usage%)" >> "$LOG_FILE"
        return 1
    fi
}

check_memory() {
    local threshold=90
    local usage=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100)}')
    if [[ $usage -lt $threshold ]]; then
        echo "[$TIMESTAMP] Memory: OK ($usage%)" >> "$LOG_FILE"
        return 0
    else
        echo "[$TIMESTAMP] Memory: WARNING ($usage%)" >> "$LOG_FILE"
        return 1
    fi
}

# Run checks
ERRORS=0
check_service "financial-ml-ingestion" || ((ERRORS++))
check_database || ((ERRORS++))
check_disk_space || ((ERRORS++))
check_memory || ((ERRORS++))

exit $ERRORS
EOFHEALTH
    
    # Create CLI wrapper
    cat > "$SCRIPTS_DIR/financial-ml" << 'EOFCLI'
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
    "restart")
        echo "Restarting data ingestion service..."
        sudo systemctl restart financial-ml-ingestion
        ;;
    "status")
        echo "=== Service Status ==="
        echo ""
        echo "Data Ingestion:"
        systemctl status financial-ml-ingestion --no-pager -l 2>/dev/null || echo "  Not running"
        echo ""
        echo "=== Active Timers ==="
        systemctl list-timers 'financial-ml-*' --no-pager 2>/dev/null || echo "  No timers found"
        echo ""
        echo "=== Recent Logs ==="
        journalctl -u financial-ml-ingestion -n 10 --no-pager 2>/dev/null || echo "  No logs found"
        ;;
    "logs")
        local service="${2:-ingestion}"
        journalctl -u "financial-ml-$service" -f
        ;;
    "train")
        echo "Starting model training..."
        shift
        cd $PROJECT_DIR && $VENV_PYTHON -m ml_pipeline --mode train "$@"
        ;;
    "train-algo")
        echo "Starting algorithmic model training..."
        cd $PROJECT_DIR && $VENV_PYTHON -m ml_pipeline --mode train --timeframe algorithmic "$@"
        ;;
    "train-swing")
        echo "Starting swing model training..."
        cd $PROJECT_DIR && $VENV_PYTHON -m ml_pipeline --mode train --timeframe swing "$@"
        ;;
    "discover")
        echo "Starting strategy discovery..."
        shift
        cd $PROJECT_DIR && $VENV_PYTHON -m strategy_discovery discover "$@"
        ;;
    "signals")
        echo "Generating trading signals..."
        shift
        cd $PROJECT_DIR && $VENV_PYTHON -m strategy_discovery signals "$@"
        ;;
    "summary")
        echo "Strategy summary..."
        cd $PROJECT_DIR && $VENV_PYTHON -m strategy_discovery summary
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
        echo "=== Database Size ==="
        du -h $PROJECT_DIR/data/*.db 2>/dev/null || echo "  No database found"
        echo ""
        echo "=== Model Files ==="
        ls -lh $PROJECT_DIR/models/*.joblib 2>/dev/null || echo "  No models found"
        ;;
    "backup")
        echo "Running backup..."
        $PROJECT_DIR/scripts/backup.sh
        ;;
    "health")
        echo "Running health check..."
        $PROJECT_DIR/scripts/health-check.sh
        cat /var/log/financial_ml/health.log | tail -20
        ;;
    "timers")
        echo "=== Scheduled Tasks ==="
        systemctl list-timers 'financial-ml-*' --all
        ;;
    *)
        echo "Financial ML Pipeline CLI"
        echo ""
        echo "Usage: $0 COMMAND [OPTIONS]"
        echo ""
        echo "Service Commands:"
        echo "  start           Start data ingestion service"
        echo "  stop            Stop data ingestion service"
        echo "  restart         Restart data ingestion service"
        echo "  status          Show service status and recent logs"
        echo "  logs [service]  Follow service logs (default: ingestion)"
        echo "  timers          Show scheduled tasks"
        echo ""
        echo "Training Commands:"
        echo "  train           Run model training (generic)"
        echo "  train-algo      Run algorithmic model training"
        echo "  train-swing     Run swing model training"
        echo ""
        echo "Strategy Commands:"
        echo "  discover        Run strategy discovery"
        echo "  signals         Generate trading signals"
        echo "  summary         Show strategy summary"
        echo ""
        echo "Maintenance Commands:"
        echo "  monitor         Show system monitoring info"
        echo "  backup          Run database backup"
        echo "  health          Run health check"
        ;;
esac
EOFCLI
    
    # Make scripts executable
    chmod +x "$SCRIPTS_DIR/backup.sh"
    chmod +x "$SCRIPTS_DIR/health-check.sh"
    chmod +x "$SCRIPTS_DIR/financial-ml"
    
    # Create symlink to CLI
    ln -sf "$SCRIPTS_DIR/financial-ml" /usr/local/bin/financial-ml
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$SCRIPTS_DIR"
}

enable_services() {
    log_step "Enabling systemd services and timers"
    
    # Enable main service
    run_cmd systemctl enable financial-ml-ingestion
    
    # Enable all timers
    local TIMERS=(
        "financial-ml-training-algo.timer"
        "financial-ml-training-swing.timer"
        "financial-ml-signals-algo.timer"
        "financial-ml-signals-swing.timer"
        "financial-ml-discovery-algo.timer"
        "financial-ml-discovery-swing.timer"
        "financial-ml-backup.timer"
        "financial-ml-health.timer"
    )
    
    for timer in "${TIMERS[@]}"; do
        if [[ -f "$SYSTEMD_DIR/$timer" ]]; then
            run_cmd systemctl enable "$timer"
            run_cmd systemctl start "$timer"
            log_info "Enabled $timer"
        fi
    done
}

initialize_database() {
    log_step "Initializing database"
    
    local DB_PATH="$DATA_DIR/financial_data.db"
    
    if [[ -f "$DB_PATH" ]]; then
        log_info "Database already exists at $DB_PATH"
        return
    fi
    
    run_cmd sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/python" -c "
import sqlite3
import os

db_path = '$DB_PATH'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Enable WAL mode
cursor.execute('PRAGMA journal_mode=WAL;')
cursor.execute('PRAGMA synchronous=NORMAL;')
cursor.execute('PRAGMA cache_size=200000;')
cursor.execute('PRAGMA mmap_size=268435456;')

# Create tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        price REAL NOT NULL,
        volume REAL NOT NULL,
        side TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS bars (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        bar_type TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        vwap REAL,
        dollar_volume REAL,
        trade_count INTEGER
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        event_type TEXT NOT NULL,
        price REAL NOT NULL,
        metadata TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        strategy_name TEXT NOT NULL,
        strategy_type TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        model_prediction INTEGER,
        position_size REAL,
        risk_score REAL,
        feature_values TEXT,
        metadata TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS strategy_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_name TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        pnl REAL,
        return_pct REAL,
        position_size REAL,
        hold_time_hours REAL
    )
''')

# Create indexes
cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_bars_timestamp ON bars(timestamp)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_bars_symbol_type ON bars(symbol, bar_type, timeframe)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)')

conn.commit()
conn.close()
print('Database initialized successfully')
"
    
    log_info "Database initialized at $DB_PATH"
}

setup_logrotate() {
    log_step "Setting up log rotation"
    
    cat > /etc/logrotate.d/financial-ml << EOF
/var/log/financial_ml/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 $SERVICE_USER $SERVICE_GROUP
    sharedscripts
    postrotate
        systemctl reload financial-ml-ingestion 2>/dev/null || true
    endscript
}
EOF
    
    log_info "Log rotation configured"
}

# =============================================================================
# Verification Functions
# =============================================================================

verify_deployment() {
    log_step "Verifying deployment"
    
    local ERRORS=0
    
    # Check directories
    for dir in "$INSTALL_DIR" "$DATA_DIR" "$CONFIG_DIR" "$LOG_DIR"; do
        if [[ -d "$dir" ]]; then
            echo -e "  ${GREEN}✓${NC} Directory exists: $dir"
        else
            echo -e "  ${RED}✗${NC} Directory missing: $dir"
            ((ERRORS++))
        fi
    done
    
    # Check Python modules
    for module in data_ingestion.py ml_pipeline.py strategy_discovery.py; do
        if [[ -f "$INSTALL_DIR/$module" ]]; then
            echo -e "  ${GREEN}✓${NC} Module exists: $module"
        else
            echo -e "  ${RED}✗${NC} Module missing: $module"
            ((ERRORS++))
        fi
    done
    
    # Check config
    if [[ -f "$CONFIG_DIR/config.yaml" ]]; then
        echo -e "  ${GREEN}✓${NC} Configuration exists"
    else
        echo -e "  ${RED}✗${NC} Configuration missing"
        ((ERRORS++))
    fi
    
    # Check virtual environment
    if [[ -f "$INSTALL_DIR/venv/bin/python" ]]; then
        echo -e "  ${GREEN}✓${NC} Virtual environment exists"
    else
        echo -e "  ${RED}✗${NC} Virtual environment missing"
        ((ERRORS++))
    fi
    
    # Check systemd services
    for service in financial-ml-ingestion.service financial-ml-training-algo.timer; do
        if [[ -f "$SYSTEMD_DIR/$service" ]]; then
            echo -e "  ${GREEN}✓${NC} Service exists: $service"
        else
            echo -e "  ${RED}✗${NC} Service missing: $service"
            ((ERRORS++))
        fi
    done
    
    # Check CLI
    if [[ -f /usr/local/bin/financial-ml ]]; then
        echo -e "  ${GREEN}✓${NC} CLI installed"
    else
        echo -e "  ${RED}✗${NC} CLI missing"
        ((ERRORS++))
    fi
    
    return $ERRORS
}

print_summary() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Directory structure:"
    echo "  Base:      $INSTALL_DIR"
    echo "  Data:      $DATA_DIR"
    echo "  Models:    $MODELS_DIR"
    echo "  Config:    $CONFIG_DIR/config.yaml"
    echo "  Logs:      $LOG_DIR"
    echo "  Backups:   $BACKUP_DIR"
    echo ""
    echo "Pipeline Schedule:"
    echo "  Algorithmic:"
    echo "    - Training:  Daily at 2:00 AM"
    echo "    - Signals:   Hourly"
    echo "    - Discovery: Weekly (Sunday 3:00 AM)"
    echo "  Swing:"
    echo "    - Training:  Weekly (Wednesday 3:00 AM)"
    echo "    - Signals:   Daily at 6:00 AM"
    echo "    - Discovery: Monthly (1st Sunday 4:00 AM)"
    echo ""
    echo "Quick Start Commands:"
    echo "  financial-ml start     Start data ingestion"
    echo "  financial-ml status    Check status"
    echo "  financial-ml logs      View logs"
    echo "  financial-ml monitor   System monitoring"
    echo "  financial-ml timers    View scheduled tasks"
    echo ""
    echo "To start collecting data:"
    echo "  sudo systemctl start financial-ml-ingestion"
    echo ""
    echo "To view logs:"
    echo "  journalctl -u financial-ml-ingestion -f"
    echo ""
    echo -e "${YELLOW}Note: Data collection should run for 3+ months before${NC}"
    echo -e "${YELLOW}training algorithmic models (12+ months for swing).${NC}"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║     Financial ML Pipeline - Deployment Script             ║"
    echo "║     Optimized for Hetzner Cloud                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_root
    parse_args "$@"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "Running in dry-run mode - no changes will be made"
    fi
    
    # System setup
    check_system
    
    if [[ "$SKIP_SETUP" != "true" ]]; then
        install_dependencies
        install_talib
        create_user
    fi
    
    create_directories
    
    # Clone and deploy
    CLONE_DIR=$(clone_repository)
    
    setup_virtualenv
    install_python_packages "$CLONE_DIR"
    deploy_python_modules "$CLONE_DIR"
    deploy_config "$CLONE_DIR"
    
    # Systemd services
    if [[ "$SKIP_SERVICES" != "true" ]]; then
        parse_and_install_services "$CLONE_DIR"
        create_utility_scripts
        enable_services
    fi
    
    # Initialize
    initialize_database
    setup_logrotate
    
    # Cleanup
    rm -rf "$CLONE_DIR"
    
    # Verify
    verify_deployment
    
    print_summary
}

main "$@"
