#!/usr/bin/env bash
###############################################################################
# HRP Interactive Setup Script
#
# Bootstraps the entire HRP environment on a new machine:
#   - Python venv + dependencies
#   - Directory structure
#   - .env configuration
#   - Database initialization
#   - Config file fixes (launchd plists, .mcp.json)
#   - Optional: dashboard auth, data bootstrap, launchd jobs
#
# Usage:
#   ./scripts/setup.sh              # Interactive setup
#   ./scripts/setup.sh --check      # Verification only (Phase 10)
#
# Safe to re-run: skips completed steps.
###############################################################################

set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Project root
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly DATA_DIR="${HRP_DATA_DIR:-$HOME/hrp-data}"

# Counters for final summary
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

###############################################################################
# Utility Functions
###############################################################################

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

log_phase() {
    echo ""
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  Phase $1: $2${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

ask_yes_no() {
    local prompt="$1"
    local default="${2:-y}"
    local yn_hint
    if [[ "$default" == "y" ]]; then
        yn_hint="[Y/n]"
    else
        yn_hint="[y/N]"
    fi

    while true; do
        echo -en "${BOLD}$prompt $yn_hint: ${NC}" > /dev/tty
        read -r answer < /dev/tty || answer=""
        answer="${answer:-$default}"
        case "${answer,,}" in
            y|yes) return 0 ;;
            n|no)  return 1 ;;
            *)     echo "Please answer y or n." > /dev/tty ;;
        esac
    done
}

ask_input() {
    local prompt="$1"
    local default="${2:-}"
    local show_default=""
    if [[ -n "$default" ]]; then
        show_default=" [${default}]"
    fi
    echo -en "${BOLD}${prompt}${show_default}: ${NC}" > /dev/tty
    read -r answer < /dev/tty || answer=""
    echo "${answer:-$default}"
}

ask_secret() {
    local prompt="$1"
    local default="${2:-}"
    echo -en "${BOLD}${prompt}: ${NC}" > /dev/tty
    read -rs answer < /dev/tty || answer=""
    echo "" > /dev/tty
    echo "${answer:-$default}"
}

check_result() {
    local description="$1"
    local status="$2"  # pass, fail, skip
    case "$status" in
        pass)
            echo -e "  ${GREEN}✓${NC} $description"
            ((PASS_COUNT++)) || true
            ;;
        fail)
            echo -e "  ${RED}✗${NC} $description"
            ((FAIL_COUNT++)) || true
            ;;
        skip)
            echo -e "  ${YELLOW}~${NC} $description (skipped)"
            ((SKIP_COUNT++)) || true
            ;;
    esac
}

###############################################################################
# Phase 0: Pre-flight Checks
###############################################################################

phase_preflight() {
    log_phase "0" "Pre-flight Checks"

    # Check we're in the HRP repo
    if [[ ! -f "$PROJECT_ROOT/hrp/api/platform.py" ]]; then
        log_error "Not in HRP repository. Run from the project root."
        exit 1
    fi
    log_success "HRP repository detected at $PROJECT_ROOT"

    # Check OS
    local os_name
    os_name="$(uname -s)"
    if [[ "$os_name" == "Darwin" ]]; then
        log_success "macOS detected ($(sw_vers -productVersion 2>/dev/null || echo 'unknown'))"
    elif [[ "$os_name" == "Linux" ]]; then
        log_success "Linux detected"
    else
        log_warning "Unsupported OS: $os_name. Some features may not work."
    fi

    # Check Python
    local python_cmd=""
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")"
            local major minor
            major="${ver%%.*}"
            minor="${ver#*.}"
            if [[ "$major" -ge 3 ]] && [[ "$minor" -ge 11 ]]; then
                python_cmd="$cmd"
                log_success "Python $ver found ($cmd)"
                if [[ "$minor" -ge 14 ]]; then
                    log_warning "Python 3.14+ detected — some dependencies may not have wheels yet"
                fi
                break
            fi
        fi
    done

    if [[ -z "$python_cmd" ]]; then
        log_error "Python 3.11+ is required but not found."
        log_info "Install via: brew install python@3.11"
        exit 1
    fi
    PYTHON_CMD="$python_cmd"

    # Check for uv (fast installer) or pip
    if command -v uv &>/dev/null; then
        log_success "uv package manager detected ($(uv --version 2>/dev/null || echo 'unknown'))"
        USE_UV=true
    else
        log_info "uv not found — will use pip (install uv for faster installs: brew install uv)"
        USE_UV=false
    fi

    # Check for Homebrew (macOS)
    if [[ "$(uname -s)" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            log_success "Homebrew detected"
            HAS_BREW=true
        else
            log_warning "Homebrew not found. Install from https://brew.sh"
            HAS_BREW=false
        fi
    else
        HAS_BREW=false
    fi
}

###############################################################################
# Phase 1: System Dependencies
###############################################################################

phase_system_deps() {
    log_phase "1" "System Dependencies"

    if [[ "$(uname -s)" != "Darwin" ]]; then
        log_info "Skipping macOS-specific system deps on Linux"
        return 0
    fi

    # Check libomp (needed for LightGBM/XGBoost)
    if brew list libomp &>/dev/null 2>&1; then
        log_success "libomp already installed (required by LightGBM/XGBoost)"
    elif [[ "$HAS_BREW" == true ]]; then
        if ask_yes_no "Install libomp via Homebrew? (required for LightGBM/XGBoost)" "y"; then
            log_info "Installing libomp..."
            brew install libomp
            log_success "libomp installed"
        else
            log_warning "Skipped libomp — LightGBM/XGBoost may fail to import"
        fi
    else
        log_warning "Cannot install libomp without Homebrew. LightGBM/XGBoost may fail."
    fi
}

###############################################################################
# Phase 2: Python Environment
###############################################################################

phase_python_env() {
    log_phase "2" "Python Environment"

    local venv_dir="$PROJECT_ROOT/.venv"

    # Create venv if needed
    if [[ -d "$venv_dir" ]] && [[ -f "$venv_dir/bin/python" ]]; then
        log_success "Virtual environment already exists at .venv/"
    else
        log_info "Creating virtual environment..."
        "$PYTHON_CMD" -m venv "$venv_dir"
        log_success "Virtual environment created"
    fi

    # Activate
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    log_success "Activated .venv ($(python --version))"

    # Install core dependencies
    log_info "Installing core + dev dependencies..."
    if [[ "$USE_UV" == true ]]; then
        uv pip install -e ".[dev]"
    else
        pip install -e ".[dev]"
    fi
    log_success "Core + dev dependencies installed"

    # Ask about optional extras
    if ask_yes_no "Install ops dependencies (health server, monitoring)?" "y"; then
        log_info "Installing ops dependencies..."
        if [[ "$USE_UV" == true ]]; then
            uv pip install -e ".[ops]"
        else
            pip install -e ".[ops]"
        fi
        log_success "Ops dependencies installed"
    fi

    if ask_yes_no "Install trading dependencies (IBKR integration)?" "n"; then
        log_info "Installing trading dependencies..."
        if [[ "$USE_UV" == true ]]; then
            uv pip install -e ".[trading]"
        else
            pip install -e ".[trading]"
        fi
        log_success "Trading dependencies installed"
    fi

    # Verify critical imports
    log_info "Verifying critical imports..."
    local import_failures=0
    for pkg in duckdb pandas numpy vectorbt mlflow lightgbm xgboost streamlit empyrical scikit-learn; do
        # Map package names to Python import names
        local import_name="$pkg"
        case "$pkg" in
            scikit-learn) import_name="sklearn" ;;
        esac
        if python -c "import $import_name" 2>/dev/null; then
            log_success "  import $import_name"
        else
            log_error "  import $import_name FAILED"
            ((import_failures++)) || true
        fi
    done

    if [[ "$import_failures" -gt 0 ]]; then
        log_warning "$import_failures import(s) failed. Some features may not work."
    else
        log_success "All critical imports verified"
    fi
}

###############################################################################
# Phase 3: Directory Structure
###############################################################################

phase_directories() {
    log_phase "3" "Directory Structure"

    local dirs=(
        "$DATA_DIR"
        "$DATA_DIR/logs"
        "$DATA_DIR/auth"
        "$DATA_DIR/optuna"
        "$DATA_DIR/cache"
        "$DATA_DIR/output"
        "$DATA_DIR/backups"
        "$DATA_DIR/config"
        "$DATA_DIR/mlflow"
    )

    local created=0
    for dir in "${dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            log_success "  $dir (exists)"
        else
            mkdir -p "$dir"
            log_success "  $dir (created)"
            ((created++)) || true
        fi
    done

    if [[ "$created" -eq 0 ]]; then
        log_success "All directories already exist"
    else
        log_success "Created $created new directories"
    fi
}

###############################################################################
# Phase 4: .env Configuration
###############################################################################

phase_env_config() {
    log_phase "4" ".env Configuration"

    local env_file="$PROJECT_ROOT/.env"

    if [[ -f "$env_file" ]]; then
        log_success ".env file already exists"
        if ! ask_yes_no "Overwrite existing .env? (current file will be backed up)" "n"; then
            log_info "Keeping existing .env"
            return 0
        fi
        cp "$env_file" "${env_file}.bak.$(date +%Y%m%d%H%M%S)"
        log_info "Backed up existing .env"
    fi

    log_info "Let's configure your environment. Press Enter to accept defaults."
    echo ""

    # Environment
    local hrp_env
    hrp_env=$(ask_input "HRP_ENVIRONMENT (development/staging/production)" "development")

    # Data directory
    local hrp_data_dir
    hrp_data_dir=$(ask_input "HRP_DATA_DIR" "$DATA_DIR")

    # API Keys
    echo ""
    log_info "API Keys (leave blank to skip, can be added later):"

    local polygon_key
    polygon_key=$(ask_secret "POLYGON_API_KEY (for real-time data)")

    local alpaca_key
    alpaca_key=$(ask_secret "ALPACA_API_KEY (for market data)")

    local alpaca_secret
    alpaca_secret=$(ask_secret "ALPACA_SECRET_KEY")

    local tiingo_key
    tiingo_key=$(ask_secret "TIINGO_API_KEY (for market data)")

    local simfin_key
    simfin_key=$(ask_secret "SIMFIN_API_KEY (for fundamentals, falls back to YFinance)")

    local anthropic_key
    anthropic_key=$(ask_secret "ANTHROPIC_API_KEY (for Claude agents)")

    local resend_key
    resend_key=$(ask_secret "RESEND_API_KEY (for email notifications)")

    local notification_email=""
    if [[ -n "$resend_key" ]]; then
        notification_email=$(ask_input "NOTIFICATION_EMAIL" "")
    fi

    # Auth cookie key — auto-generate
    local auth_cookie_key
    auth_cookie_key=$(python -c "import secrets; print(secrets.token_hex(32))")

    # Broker selection
    echo ""
    log_info "Broker Configuration:"
    local broker_type=""
    broker_type=$(ask_input "HRP_BROKER_TYPE (ibkr/robinhood/paper)" "ibkr")

    # IBKR (only if selected)
    local ibkr_host="" ibkr_port="" ibkr_client_id="" ibkr_account="" ibkr_paper=""
    if [[ "$broker_type" == "ibkr" ]]; then
        if ask_yes_no "Configure Interactive Brokers settings?" "n"; then
            ibkr_host=$(ask_input "IBKR_HOST" "127.0.0.1")
            ibkr_port=$(ask_input "IBKR_PORT (7497=paper, 7496=live)" "7497")
            ibkr_client_id=$(ask_input "IBKR_CLIENT_ID" "1")
            ibkr_account=$(ask_input "IBKR_ACCOUNT (DU prefix=paper, U prefix=live)" "DU")
            ibkr_paper=$(ask_input "IBKR_PAPER_TRADING (true/false)" "true")
        fi
    fi

    # Robinhood (only if selected)
    local rh_username="" rh_password="" rh_totp="" rh_account=""
    if [[ "$broker_type" == "robinhood" ]]; then
        rh_username=$(ask_input "ROBINHOOD_USERNAME (email)")
        rh_password=$(ask_secret "ROBINHOOD_PASSWORD")
        rh_totp=$(ask_secret "ROBINHOOD_TOTP_SECRET (from authenticator setup)")
        rh_account=$(ask_input "ROBINHOOD_ACCOUNT_NUMBER (optional)")
    fi

    # Real-time ingestion
    local realtime_enabled="false"
    if [[ -n "$polygon_key" ]]; then
        echo ""
        log_info "Real-Time Ingestion (requires Polygon API key):"
        if ask_yes_no "Enable real-time intraday data ingestion?" "n"; then
            realtime_enabled="true"
        fi
    fi

    # Portfolio sizing
    echo ""
    log_info "Portfolio & Position Sizing (defaults shown, press Enter to accept):"
    local portfolio_value="" max_positions="" max_position_pct=""
    if ask_yes_no "Customize portfolio sizing? (defaults: $100k, 20 positions, 10% max)" "n"; then
        portfolio_value=$(ask_input "HRP_PORTFOLIO_VALUE (dollars)" "100000")
        max_positions=$(ask_input "HRP_MAX_POSITIONS" "20")
        max_position_pct=$(ask_input "HRP_MAX_POSITION_PCT (fraction, e.g. 0.10)" "0.10")
    fi

    # Write .env
    cat > "$env_file" << ENVEOF
# HRP Environment Variables
# Generated by setup.sh on $(date)
# See .env.example for full variable reference

# Environment
HRP_ENVIRONMENT=${hrp_env}
HRP_DATA_DIR=${hrp_data_dir}

# Database
HRP_DB_PATH=${hrp_data_dir}/hrp.duckdb

# Data Sources
POLYGON_API_KEY=${polygon_key}
ALPACA_API_KEY=${alpaca_key}
ALPACA_SECRET_KEY=${alpaca_secret}
TIINGO_API_KEY=${tiingo_key}
SIMFIN_API_KEY=${simfin_key}

# Claude API
ANTHROPIC_API_KEY=${anthropic_key}

# Notifications
RESEND_API_KEY=${resend_key}
NOTIFICATION_EMAIL=${notification_email}
NOTIFICATION_FROM_EMAIL=onboarding@resend.dev

# MLflow
MLFLOW_TRACKING_URI=sqlite:///${hrp_data_dir}/mlflow/mlflow.db

# Logging
LOG_LEVEL=INFO

# Dashboard Authentication
HRP_AUTH_ENABLED=true
HRP_AUTH_COOKIE_KEY=${auth_cookie_key}
HRP_AUTH_USERS_FILE=${hrp_data_dir}/auth/users.yaml
HRP_AUTH_COOKIE_NAME=hrp_auth
HRP_AUTH_COOKIE_EXPIRY_DAYS=30

# Ops Server
HRP_OPS_HOST=0.0.0.0
HRP_OPS_PORT=8080

# Broker Selection
HRP_BROKER_TYPE=${broker_type}
ENVEOF

    # Append IBKR if configured
    if [[ -n "$ibkr_host" ]]; then
        cat >> "$env_file" << IBKREOF

# Interactive Brokers
IBKR_HOST=${ibkr_host}
IBKR_PORT=${ibkr_port}
IBKR_CLIENT_ID=${ibkr_client_id}
IBKR_ACCOUNT=${ibkr_account}
IBKR_PAPER_TRADING=${ibkr_paper}
IBKREOF
    else
        cat >> "$env_file" << IBKRDEFEOF

# Interactive Brokers (defaults)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
IBKR_ACCOUNT=DU
IBKR_PAPER_TRADING=true
IBKRDEFEOF
    fi

    # Append Robinhood if configured
    if [[ "$broker_type" == "robinhood" ]]; then
        cat >> "$env_file" << RHEOF

# Robinhood
ROBINHOOD_USERNAME=${rh_username}
ROBINHOOD_PASSWORD=${rh_password}
ROBINHOOD_TOTP_SECRET=${rh_totp}
ROBINHOOD_ACCOUNT_NUMBER=${rh_account}
ROBINHOOD_PAPER_TRADING=true
ROBINHOOD_PICKLE_NAME=
RHEOF
    else
        cat >> "$env_file" << RHDEFEOF

# Robinhood (only if HRP_BROKER_TYPE=robinhood)
ROBINHOOD_USERNAME=
ROBINHOOD_PASSWORD=
ROBINHOOD_TOTP_SECRET=
ROBINHOOD_ACCOUNT_NUMBER=
ROBINHOOD_PAPER_TRADING=true
ROBINHOOD_PICKLE_NAME=
RHDEFEOF
    fi

    # Portfolio & Position Sizing
    cat >> "$env_file" << PORTFOLIOEOF

# Portfolio & Position Sizing
HRP_PORTFOLIO_VALUE=${portfolio_value:-100000}
HRP_MAX_POSITIONS=${max_positions:-20}
HRP_MAX_POSITION_PCT=${max_position_pct:-0.10}
HRP_MIN_ORDER_VALUE=100
HRP_TRADING_DRY_RUN=true

# Risk & VaR
HRP_USE_VAR_SIZING=true
HRP_AUTO_STOP_LOSS_PCT=
HRP_MAX_PORTFOLIO_VAR_PCT=0.02
HRP_MAX_POSITION_VAR_PCT=0.005

# Real-Time Ingestion
HRP_REALTIME_ENABLED=${realtime_enabled}
HRP_REALTIME_SYMBOLS=
HRP_REALTIME_FLUSH_INTERVAL=10
HRP_REALTIME_MAX_BUFFER_SIZE=10000
HRP_REALTIME_RECONNECT_MAX_DELAY=60

# Alert Thresholds (optional — have code defaults)
HRP_THRESHOLD_HEALTH_SCORE_WARNING=90
HRP_THRESHOLD_HEALTH_SCORE_CRITICAL=70
HRP_THRESHOLD_FRESHNESS_WARNING_DAYS=3
HRP_THRESHOLD_FRESHNESS_CRITICAL_DAYS=5
HRP_THRESHOLD_ANOMALY_COUNT_WARNING=50
HRP_THRESHOLD_ANOMALY_COUNT_CRITICAL=100
HRP_THRESHOLD_KL_DIVERGENCE_THRESHOLD=0.2
HRP_THRESHOLD_PSI_THRESHOLD=0.2
HRP_THRESHOLD_IC_DECAY_THRESHOLD=0.2
HRP_THRESHOLD_INGESTION_SUCCESS_RATE_WARNING=95
HRP_THRESHOLD_INGESTION_SUCCESS_RATE_CRITICAL=80
PORTFOLIOEOF

    log_success ".env written to $env_file"
}

###############################################################################
# Phase 5: Database Initialization
###############################################################################

phase_database() {
    log_phase "5" "Database Initialization"

    # Ensure venv is active
    local venv_python="$PROJECT_ROOT/.venv/bin/python"
    if [[ ! -f "$venv_python" ]]; then
        log_error "Virtual environment not found. Run Phase 2 first."
        return 1
    fi

    # Load .env if it exists
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.env"
        set +a
    fi

    local db_path="${HRP_DB_PATH:-$DATA_DIR/hrp.duckdb}"

    if [[ -f "$db_path" ]]; then
        log_info "Database already exists at $db_path"
        log_info "Verifying schema..."
        if "$venv_python" -m hrp.data.schema --verify 2>/dev/null; then
            log_success "Database schema is valid"
            return 0
        else
            log_warning "Schema verification failed — re-initializing"
        fi
    fi

    log_info "Initializing database schema..."
    if "$venv_python" -m hrp.data.schema --init; then
        log_success "Database schema initialized"
    else
        log_error "Database initialization failed"
        return 1
    fi

    log_info "Verifying schema..."
    if "$venv_python" -m hrp.data.schema --verify; then
        log_success "Database schema verified"
    else
        log_error "Schema verification failed after init"
        return 1
    fi
}

###############################################################################
# Phase 6: Fix Configuration Files
###############################################################################

phase_fix_configs() {
    log_phase "6" "Fix Configuration Files"

    local current_user
    current_user="$(whoami)"
    local current_home
    current_home="$HOME"

    # --- Fix .mcp.json ---
    local mcp_file="$PROJECT_ROOT/.mcp.json"
    if [[ -f "$mcp_file" ]]; then
        local current_pythonpath
        current_pythonpath=$(python -c "
import json, sys
with open('$mcp_file') as f:
    d = json.load(f)
pp = d.get('mcpServers', {}).get('hrp-research', {}).get('env', {}).get('PYTHONPATH', '')
print(pp)
" 2>/dev/null || echo "")

        if [[ "$current_pythonpath" == "$PROJECT_ROOT" ]]; then
            log_success ".mcp.json PYTHONPATH already correct"
        else
            log_info "Updating .mcp.json PYTHONPATH to $PROJECT_ROOT..."
            python -c "
import json
with open('$mcp_file') as f:
    d = json.load(f)
d['mcpServers']['hrp-research']['env']['PYTHONPATH'] = '$PROJECT_ROOT'
with open('$mcp_file', 'w') as f:
    json.dump(d, f, indent=2)
    f.write('\n')
"
            log_success ".mcp.json updated"
        fi
    else
        log_warning ".mcp.json not found — skipping"
    fi

    # --- Fix launchd plists ---
    local plist_dir="$PROJECT_ROOT/launchd"
    if [[ -d "$plist_dir" ]]; then
        local plist_count=0
        local fixed_count=0

        for plist in "$plist_dir"/*.plist; do
            [[ -f "$plist" ]] || continue
            ((plist_count++)) || true

            if grep -q '/Users/fer/' "$plist" 2>/dev/null; then
                # Replace all /Users/fer/ references with current user's paths
                sed -i '' "s|/Users/fer/Projects/HRP/.venv|${PROJECT_ROOT}/.venv|g" "$plist"
                sed -i '' "s|/Users/fer/Projects/HRP|${PROJECT_ROOT}|g" "$plist"
                sed -i '' "s|/Users/fer/hrp-data|${current_home}/hrp-data|g" "$plist"
                ((fixed_count++)) || true
            fi
        done

        if [[ "$fixed_count" -gt 0 ]]; then
            log_success "Fixed $fixed_count/$plist_count launchd plists (updated paths for $current_user)"
        else
            log_success "All $plist_count launchd plists already have correct paths"
        fi
    else
        log_warning "launchd/ directory not found — skipping"
    fi
}

###############################################################################
# Phase 7: Dashboard Authentication
###############################################################################

phase_dashboard_auth() {
    log_phase "7" "Dashboard Authentication"

    if ! ask_yes_no "Create a dashboard user now?" "y"; then
        log_info "Skipped — create later with: python -m hrp.dashboard.auth_cli add-user"
        return 0
    fi

    local venv_python="$PROJECT_ROOT/.venv/bin/python"

    # Load .env for auth config
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.env"
        set +a
    fi

    local username
    username=$(ask_input "Username" "admin")
    local email
    email=$(ask_input "Email" "")
    local display_name
    display_name=$(ask_input "Display name" "$username")

    echo -en "${BOLD}Password: ${NC}" > /dev/tty
    read -rs password < /dev/tty
    echo "" > /dev/tty

    if "$venv_python" -m hrp.dashboard.auth_cli add-user \
        --username "$username" \
        --email "$email" \
        --name "$display_name" \
        --password "$password" 2>/dev/null; then
        log_success "Dashboard user '$username' created"
    else
        log_warning "Failed to create user. Create later with: python -m hrp.dashboard.auth_cli add-user"
    fi
}

###############################################################################
# Phase 8: Data Bootstrap (Optional)
###############################################################################

phase_data_bootstrap() {
    log_phase "8" "Data Bootstrap (Optional)"

    # Top 20 most traded S&P 500 stocks (by average daily volume)
    local BOOTSTRAP_SYMBOLS="AAPL,MSFT,NVDA,AMZN,META,TSLA,GOOGL,GOOG,AMD,AVGO,NFLX,COST,ADBE,CRM,PEP,CSCO,INTC,QCOM,TMUS,INTU"

    log_info "Bootstrap loads 2 years of data for the top 20 most traded S&P 500 stocks:"
    log_info "  ${BOOTSTRAP_SYMBOLS}"
    log_info "Uses Yahoo Finance (no API key required). Takes ~2-5 minutes."
    echo ""

    if ! ask_yes_no "Run data bootstrap now?" "y"; then
        log_info "Skipped — run later with:"
        log_info "  python -m hrp.agents.run_job --job universe"
        log_info "  python -m hrp.agents.run_job --job prices"
        log_info "  python -m hrp.agents.run_job --job features"
        return 0
    fi

    local venv_python="$PROJECT_ROOT/.venv/bin/python"

    # Load .env
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.env"
        set +a
    fi

    log_info "Step 1/3: Loading S&P 500 universe..."
    if "$venv_python" -m hrp.agents.run_job --job universe; then
        log_success "Universe loaded (396 symbols)"
    else
        log_error "Universe load failed"
        return 1
    fi

    log_info "Step 2/3: Ingesting 2 years of prices for top 20 stocks..."
    if "$venv_python" -c "
from datetime import date, timedelta
from hrp.agents.jobs import PriceIngestionJob

symbols = '${BOOTSTRAP_SYMBOLS}'.split(',')
start = date.today() - timedelta(days=730)
end = date.today()

job = PriceIngestionJob(symbols=symbols, start=start, end=end)
job.run()
print(f'Bootstrap complete: {len(symbols)} symbols, {start} to {end}')
"; then
        log_success "Prices ingested (20 stocks, 2 years)"
    else
        log_error "Price ingestion failed"
        return 1
    fi

    log_info "Step 3/3: Computing features for bootstrapped stocks..."
    if "$venv_python" -c "
from hrp.agents.jobs import FeatureComputationJob

symbols = '${BOOTSTRAP_SYMBOLS}'.split(',')
job = FeatureComputationJob(symbols=symbols)
job.run()
print(f'Features computed for {len(symbols)} symbols')
"; then
        log_success "Features computed (20 stocks)"
    else
        log_error "Feature computation failed"
        return 1
    fi

    log_success "Data bootstrap complete! (20 stocks, 2 years)"
    log_info "To load the full universe later: python -m hrp.agents.run_job --job prices"
    echo ""
    echo -e "${BOLD}To enable automated research reports:${NC}"
    echo "  1. Add ANTHROPIC_API_KEY to .env  (powers Signal Scientist, Alpha Researcher, CIO, Report Generator)"
    echo "  2. Add RESEND_API_KEY + NOTIFICATION_EMAIL to .env  (delivers reports via email)"
    echo "  3. Start the scheduler:  ./scripts/startup.sh start --full"
    echo "     Or install launchd:   ./scripts/manage_launchd.sh install"
    echo ""
    echo -e "${BOLD}Pipeline:${NC} Signal Scientist → Alpha Researcher → ML Scientist → ... → Report Generator"
    echo "  Without ANTHROPIC_API_KEY, Claude-powered agents will not run."
}

###############################################################################
# Phase 9: Launchd Jobs (macOS Only)
###############################################################################

phase_launchd() {
    log_phase "9" "Launchd Jobs (macOS)"

    if [[ "$(uname -s)" != "Darwin" ]]; then
        log_info "Not macOS — skipping launchd setup"
        return 0
    fi

    if ! ask_yes_no "Install launchd jobs for scheduled data ingestion?" "n"; then
        log_info "Skipped — install later with: scripts/manage_launchd.sh install"
        return 0
    fi

    local manage_script="$PROJECT_ROOT/scripts/manage_launchd.sh"
    if [[ ! -x "$manage_script" ]]; then
        chmod +x "$manage_script"
    fi

    if "$manage_script" install; then
        log_success "Launchd jobs installed"
        log_info "Check status: scripts/manage_launchd.sh status"
    else
        log_error "Launchd install failed"
        return 1
    fi
}

###############################################################################
# Phase 10: Verification
###############################################################################

phase_verification() {
    log_phase "10" "Verification"

    local venv_python="$PROJECT_ROOT/.venv/bin/python"

    # Load .env
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.env"
        set +a
    fi

    PASS_COUNT=0
    FAIL_COUNT=0
    SKIP_COUNT=0

    # 1. Python version
    local pyver
    pyver="$("$venv_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")"
    local major="${pyver%%.*}"
    local minor="${pyver#*.}"
    if [[ "$major" -ge 3 ]] && [[ "$minor" -ge 11 ]]; then
        check_result "Python >= 3.11 ($pyver)" "pass"
    else
        check_result "Python >= 3.11 (got $pyver)" "fail"
    fi

    # 2. Critical imports
    for pkg in duckdb vectorbt mlflow lightgbm xgboost streamlit empyrical; do
        if "$venv_python" -c "import $pkg" 2>/dev/null; then
            check_result "import $pkg" "pass"
        else
            check_result "import $pkg" "fail"
        fi
    done

    # 3. .env file
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        check_result ".env file exists" "pass"
    else
        check_result ".env file exists" "fail"
    fi

    # 4. Database schema
    if "$venv_python" -m hrp.data.schema --verify 2>/dev/null; then
        check_result "Database schema valid" "pass"
    else
        check_result "Database schema valid" "fail"
    fi

    # 5. PlatformAPI initializes
    if "$venv_python" -c "from hrp.api.platform import PlatformAPI; PlatformAPI()" 2>/dev/null; then
        check_result "PlatformAPI initializes" "pass"
    else
        check_result "PlatformAPI initializes" "fail"
    fi

    # 6. Directory structure
    local dirs_ok=true
    for dir in logs auth mlflow; do
        if [[ ! -d "$DATA_DIR/$dir" ]]; then
            dirs_ok=false
            break
        fi
    done
    if [[ "$dirs_ok" == true ]]; then
        check_result "Directory structure ($DATA_DIR/)" "pass"
    else
        check_result "Directory structure ($DATA_DIR/)" "fail"
    fi

    # 7. .mcp.json paths
    local mcp_file="$PROJECT_ROOT/.mcp.json"
    if [[ -f "$mcp_file" ]]; then
        local mcp_path
        mcp_path=$("$venv_python" -c "
import json
with open('$mcp_file') as f:
    d = json.load(f)
print(d.get('mcpServers',{}).get('hrp-research',{}).get('env',{}).get('PYTHONPATH',''))
" 2>/dev/null || echo "")
        if [[ "$mcp_path" == "$PROJECT_ROOT" ]]; then
            check_result ".mcp.json PYTHONPATH correct" "pass"
        else
            check_result ".mcp.json PYTHONPATH correct (got: $mcp_path)" "fail"
        fi
    else
        check_result ".mcp.json exists" "skip"
    fi

    # 8. Launchd plists — no /Users/fer/ references
    local plist_dir="$PROJECT_ROOT/launchd"
    if [[ -d "$plist_dir" ]]; then
        if grep -rl '/Users/fer/' "$plist_dir"/*.plist 2>/dev/null | head -1 | grep -q .; then
            check_result "Launchd plists (no /Users/fer/ references)" "fail"
        else
            check_result "Launchd plists (no /Users/fer/ references)" "pass"
        fi
    else
        check_result "Launchd plists" "skip"
    fi

    # 9. Test collection
    local collection_errors=""
    collection_errors=$("$venv_python" -m pytest "$PROJECT_ROOT/tests/" --co -q 2>&1 | tail -3 || true)
    if echo "$collection_errors" | grep -q "0 errors\|no tests ran\|tests collected\|test"; then
        # Check specifically for "error" in collection output
        local error_count
        error_count=$("$venv_python" -m pytest "$PROJECT_ROOT/tests/" --co -q 2>&1 | grep -c " error" || echo "0")
        if [[ "$error_count" -eq 0 ]]; then
            check_result "Test collection (0 errors)" "pass"
        else
            check_result "Test collection ($error_count errors)" "fail"
        fi
    else
        # Try to get error count from output
        local err_line
        err_line=$("$venv_python" -m pytest "$PROJECT_ROOT/tests/" --co -q 2>&1 | grep -oE '[0-9]+ errors?' | head -1 || echo "")
        if [[ -n "$err_line" ]]; then
            check_result "Test collection ($err_line)" "fail"
        else
            check_result "Test collection" "pass"
        fi
    fi

    # Summary
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}$PASS_COUNT passed${NC}  ${RED}$FAIL_COUNT failed${NC}  ${YELLOW}$SKIP_COUNT skipped${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if [[ "$FAIL_COUNT" -eq 0 ]]; then
        echo ""
        log_success "HRP is fully set up! 🎉"
    else
        echo ""
        log_warning "$FAIL_COUNT check(s) failed. Review the items above."
    fi

    # Next steps
    echo ""
    echo -e "${BOLD}Next Steps:${NC}"
    echo "  1. Activate venv:       source .venv/bin/activate"
    echo "  2. Run tests:           pytest tests/ -v"
    echo "  3. Start services:      ./scripts/startup.sh start"
    echo "  4. Open dashboard:      http://localhost:8501"
    echo "  5. Load data:           python -m hrp.agents.run_job --job prices"
    echo ""
}

###############################################################################
# Main
###############################################################################

print_banner() {
    # Read version from pyproject.toml
    local version
    version=$(grep '^version' "$PROJECT_ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/' 2>/dev/null || echo "?.?.?")
    local build
    build=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")

    echo ""
    echo -e "${BLUE}${BOLD}"
    cat << 'LOGO'

         ██╗  ██╗ ██████╗  ██████╗
         ██║  ██║ ██╔══██╗ ██╔══██╗
         ███████║ ██████╔╝ ██████╔╝
         ██╔══██║ ██╔══██╗ ██╔═══╝
         ██║  ██║ ██║  ██║ ██║
         ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝

LOGO
    echo -e "${NC}"
    echo -e "     ${BOLD}Hedgefund Research Platform${NC}"
    echo -e "     Quantitative Research for Systematic Trading"
    echo ""
    echo -e "     ${BOLD}Version${NC} ${GREEN}${version}${NC}  ${BOLD}Build${NC} ${GREEN}${build}${NC}"
    echo ""
}

main() {
    cd "$PROJECT_ROOT"

    # Check-only mode
    if [[ "${1:-}" == "--check" ]]; then
        print_banner
        phase_verification
        exit $FAIL_COUNT
    fi

    print_banner

    log_info "This script will set up HRP on your machine."
    log_info "It's safe to re-run — completed steps will be skipped."
    echo ""

    if ! ask_yes_no "Ready to begin?" "y"; then
        log_info "Setup cancelled."
        exit 0
    fi

    phase_preflight
    phase_system_deps
    phase_python_env
    phase_directories
    phase_env_config
    phase_database
    phase_fix_configs
    phase_dashboard_auth
    phase_data_bootstrap
    phase_launchd
    phase_verification
}

main "$@"
