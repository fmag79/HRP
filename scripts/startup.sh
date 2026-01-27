#!/usr/bin/env bash
###############################################################################
# HRP System Startup Script
#
# Starts/stops all HRP services:
# - Dashboard (Streamlit) on port 8501
# - MLflow UI on port 5000
# - Scheduler (with optional research agents)
#
# Usage:
#   ./scripts/startup.sh [start|stop|status|restart] [options]
#
# Examples:
#   ./scripts/startup.sh start                    # Start all core services
#   ./scripts/startup.sh start --full             # Start with all research agents
#   ./scripts/startup.sh start --dashboard-only   # Start dashboard only
#   ./scripts/startup.sh stop                     # Stop all services
#   ./scripts/startup.sh status                   # Show service status
###############################################################################

set -e

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Project root
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# PID directory
readonly PID_DIR="$PROJECT_ROOT/.hrp_pids"
readonly LOG_DIR="$HOME/hrp-data/logs"

# Service PIDs
PID_DASHBOARD="$PID_DIR/dashboard.pid"
PID_MLFLOW="$PID_DIR/mlflow.pid"
PID_SCHEDULER="$PID_DIR/scheduler.pid"

# Service ports
DASHBOARD_PORT="${HRP_DASHBOARD_PORT:-8501}"
MLFLOW_PORT="${HRP_MLFLOW_PORT:-5000}"

# MLflow backend
MLFLOW_BACKEND="${HRP_MLFLOW_BACKEND:-sqlite:///~/hrp-data/mlflow/mlflow.db}"

# Ensure directories exist
mkdir -p "$PID_DIR"
mkdir -p "$LOG_DIR"

###############################################################################
# Utility Functions
###############################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  HRP System Manager${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Check if a service is running by PID file
is_running() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            # Stale PID file
            rm -f "$pid_file"
            return 1
        fi
    fi
    return 1
}

# Wait for a service to be ready
wait_for_service() {
    local service_name="$1"
    local port="$2"
    local max_wait="${3:-30}"
    local count=0

    while [[ $count -lt $max_wait ]]; do
        if lsof -i ":$port" > /dev/null 2>&1 || nc -z localhost "$port" 2>/dev/null; then
            log_success "$service_name is ready on port $port"
            return 0
        fi
        sleep 1
        ((count++))
    done

    log_warning "$service_name did not start within ${max_wait}s"
    return 1
}

# Check if a port is available
is_port_available() {
    local port="$1"
    if lsof -i ":$port" > /dev/null 2>&1; then
        return 1  # Port is in use
    fi
    return 0  # Port is available
}

# Get process using a port
get_port_user() {
    local port="$1"
    lsof -i ":$port" -t 2>/dev/null | head -1
}

###############################################################################
# Service Management Functions
###############################################################################

start_dashboard() {
    if is_running "$PID_DASHBOARD"; then
        log_warning "Dashboard is already running (PID: $(cat $PID_DASHBOARD))"
        return 0
    fi

    # Check if port is available
    if ! is_port_available "$DASHBOARD_PORT"; then
        local port_user=$(get_port_user "$DASHBOARD_PORT")
        log_error "Port $DASHBOARD_PORT is already in use (PID: $port_user)"
        log_info "Try: HRP_DASHBOARD_PORT=8502 ./scripts/startup.sh start --dashboard-only"
        return 1
    fi

    log_info "Starting HRP Dashboard on port $DASHBOARD_PORT..."
    nohup streamlit run hrp/dashboard/app.py \
        --browser.gatherUsageStats=false \
        --server.port="$DASHBOARD_PORT" \
        --server.headless=true \
        > "$LOG_DIR/dashboard.out.log" \
        2> "$LOG_DIR/dashboard.error.log" &
    local pid=$!
    echo $pid > "$PID_DASHBOARD"

    if wait_for_service "Dashboard" "$DASHBOARD_PORT"; then
        log_success "Dashboard started with PID: $pid"
        log_info "URL: http://localhost:$DASHBOARD_PORT"
        return 0
    else
        log_error "Dashboard failed to start. Check $LOG_DIR/dashboard.error.log"
        rm -f "$PID_DASHBOARD"
        return 1
    fi
}

stop_dashboard() {
    if is_running "$PID_DASHBOARD"; then
        local pid=$(cat "$PID_DASHBOARD")
        log_info "Stopping Dashboard (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        rm -f "$PID_DASHBOARD"
        log_success "Dashboard stopped"
    else
        log_warning "Dashboard is not running"
    fi
}

start_mlflow() {
    if is_running "$PID_MLFLOW"; then
        log_warning "MLflow UI is already running (PID: $(cat $PID_MLFLOW))"
        return 0
    fi

    # Check if port is available
    if ! is_port_available "$MLFLOW_PORT"; then
        local port_user=$(get_port_user "$MLFLOW_PORT")
        log_error "Port $MLFLOW_PORT is already in use (PID: $port_user)"
        log_info "Try: HRP_MLFLOW_PORT=5001 ./scripts/startup.sh start --mlflow-only"
        return 1
    fi

    log_info "Starting MLflow UI on port $MLFLOW_PORT..."
    nohup mlflow ui \
        --backend-store-uri "$MLFLOW_BACKEND" \
        --port "$MLFLOW_PORT" \
        --host 0.0.0.0 \
        > "$LOG_DIR/mlflow.out.log" \
        2> "$LOG_DIR/mlflow.error.log" &
    local pid=$!
    echo $pid > "$PID_MLFLOW"

    if wait_for_service "MLflow UI" "$MLFLOW_PORT" 60; then
        log_success "MLflow UI started with PID: $pid"
        log_info "URL: http://localhost:$MLFLOW_PORT"
        return 0
    else
        log_error "MLflow UI failed to start. Check $LOG_DIR/mlflow.error.log"
        rm -f "$PID_MLFLOW"
        return 1
    fi
}

stop_mlflow() {
    if is_running "$PID_MLFLOW"; then
        local pid=$(cat "$PID_MLFLOW")
        log_info "Stopping MLflow UI (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        rm -f "$PID_MLFLOW"
        log_success "MLflow UI stopped"
    else
        log_warning "MLflow UI is not running"
    fi
}

start_scheduler() {
    if is_running "$PID_SCHEDULER"; then
        log_warning "Scheduler is already running (PID: $(cat $PID_SCHEDULER))"
        return 0
    fi

    log_info "Starting HRP Scheduler..."

    local scheduler_args=()
    scheduler_args+=("--price-time" "${HRP_PRICE_TIME:-18:00}")
    scheduler_args+=("--universe-time" "${HRP_UNIVERSE_TIME:-18:05}")
    scheduler_args+=("--feature-time" "${HRP_FEATURE_TIME:-18:10}")

    if [[ "${HRP_NO_BACKUP:-false}" == "true" ]]; then
        scheduler_args+=("--no-backup")
    fi

    if [[ "${HRP_NO_FUNDAMENTALS:-false}" == "true" ]]; then
        scheduler_args+=("--no-fundamentals")
    fi

    # Full mode: enable all research agents
    if [[ "$FULL_MODE" == "true" ]]; then
        scheduler_args+=("--with-research-triggers")
        scheduler_args+=("--with-signal-scan")
        scheduler_args+=("--with-quality-sentinel")
        scheduler_args+=("--with-daily-report")
        scheduler_args+=("--with-weekly-report")
        log_info "Starting scheduler with FULL research agent pipeline"
    elif [[ "$MINIMAL_MODE" == "true" ]]; then
        scheduler_args+=("--no-backup")
        scheduler_args+=("--no-fundamentals")
        log_info "Starting scheduler in MINIMAL mode (ingestion only)"
    fi

    nohup python -m hrp.agents.run_scheduler "${scheduler_args[@]}" \
        > "$LOG_DIR/scheduler.out.log" \
        2> "$LOG_DIR/scheduler.error.log" &
    local pid=$!
    echo $pid > "$PID_SCHEDULER"

    # Give scheduler a moment to start
    sleep 3

    if ps -p "$pid" > /dev/null 2>&1; then
        log_success "Scheduler started with PID: $pid"
        log_info "Logs: $LOG_DIR/scheduler.out.log"
        return 0
    else
        log_error "Scheduler failed to start. Check $LOG_DIR/scheduler.error.log"
        rm -f "$PID_SCHEDULER"
        return 1
    fi
}

stop_scheduler() {
    if is_running "$PID_SCHEDULER"; then
        local pid=$(cat "$PID_SCHEDULER")
        log_info "Stopping Scheduler (PID: $pid)..."
        kill "$pid" 2>/dev/null || true

        # Wait for graceful shutdown
        local count=0
        while [[ $count -lt 10 ]] && ps -p "$pid" > /dev/null 2>&1; do
            sleep 1
            ((count++))
        done

        # Force kill if still running
        if ps -p "$pid" > /dev/null 2>&1; then
            log_warning "Force killing scheduler..."
            kill -9 "$pid" 2>/dev/null || true
        fi

        rm -f "$PID_SCHEDULER"
        log_success "Scheduler stopped"
    else
        log_warning "Scheduler is not running"
    fi
}

###############################################################################
# Status Functions
###############################################################################

show_status() {
    print_header

    local running=0
    local stopped=0

    # Dashboard
    if is_running "$PID_DASHBOARD"; then
        local dash_pid
        dash_pid=$(cat "$PID_DASHBOARD" 2>/dev/null || echo "unknown")
        echo -e "  Dashboard:      ${GREEN}RUNNING${NC} (PID: $dash_pid, Port: $DASHBOARD_PORT)"
        running=$((running + 1))
    else
        echo -e "  Dashboard:      ${RED}STOPPED${NC}"
        stopped=$((stopped + 1))
    fi

    # MLflow
    if is_running "$PID_MLFLOW"; then
        local mlflow_pid
        mlflow_pid=$(cat "$PID_MLFLOW" 2>/dev/null || echo "unknown")
        echo -e "  MLflow UI:      ${GREEN}RUNNING${NC} (PID: $mlflow_pid, Port: $MLFLOW_PORT)"
        running=$((running + 1))
    else
        echo -e "  MLflow UI:      ${RED}STOPPED${NC}"
        stopped=$((stopped + 1))
    fi

    # Scheduler
    if is_running "$PID_SCHEDULER"; then
        local sched_pid
        sched_pid=$(cat "$PID_SCHEDULER" 2>/dev/null || echo "unknown")
        echo -e "  Scheduler:      ${GREEN}RUNNING${NC} (PID: $sched_pid)"
        running=$((running + 1))
    else
        echo -e "  Scheduler:      ${RED}STOPPED${NC}"
        stopped=$((stopped + 1))
    fi

    echo ""
    echo -e "  Total: ${GREEN}$running running${NC}, ${RED}$stopped stopped${NC}"
    echo ""

    if [[ $running -gt 0 ]]; then
        echo -e "  ${BLUE}Logs directory:${NC} $LOG_DIR"
        echo -e "  ${BLUE}PID directory:${NC} $PID_DIR"
    fi
}

###############################################################################
# Main Commands
###############################################################################

start_all() {
    print_header

    local failures=0

    start_dashboard || ((failures++))
    start_mlflow || ((failures++))
    start_scheduler || ((failures++))

    echo ""
    if [[ $failures -eq 0 ]]; then
        log_success "All services started successfully!"
        echo ""
        echo -e "  ${GREEN}Dashboard:${NC}   http://localhost:$DASHBOARD_PORT"
        echo -e "  ${GREEN}MLflow UI:${NC}   http://localhost:$MLFLOW_PORT"
        echo ""
        echo "Use './scripts/startup.sh status' to check status"
        echo "Use './scripts/startup.sh stop' to stop all services"
    else
        log_warning "Started with $failures failure(s). Check logs above."
        return 1
    fi
}

stop_all() {
    print_header
    log_info "Stopping all HRP services..."

    stop_scheduler
    stop_mlflow
    stop_dashboard

    log_success "All services stopped"
}

restart_all() {
    stop_all
    sleep 2
    start_all
}

###############################################################################
# CLI Interface
###############################################################################

print_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
  start               Start all services (dashboard, MLflow, scheduler)
  stop                Stop all services
  restart             Restart all services
  status              Show service status

Start Options:
  --full              Start with all research agents enabled
  --minimal           Start with minimal configuration (ingestion only)
  --dashboard-only    Start only the dashboard
  --mlflow-only       Start only MLflow UI
  --scheduler-only    Start only the scheduler

Environment Variables:
  HRP_DASHBOARD_PORT  Dashboard port (default: 8501)
  HRP_MLFLOW_PORT     MLflow UI port (default: 5000)
  HRP_PRICE_TIME      Price ingestion time (default: 18:00)
  HRP_UNIVERSE_TIME   Universe update time (default: 18:05)
  HRP_FEATURE_TIME    Feature computation time (default: 18:10)
  HRP_NO_BACKUP       Disable daily backup (true/false)
  HRP_NO_FUNDAMENTALS Disable fundamentals ingestion (true/false)

Examples:
  $0 start                           # Start all core services
  $0 start --full                    # Start with research agents
  $0 start --minimal                 # Minimal ingestion mode
  $0 start --dashboard-only          # Dashboard only
  $0 stop                            # Stop all services
  $0 status                          # Show service status
  HRP_DASHBOARD_PORT=8080 $0 start   # Custom dashboard port

EOF
}

###############################################################################
# Script Entry Point
###############################################################################

MAIN_COMMAND="${1:-}"
shift || true

# Parse options
FULL_MODE="false"
MINIMAL_MODE="false"
DASHBOARD_ONLY="false"
MLFLOW_ONLY="false"
SCHEDULER_ONLY="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            FULL_MODE="true"
            shift
            ;;
        --minimal)
            MINIMAL_MODE="true"
            shift
            ;;
        --dashboard-only)
            DASHBOARD_ONLY="true"
            shift
            ;;
        --mlflow-only)
            MLFLOW_ONLY="true"
            shift
            ;;
        --scheduler-only)
            SCHEDULER_ONLY="true"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Execute command
# Handle help as main command
case "$MAIN_COMMAND" in
    -h|--help)
        print_usage
        exit 0
        ;;
    start)
        if [[ "$DASHBOARD_ONLY" == "true" ]]; then
            print_header
            start_dashboard
        elif [[ "$MLFLOW_ONLY" == "true" ]]; then
            print_header
            start_mlflow
        elif [[ "$SCHEDULER_ONLY" == "true" ]]; then
            print_header
            start_scheduler
        else
            start_all
        fi
        ;;
    stop)
        stop_all
        ;;
    restart)
        restart_all
        ;;
    status)
        show_status
        ;;
    "")
        print_header
        echo "No command specified. Available commands:"
        echo "  start   - Start all HRP services"
        echo "  stop    - Stop all HRP services"
        echo "  restart - Restart all HRP services"
        echo "  status  - Show service status"
        echo ""
        echo "Use '$0 --help' for more information."
        ;;
    *)
        log_error "Unknown command: $MAIN_COMMAND"
        print_usage
        exit 1
        ;;
esac
