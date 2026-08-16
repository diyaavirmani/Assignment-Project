#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 3GPP RAG Assistant — Local Development Startup Script
# ---------------------------------------------------------------------------
# Starts the FastAPI backend and Streamlit UI in the background.
# Assumes a Python virtual environment at .venv/ in the project root.
#
# Usage:
#   ./scripts/start.sh          # start both services
#   ./scripts/start.sh api      # start API only
#   ./scripts/start.sh ui       # start UI only
#   ./scripts/start.sh stop     # stop all services
#   ./scripts/start.sh status   # show running services
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve project root (parent of this scripts/ directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

VENV="$PROJECT_ROOT/.venv"
PYTHON="$VENV/bin/python3"
UVICORN="$VENV/bin/uvicorn"
STREAMLIT="$VENV/bin/streamlit"

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
LOG_DIR="$PROJECT_ROOT/logs"
API_LOG="$LOG_DIR/api.log"
UI_LOG="$LOG_DIR/ui.log"
PID_DIR="$PROJECT_ROOT/.pids"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()    { echo "[INFO]  $*"; }
success() { echo "[OK]    $*"; }
warn()    { echo "[WARN]  $*"; }
error()   { echo "[ERROR] $*" >&2; }

check_venv() {
    if [[ ! -f "$PYTHON" ]]; then
        error "Virtual environment not found at $VENV"
        echo "  Create it with:  python3 -m venv .venv"
        echo "  Then install:    .venv/bin/pip install -r requirements.txt"
        echo "  Then install src: .venv/bin/pip install -e ."
        exit 1
    fi
}

check_ollama() {
    if ! curl -s "${OLLAMA_BASE_URL:-http://localhost:11434}" > /dev/null 2>&1; then
        warn "Ollama does not appear to be running at ${OLLAMA_BASE_URL:-http://localhost:11434}"
        warn "Start it with: ollama serve"
        warn "Pull a model:  ollama pull llama3.2"
    else
        success "Ollama is reachable"
    fi
}

check_index() {
    if ! "$PYTHON" -c "
import sys; sys.path.insert(0,'.')
from src.core.vector_store import VectorStore
s = VectorStore().get_stats()
sys.exit(0 if s['total_chunks'] > 0 else 1)
" 2>/dev/null; then
        warn "Vector index is empty."
        warn "Build it with:"
        warn "  python src/core/document_processor.py"
        warn "  python scripts/build_index.py"
    else
        success "Vector index contains data"
    fi
}

start_api() {
    mkdir -p "$LOG_DIR" "$PID_DIR"

    if [[ -f "$PID_DIR/api.pid" ]] && kill -0 "$(cat "$PID_DIR/api.pid")" 2>/dev/null; then
        warn "API already running (PID $(cat "$PID_DIR/api.pid"))"
        return
    fi

    info "Starting FastAPI backend on port $API_PORT..."
    "$UVICORN" src.api.main:app \
        --host 0.0.0.0 \
        --port "$API_PORT" \
        --reload \
        > "$API_LOG" 2>&1 &
    echo $! > "$PID_DIR/api.pid"
    success "API started (PID $!, log: $API_LOG)"
    info "  Docs:   http://localhost:$API_PORT/docs"
    info "  Health: http://localhost:$API_PORT/health"
}

start_ui() {
    mkdir -p "$LOG_DIR" "$PID_DIR"

    if [[ -f "$PID_DIR/ui.pid" ]] && kill -0 "$(cat "$PID_DIR/ui.pid")" 2>/dev/null; then
        warn "UI already running (PID $(cat "$PID_DIR/ui.pid"))"
        return
    fi

    # Ensure streamlit doesn't prompt for email
    mkdir -p ~/.streamlit
    if [[ ! -f ~/.streamlit/credentials.toml ]]; then
        echo '[general]' > ~/.streamlit/credentials.toml
        echo 'email = ""' >> ~/.streamlit/credentials.toml
    fi

    info "Starting Streamlit UI on port $UI_PORT..."
    "$STREAMLIT" run src/frontend/app.py \
        --server.port "$UI_PORT" \
        --server.headless true \
        --browser.gatherUsageStats false \
        > "$UI_LOG" 2>&1 &
    echo $! > "$PID_DIR/ui.pid"
    success "UI started (PID $!, log: $UI_LOG)"
    info "  URL: http://localhost:$UI_PORT"
}

stop_service() {
    local name="$1"
    local pidfile="$PID_DIR/$name.pid"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            success "Stopped $name (PID $pid)"
        else
            warn "$name PID $pid not found (already stopped?)"
        fi
        rm -f "$pidfile"
    else
        warn "No PID file for $name — is it running?"
    fi
}

status() {
    echo ""
    echo "=== 3GPP RAG Assistant Status ==="

    for svc in api ui; do
        local pidfile="$PID_DIR/$svc.pid"
        if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            success "$svc is running (PID $(cat "$pidfile"))"
        else
            echo "[STOPPED] $svc"
        fi
    done

    echo ""
    if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        success "API health check: OK (http://localhost:$API_PORT/health)"
    else
        warn "API not responding on port $API_PORT"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CMD="${1:-all}"

case "$CMD" in
    all)
        check_venv
        check_ollama
        check_index
        echo ""
        start_api
        sleep 3
        start_ui
        echo ""
        info "Both services started."
        info "Stop with: ./scripts/start.sh stop"
        ;;
    api)
        check_venv
        start_api
        ;;
    ui)
        check_venv
        start_ui
        ;;
    stop)
        stop_service api
        stop_service ui
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 [all|api|ui|stop|status]"
        exit 1
        ;;
esac
