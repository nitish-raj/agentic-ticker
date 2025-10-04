#!/bin/bash

# Agentic Ticker Launcher Script
# A user-friendly launcher for the Streamlit application

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="Agentic Ticker"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
STREAMLIT_APP="${SCRIPT_DIR}/agentic_ticker.py"
FASTAPI_APP="${SCRIPT_DIR}/src/main.py"

# Default values
DEFAULT_PORT=8501
DEFAULT_HOST="localhost"
LOG_LEVEL="INFO"

# Help message
show_help() {
    cat << EOF
${CYAN}Agentic Ticker Launcher${NC}

${GREEN}Usage:${NC} ./launch.sh [OPTIONS]

${GREEN}Options:${NC}
  -h, --help              Show this help message
  -m, --mode MODE         Launch mode: streamlit, fastapi, or both (default: streamlit)
  -p, --port PORT         Port number (default: 8501 for streamlit, 8000 for fastapi)
  -H, --host HOST         Host to bind to (default: localhost)
  -d, --dev               Development mode with auto-reload
  -t, --test              Test mode (dry run)
  -s, --setup             Setup environment only (install dependencies)
  -c, --check             Check prerequisites only
  -v, --verbose           Verbose output
  --no-venv               Skip virtual environment activation
  --create-venv           Create virtual environment if it doesn't exist

${GREEN}Examples:${NC}
  ./launch.sh                           # Launch Streamlit app
  ./launch.sh -m fastapi               # Launch FastAPI app
  ./launch.sh -m both                  # Launch both apps
  ./launch.sh -d                       # Development mode
  ./launch.sh -s                       # Setup environment only
  ./launch.sh -p 8502                  # Custom port
  ./launch.sh --host 0.0.0.0           # Bind to all interfaces

${GREEN}Configuration:${NC}
   Configuration:
   - .env file            Recommended for local development (copy .env.template)
   - Environment variables Required for Streamlit Cloud deployment

${GREEN}Quick Start:${NC}
   1. Run: ./launch.sh -s              # Setup environment
   2. Configure your API keys:
      - Local: cp .env.template .env && nano .env
      - Cloud: Set environment variables in deployment
   3. Run: ./launch.sh                 # Start the application

EOF
}

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${VERBOSE:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Error handling
handle_error() {
    local line_number=$1
    log_error "Script failed on line $line_number"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_deps=()
    
    # Check Python
    if ! command_exists python3; then
        missing_deps+=("python3")
    else
        local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_info "Python version: $python_version"
        
        # Check Python version (require 3.8+)
        local major=$(echo $python_version | cut -d'.' -f1)
        local minor=$(echo $python_version | cut -d'.' -f2)
        if [[ $major -lt 3 ]] || [[ $major -eq 3 && $minor -lt 8 ]]; then
            log_error "Python 3.8+ required, found $python_version"
            exit 1
        fi
    fi
    
    # Check pip
    if ! command_exists pip3; then
        missing_deps+=("pip3")
    fi
    
    # Check git (optional)
    if ! command_exists git; then
        log_warn "git not found (optional for development)"
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install the missing dependencies and try again."
        exit 1
    fi
    
    log_info "Prerequisites check passed!"
}

# Setup virtual environment
setup_venv() {
    if [[ "${SKIP_VENV:-false}" == "true" ]]; then
        log_info "Skipping virtual environment setup"
        return
    fi
    
    if [[ ! -d "$VENV_DIR" ]]; then
        if [[ "${CREATE_VENV:-false}" == "true" ]]; then
            log_info "Creating virtual environment..."
            python3 -m venv "$VENV_DIR"
        else
            log_error "Virtual environment not found at $VENV_DIR"
            log_info "Run with --create-venv to create it automatically"
            exit 1
        fi
    fi
    
    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip >/dev/null 2>&1
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        log_error "Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
    
    pip install -r "$REQUIREMENTS_FILE" >/dev/null 2>&1
    log_info "Dependencies installed successfully!"
}

# Setup configuration file
setup_config() {
    # Check for .env file
    if [[ -f ".env" ]]; then
        log_info ".env file found"
    else
        log_warn "No .env file found. Creating .env template..."
        if [[ -f ".env.template" ]]; then
            cp .env.template .env
            log_info "Created .env from template. Please edit it with your API keys."
        else
            log_error "No .env.template found. Please create .env file."
        fi
    fi
}

# Check configuration
check_config() {
    # Check for environment variable first (highest priority)
    if [[ -n "${GEMINI_API_KEY:-}" ]]; then
        log_info "Gemini API key found in environment variables"
        return 0
    fi
    
    # Check for .env file
    if [[ -f ".env" ]]; then
        if grep -q "GEMINI_API_KEY=" .env && ! grep -q 'GEMINI_API_KEY=""' .env; then
            log_info "Gemini API key found in .env file"
            return 0
        else
            log_error "Gemini API key not set in .env file"
            log_info "Please edit .env and add your Google Gemini API key"
            exit 1
        fi
    fi
    
    # No configuration found
    log_error "No configuration found. Please set GEMINI_API_KEY environment variable or create .env file"
    exit 1
}

# Test mode
test_mode() {
    log_info "Running in test mode (dry run)..."
    log_info "Would execute the following:"
    
    case "$MODE" in
        streamlit)
            echo "  streamlit run $STREAMLIT_APP --server.port $PORT --server.address $HOST"
            ;;
        fastapi)
            echo "  python -m uvicorn main:app --host $HOST --port $PORT --reload"
            ;;
        both)
            echo "  streamlit run $STREAMLIT_APP --server.port $PORT --server.address $HOST"
            echo "  python -m uvicorn main:app --host $HOST --port $((PORT+1)) --reload"
            ;;
    esac
    
    exit 0
}

# Launch Streamlit
launch_streamlit() {
    log_info "Launching Streamlit application..."
    log_info "App: $STREAMLIT_APP"
    log_info "Host: $HOST"
    log_info "Port: $PORT"
    
    local streamlit_args=("run" "$STREAMLIT_APP" "--server.port" "$PORT" "--server.address" "$HOST")
    
    if [[ "${DEV_MODE:-false}" == "true" ]]; then
        streamlit_args+=("--server.runOnSave" "true")
        log_info "Development mode enabled (auto-reload on save)"
    fi
    
    log_info "Starting Streamlit server..."
    log_info "Access the application at: http://$HOST:$PORT"
    
    cd "$SCRIPT_DIR"
    exec streamlit "${streamlit_args[@]}"
}

# Launch FastAPI
launch_fastapi() {
    log_info "Launching FastAPI application..."
    log_info "App: $FASTAPI_APP"
    log_info "Host: $HOST"
    log_info "Port: $PORT"
    
    local uvicorn_args=("--host" "$HOST" "--port" "$PORT")
    
    if [[ "${DEV_MODE:-false}" == "true" ]]; then
        uvicorn_args+=("--reload")
        log_info "Development mode enabled (auto-reload on file changes)"
    fi
    
    log_info "Starting FastAPI server..."
    log_info "API documentation: http://$HOST:$PORT/docs"
    log_info "Health check: http://$HOST:$PORT/health"
    
    cd "$SCRIPT_DIR/src"
    exec python -m uvicorn main:app "${uvicorn_args[@]}"
}

# Launch both applications
launch_both() {
    log_info "Launching both Streamlit and FastAPI applications..."
    
    # Launch FastAPI in background
    launch_fastapi &
    local fastapi_pid=$!
    
    # Wait a bit for FastAPI to start
    sleep 2
    
    # Launch Streamlit in foreground
    launch_streamlit
    
    # Cleanup on exit
    trap "kill $fastapi_pid 2>/dev/null || true" EXIT
}

# Main function
main() {
    # Default values
    MODE="streamlit"
    PORT=$DEFAULT_PORT
    HOST=$DEFAULT_HOST
    DEV_MODE=false
    TEST_MODE=false
    SETUP_ONLY=false
    CHECK_ONLY=false
    VERBOSE=false
    SKIP_VENV=false
    CREATE_VENV=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -m|--mode)
                MODE="$2"
                shift 2
                ;;
            -p|--port)
                PORT="$2"
                shift 2
                ;;
            -H|--host)
                HOST="$2"
                shift 2
                ;;
            -d|--dev)
                DEV_MODE=true
                shift
                ;;
            -t|--test)
                TEST_MODE=true
                shift
                ;;
            -s|--setup)
                SETUP_ONLY=true
                shift
                ;;
            -c|--check)
                CHECK_ONLY=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --no-venv)
                SKIP_VENV=true
                shift
                ;;
            --create-venv)
                CREATE_VENV=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate mode
    case "$MODE" in
        streamlit|fastapi|both)
            ;;
        *)
            log_error "Invalid mode: $MODE. Use: streamlit, fastapi, or both"
            exit 1
            ;;
    esac
    
    # Welcome message
    echo -e "${PURPLE}"
    cat << "EOF"
     ___       _       _         _           _     
    / _ \     | |     | |       | |         | |    
   / /_\ \ ___| | __  | |  _ __ | |__   ___ | | __ 
   |  _  |/ __| |/ /  | | | '_ \| '_ \ / _ \| |/ / 
   | | | | (__|   <   | | | |_) | | | | (_) |   <  
   \_| |_/\___|_|\_\  | | | .__/|_| |_|\___/|_|\_\ 
                      |_| | |                       
                          |_|                       
EOF
    echo -e "${NC}"
    log_info "Starting $PROJECT_NAME launcher..."
    
    # Check prerequisites
    check_prerequisites
    
    if [[ "$CHECK_ONLY" == "true" ]]; then
        log_info "Prerequisites check completed!"
        exit 0
    fi
    
    # Setup virtual environment
    setup_venv
    
    # Setup configuration
    setup_config
    
    if [[ "$SETUP_ONLY" == "true" ]]; then
        log_info "Setup completed!"
        log_info "Next steps:"
        log_info "  1. Edit $CONFIG_FILE with your API keys"
        log_info "  2. Run: ./launch.sh"
        exit 0
    fi
    
    # Check configuration
    check_config
    
    # Install dependencies
    install_dependencies
    
    # Test mode
    if [[ "$TEST_MODE" == "true" ]]; then
        test_mode
    fi
    
    # Launch application
    log_info "Launching $PROJECT_NAME in $MODE mode..."
    
    case "$MODE" in
        streamlit)
            launch_streamlit
            ;;
        fastapi)
            launch_fastapi
            ;;
        both)
            launch_both
            ;;
    esac
}

# Run main function
main "$@"