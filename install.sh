#!/usr/bin/env bash
# Install prompt-enhancer-plugin for Hermes Agent (v2.2)
# Usage: curl -fsSL <url> | bash
#        or: bash install.sh

set -euo pipefail

PLUGIN_NAME="prompt-enhancer-plugin"
PLUGIN_DIR="${HOME}/.hermes/plugins/${PLUGIN_NAME}"
CONFIG_FILE="${HOME}/.hermes/config.yaml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Hermes config directory exists
if [ ! -d "${HOME}/.hermes" ]; then
    log_warn "~/.hermes directory does not exist. Creating it..."
    mkdir -p "${HOME}/.hermes"
fi

# Create plugin directory
log_info "Creating plugin directory: ${PLUGIN_DIR}"
mkdir -p "${PLUGIN_DIR}"

# Detect source directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy files from the repo/source directory
log_info "Installing plugin files from ${SCRIPT_DIR}..."
cp "${SCRIPT_DIR}/__init__.py" "${PLUGIN_DIR}/"
cp "${SCRIPT_DIR}/plugin.yaml" "${PLUGIN_DIR}/"
cp "${SCRIPT_DIR}/README.md" "${PLUGIN_DIR}/" 2>/dev/null || true

# Create logs directory
mkdir -p "${PLUGIN_DIR}/logs"

log_ok "Plugin files installed to ${PLUGIN_DIR}"

# Check if plugin is already enabled in config
if [ -f "${CONFIG_FILE}" ]; then
    if grep -q "prompt-enhancer-plugin" "${CONFIG_FILE}" 2>/dev/null; then
        log_ok "prompt-enhancer-plugin already referenced in config.yaml"
    else
        log_warn "Add the following to ~/.hermes/config.yaml to enable:"
        echo ""
        echo "plugins:"
        echo "  enabled:"
        echo "    - prompt-enhancer-plugin"
        echo ""
    fi
else
    log_warn "~/.hermes/config.yaml not found. Create it with:"
    echo ""
    echo "plugins:"
    echo "  enabled:"
    echo "    - prompt-enhancer-plugin"
    echo ""
fi

# Check for httpx dependency
log_info "Checking dependencies..."
if python3 -c "import httpx" 2>/dev/null; then
    log_ok "httpx is already installed"
else
    log_warn "httpx not found. Install with: pip install httpx"
fi

log_ok "Installation complete!"
log_info "Start a new session to load the plugin:"
echo "  hermes /reset"
echo "  # or exit and restart hermes"
echo ""
log_info "Verify it's working:"
echo "  cat ~/.hermes/plugins/prompt-enhancer-plugin/logs/enhancer.log"
