#!/usr/bin/env bash
# Install starnose — context window observability for LLM agents
# Usage: curl -fsSL https://raw.githubusercontent.com/eitanlebras/starnose/main/scripts/install.sh | bash

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
DIM='\033[2m'
RESET='\033[0m'

echo -e "${BOLD}starnose${RESET} installer"
echo ""

# Check for Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python 3.10+ is required but not found."
    echo "Install Python from https://python.org or via your package manager."
    exit 1
fi

# Check Python version
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "Error: Python 3.10+ required, found $PY_VERSION"
    exit 1
fi

echo -e "  Python: ${GREEN}$PY_VERSION${RESET}"

# Prefer pipx for isolated install
if command -v pipx &>/dev/null; then
    echo -e "  Method: ${GREEN}pipx${RESET}"
    echo ""
    pipx install starnose
elif command -v pip3 &>/dev/null; then
    echo -e "  Method: ${GREEN}pip3${RESET}"
    echo ""
    pip3 install --user starnose
elif command -v pip &>/dev/null; then
    echo -e "  Method: ${GREEN}pip${RESET}"
    echo ""
    pip install --user starnose
else
    echo "Error: pip not found. Install pip first."
    exit 1
fi

echo ""

# Verify installation
if command -v snose &>/dev/null; then
    echo -e "${GREEN}Installed!${RESET} Run: ${BOLD}snose run python my_agent.py${RESET}"
else
    echo -e "${GREEN}Installed!${RESET}"
    echo ""
    echo "If 'snose' isn't in your PATH, add this to your shell profile:"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
    echo ""
    echo "Then run: snose run python my_agent.py"
fi
