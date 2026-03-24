#!/usr/bin/env bash
# Update the Homebrew formula after a PyPI release.
# Usage: ./scripts/update-homebrew.sh 0.1.0

set -euo pipefail

VERSION="${1:?Usage: $0 <version>}"
URL="https://files.pythonhosted.org/packages/source/s/starnose/starnose-${VERSION}.tar.gz"

echo "Downloading ${URL}..."
SHA256=$(curl -fsSL "$URL" | shasum -a 256 | cut -d' ' -f1)

echo "Version: ${VERSION}"
echo "SHA256:  ${SHA256}"
echo ""

FORMULA="Formula/starnose.rb"
if [ -f "$FORMULA" ]; then
    sed -i '' "s|url \".*\"|url \"${URL}\"|" "$FORMULA"
    sed -i '' "s|sha256 \".*\"|sha256 \"${SHA256}\"|" "$FORMULA"
    echo "Updated ${FORMULA}"
    echo ""
    echo "Next steps:"
    echo "  1. git add Formula/starnose.rb"
    echo "  2. git commit -m 'brew: update to ${VERSION}'"
    echo "  3. git push"
else
    echo "Formula not found at ${FORMULA}"
    exit 1
fi
