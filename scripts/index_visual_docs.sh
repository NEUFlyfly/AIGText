#!/usr/bin/env bash
set -u

# Build text and visual RAG indexes after printing the local asset readiness report.
# Usage:
#   bash scripts/index_visual_docs.sh
#   PYTHON=/path/to/python bash scripts/index_visual_docs.sh --fixtures

PYTHON_BIN="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

echo "========================================"
echo "  AIGText - Visual RAG Index Preparation"
echo "========================================"
echo ""

echo "[0/3] Checking local assets..."
"$PYTHON_BIN" -m src.rag.check_visual_rag_assets
asset_exit=$?

echo ""
echo "[1/3] Building text vector index..."
"$PYTHON_BIN" -m src.rag.index
text_exit=$?

echo ""
echo "[2/3] Building visual vector index..."
"$PYTHON_BIN" -m src.rag.visual_index "$@"
visual_exit=$?

echo ""
echo "========================================"
if [ $asset_exit -eq 0 ] && [ $text_exit -eq 0 ] && [ $visual_exit -eq 0 ]; then
    echo "  Index preparation complete"
else
    echo "  Index preparation failed (assets: $asset_exit, text: $text_exit, visual: $visual_exit)"
fi
echo "========================================"

if [ $text_exit -ne 0 ]; then
    exit $text_exit
fi
exit $visual_exit
