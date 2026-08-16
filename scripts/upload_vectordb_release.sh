#!/bin/bash
# Upload the cloud vectordb as a GitHub Release asset
# Usage: ./scripts/upload_vectordb_release.sh [tag]
#
# Prerequisites: gh CLI authenticated (gh auth login)

set -e

TAG="${1:-v1.1.0}"
REPO="ericnc09/3gpp-rag-assistant"
ARCHIVE="/tmp/vectordb_cloud.tar.gz"
DB_PATH="data/vectordb_cloud"

echo "=== Packaging vectordb_cloud ==="
if [ ! -d "$DB_PATH" ]; then
    echo "ERROR: $DB_PATH not found. Build it first:"
    echo "  python scripts/build_index_cloud.py"
    exit 1
fi

echo "Compressing $DB_PATH..."
tar czf "$ARCHIVE" -C data vectordb_cloud
SIZE=$(ls -lh "$ARCHIVE" | awk '{print $5}')
echo "Archive size: $SIZE"

echo ""
echo "=== Creating GitHub Release $TAG ==="
gh release create "$TAG" \
    --repo "$REPO" \
    --title "3GPP RAG Assistant $TAG — Cloud Index" \
    --notes "Pre-built vector index (ChromaDB + ONNX embeddings) for Streamlit Cloud deployment.

**Contents:** 37 3GPP specs, ~43K chunks indexed with ChromaDB's built-in all-MiniLM-L6-v2 ONNX embeddings.

**Usage:** Set \`VECTORDB_URL\` in Streamlit Cloud secrets to the download URL of \`vectordb_cloud.tar.gz\` from this release." \
    || echo "Release $TAG already exists, uploading asset..."

echo ""
echo "=== Uploading archive ==="
gh release upload "$TAG" "$ARCHIVE" --repo "$REPO" --clobber

echo ""
echo "=== Done ==="
DOWNLOAD_URL="https://github.com/$REPO/releases/download/$TAG/vectordb_cloud.tar.gz"
echo "Download URL: $DOWNLOAD_URL"
echo ""
echo "Set this in Streamlit Cloud secrets:"
echo "  VECTORDB_URL = \"$DOWNLOAD_URL\""
