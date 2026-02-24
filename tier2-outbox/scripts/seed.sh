#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$SCRIPT_DIR/../.env" ]; then
  export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
fi

echo "Seeding Postgres with 1,000 products from HuggingFace..."
PYTHONPATH="$ROOT_DIR" python -m shared.seed

echo ""
echo "Postgres is seeded."
echo "The outbox worker will sync products to Qdrant as events are processed."
echo "To check worker status: curl http://localhost:8000/sync/status"