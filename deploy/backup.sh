#!/usr/bin/env bash
# Back up the SQLite DB, uploaded models and datasets.
# Usage:  sudo bash deploy/backup.sh [destination_dir]
set -euo pipefail

DEST="${1:-/var/backups/etd-xai}"
DB_PATH="${DATABASE_PATH:-/var/lib/etd-xai/etd_xai.db}"
APP_DIR=/opt/etd-xai
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$DEST/etd-xai_backup_$STAMP.tar.gz"

mkdir -p "$DEST"
echo "==> Backing up to $OUT"

# Use SQLite's online backup so we capture a consistent snapshot.
TMP=$(mktemp -d)
if [ -f "$DB_PATH" ]; then
  sqlite3 "$DB_PATH" ".backup '$TMP/etd_xai.db'"
fi

tar -czf "$OUT" \
  -C "$TMP" $( [ -f "$TMP/etd_xai.db" ] && echo etd_xai.db ) \
  -C "$APP_DIR/backend" uploads 2>/dev/null || true

rm -rf "$TMP"
echo "==> Backup complete: $OUT"
ls -lh "$OUT"
