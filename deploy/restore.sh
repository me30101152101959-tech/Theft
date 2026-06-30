#!/usr/bin/env bash
# Restore a backup produced by backup.sh.
# Usage:  sudo bash deploy/restore.sh /var/backups/etd-xai/etd-xai_backup_YYYYMMDD_HHMMSS.tar.gz
set -euo pipefail

ARCHIVE="${1:?Usage: restore.sh <backup.tar.gz>}"
DB_PATH="${DATABASE_PATH:-/var/lib/etd-xai/etd_xai.db}"
APP_DIR=/opt/etd-xai

[ -f "$ARCHIVE" ] || { echo "Archive not found: $ARCHIVE"; exit 1; }

echo "==> Stopping service"
systemctl stop etd-xai || true

TMP=$(mktemp -d)
tar -xzf "$ARCHIVE" -C "$TMP"

if [ -f "$TMP/etd_xai.db" ]; then
  mkdir -p "$(dirname "$DB_PATH")"
  cp "$TMP/etd_xai.db" "$DB_PATH"
  echo "==> Restored database → $DB_PATH"
fi
if [ -d "$TMP/uploads" ]; then
  cp -r "$TMP/uploads/." "$APP_DIR/backend/uploads/"
  echo "==> Restored uploads → $APP_DIR/backend/uploads"
fi

rm -rf "$TMP"
chown -R www-data:www-data "$APP_DIR/backend/uploads" "$(dirname "$DB_PATH")" || true

echo "==> Starting service"
systemctl start etd-xai
echo "==> Restore complete."
