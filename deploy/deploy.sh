#!/usr/bin/env bash
# ETD-XAI Enterprise — deploy / update on a fresh Ubuntu (DigitalOcean) droplet.
# Usage:  sudo bash deploy/deploy.sh
set -euo pipefail

APP_DIR=/opt/etd-xai
DATA_DIR=/var/lib/etd-xai
LOG_DIR=/var/log/etd-xai
REPO="${REPO:-https://github.com/me30101152101959-tech/Theft.git}"
BRANCH="${BRANCH:-main}"

echo "==> Installing system packages"
apt-get update -y
apt-get install -y python3 python3-venv python3-pip nodejs npm nginx git curl

echo "==> Fetching source into $APP_DIR"
mkdir -p "$APP_DIR" "$DATA_DIR" "$LOG_DIR"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull origin "$BRANCH"
else
  git clone -b "$BRANCH" "$REPO" "$APP_DIR"
fi

echo "==> Backend: virtualenv + dependencies"
cd "$APP_DIR/backend"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt gunicorn

echo "==> Frontend: build"
cd "$APP_DIR/frontend"
npm ci
npm run build

echo "==> Environment file"
[ -f "$APP_DIR/.env" ] || cp "$APP_DIR/.env.example" "$APP_DIR/.env"

echo "==> systemd service"
install -m 644 "$APP_DIR/deploy/etd-xai.service" /etc/systemd/system/etd-xai.service
systemctl daemon-reload
systemctl enable etd-xai
systemctl restart etd-xai

echo "==> Nginx"
install -m 644 "$APP_DIR/deploy/nginx-site.conf" /etc/nginx/sites-available/etd-xai
ln -sf /etc/nginx/sites-available/etd-xai /etc/nginx/sites-enabled/etd-xai
nginx -t && systemctl restart nginx

chown -R www-data:www-data "$APP_DIR" "$DATA_DIR" "$LOG_DIR"

echo "==> Done. Check: systemctl status etd-xai  |  curl http://localhost/api/health"
echo "    For HTTPS:  sudo certbot --nginx -d your-domain.com"
