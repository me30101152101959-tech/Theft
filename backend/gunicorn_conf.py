"""
Gunicorn configuration for ETD-XAI Enterprise (production).
Run:  gunicorn -c gunicorn_conf.py main:app
"""
import os

bind = f"{os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '8000')}"

# TensorFlow is heavy and not fork-safe with many workers; keep it small.
# 2 workers handles concurrent requests while bounding memory (each loads the model).
workers = int(os.environ.get("GUNICORN_WORKERS", "2"))
worker_class = "uvicorn.workers.UvicornWorker"

# Large batch predictions can take minutes — generous timeout.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "600"))
graceful_timeout = 60
keepalive = 5

loglevel = os.environ.get("LOG_LEVEL", "info")
accesslog = os.environ.get("ACCESS_LOG", "-")   # stdout
errorlog = os.environ.get("ERROR_LOG", "-")     # stderr

preload_app = False  # each worker loads its own TF model copy
