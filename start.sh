#!/bin/sh

if [ -z "$PORT" ]; then
  export PORT=8000
fi

echo "Starting server on port $PORT..."

exec gunicorn app:app \
  --bind 0.0.0.0:$PORT \
  --workers 1 \
  --timeout 120 \
  --max-requests 100 \
  --log-level info
