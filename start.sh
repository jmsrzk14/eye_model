#!/bin/bash
# Start script untuk Railway

# Set default port jika tidak ada
if [ -z "$PORT" ]; then
  export PORT=8000
fi

echo "Starting server on port $PORT..."

# Run gunicorn
exec gunicorn app:app \
  --bind 0.0.0.0:$PORT \
  --workers 1 \
  --timeout 120 \
  --max-requests 100 \
  --log-level info