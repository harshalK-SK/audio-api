#!/bin/bash
echo "Installing system dependencies..."
apt-get update
apt-get install -y ffmpeg libsndfile1
echo "Dependencies installed. Starting application..."
gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers=2 --worker-class uvicorn.workers.UvicornWorker app:app
