#!/usr/bin/env bash
set -euo pipefail
echo "1) Installing dependencies..."
python -m pip install -r requirements.txt
echo "2) Running retrain check (will train if new data)..."
python retrain.py
echo "3) Building docker image..."
docker build -t cicd-sales-prediction:latest .
echo "4) (Optional) Run container locally"
echo "   docker run -p 5000:5000 cicd-sales-prediction:latest"
