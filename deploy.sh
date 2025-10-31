#!/usr/bin/env bash
set -euo pipefail
IMAGE_NAME=${1:-cicd-sales-prediction:latest}
CONTAINER_NAME=${2:-cicd-sales-app}
echo "Stopping existing container (if any)..."
docker rm -f $CONTAINER_NAME || true
echo "Starting container..."
docker run -d --name $CONTAINER_NAME -p 5000:5000 $IMAGE_NAME
echo "App should be available at http://localhost:5000"
