#!/bin/bash
# deploy_worker.sh
# Run this on the worker node (10.10.0.1)

# Usage: ./deploy_worker.sh [listen_port]
# Default port is 8001 (as specified in eval_distributed.json)

PORT=${1:-8001}

echo "Starting TCP worker on port $PORT..."
./build/examples/tcp_worker --gpu 0 $PORT
