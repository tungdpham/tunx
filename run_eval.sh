#!/bin/bash
# run_eval.sh
# Run this on the coordinator node (10.10.0.2)

# Ensure the project is built
cd build && make -j8 && cd ..

echo "Starting coordinator and local worker..."
./build/examples/tcp_coordinator -c configs/eval_distributed.json
