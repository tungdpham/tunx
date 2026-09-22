#!/usr/bin/env python3
"""Launch with torchrun; see README.md for single-node and multi-node examples."""
from distributed_benchmark import main

if __name__ == '__main__':
    main('fsdp')
