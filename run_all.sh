#!/usr/bin/env bash
# Usage: bash run_all.sh /path/to/oulad [results]
set -e
python run_all.py --oulad "${1:-$OULAD_DIR}" --out "${2:-${OUT_DIR:-results}}"
