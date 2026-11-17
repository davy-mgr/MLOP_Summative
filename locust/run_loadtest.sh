#!/usr/bin/env bash
# Runs the same Locust test against 1, 2 and 3 API containers behind Nginx.
# Usage: bash locust/run_loadtest.sh [users] [spawn_rate] [duration]
set -euo pipefail
USERS=${1:-50}; RATE=${2:-10}; DUR=${3:-60s}
mkdir -p results
for N in 1 2 3; do
  echo "=== $N API container(s) ==="
  docker compose up -d --build --scale api=$N
  sleep 20   # let TensorFlow load in each container
  python -m locust -f locust/locustfile.py --headless -u "$USERS" -r "$RATE" -t "$DUR" \
         --host http://localhost:8000 --csv "results/containers_$N" --only-summary
done
python locust/summarize.py
