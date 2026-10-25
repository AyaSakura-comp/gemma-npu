#!/bin/bash
# usage: run_flm_bench.sh MODEL_TAG PORT  -- start `flm serve`, benchmark, stop it
TAG=$1; PORT=${2:-52701}
flm serve "$TAG" --port "$PORT" > "flm_$PORT.log" 2>&1 &
PID=$!
for i in $(seq 1 90); do curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 2; done
source ~/src/Triton-XDNA/sandbox/bin/activate
python bench_flm.py "$PORT" "$TAG"
kill $PID; wait $PID 2>/dev/null
