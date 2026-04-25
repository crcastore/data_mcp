#!/usr/bin/env bash
set -euo pipefail

BINARY="./target/release/mcp"
CSV="/tmp/bench_1M_50col.csv"
RUNS=4

# Build release binary if not already built.
cargo build --release 2>&1

# Generate 1M row × 50 column CSV if it doesn't exist.
if [[ ! -f "$CSV" ]]; then
    echo "Generating $CSV ..."
    python3 - <<'EOF'
import csv, math, sys

rows = 1_000_000
cols = 50
path = "/tmp/bench_1M_50col.csv"

with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"col{i}" for i in range(cols)])
    for r in range(rows):
        w.writerow([math.sin((r + c) * 0.3) * (c + 1) * 10 for c in range(cols)])

print(f"Written {rows} rows × {cols} cols to {path}")
EOF
fi

echo ""
echo "Running $BINARY on $CSV ($RUNS times)"
echo "────────────────────────────────────────"

for i in $(seq 1 $RUNS); do
    start=$( python3 -c "import time; print(int(time.time() * 1000))" )
    "$BINARY" "$CSV" > /dev/null
    end=$( python3 -c "import time; print(int(time.time() * 1000))" )
    echo "Run $i: $(( end - start )) ms"
done
