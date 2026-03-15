#!/bin/sh
set -e

BIN="./target/release/mcp"
CSV="$PWD/test_data/simple.csv"

echo "=== Building release binary ==="
cargo build --release

echo ""
echo "=== 1. Initialize ==="
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  | "$BIN" | python3 -m json.tool

echo ""
echo "=== 2. List tools ==="
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  | "$BIN" | python3 -m json.tool

echo ""
echo "=== 3. Load dataset + row_count + mean ==="
(
printf '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"load_dataset","arguments":{"path":"%s"}}}\n' "$CSV"
printf '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"row_count","arguments":{}}}\n'
printf '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"mean","arguments":{"column":"age"}}}\n'
) | "$BIN" | while IFS= read -r line; do echo "$line" | python3 -m json.tool; done

echo ""
echo "=== 4. Load dataset + all reservoir tools ==="
(
printf '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"load_dataset","arguments":{"path":"%s"}}}\n' "$CSV"
printf '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"surrogate_test","arguments":{"column":"age","num_surrogates":50}}}\n'
printf '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"bds_test","arguments":{"column":"age","embedding_dim":3,"epsilon":20.0}}}\n'
printf '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"lyapunov_exponent","arguments":{"column":"age","embedding_dim":3,"tau":1}}}\n'
printf '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"dependence_comparison","arguments":{"column":"age","max_lag":5}}}\n'
printf '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"delay_embedding","arguments":{"column":"age","max_dim":10}}}\n'
printf '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"memory_profile","arguments":{"column":"age","max_lag":5}}}\n'
) | "$BIN" | while IFS= read -r line; do echo "$line" | python3 -m json.tool; done

echo ""
echo "=== All smoke tests passed ==="
