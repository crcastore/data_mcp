#!/usr/bin/env python3
"""
Generate test CSV files for mcp profiling.

Outputs:
  test_data/simple.csv          - clean numeric + categorical, no missing
  test_data/messy.csv           - lots of nulls, skewed distributions, sparse cols
  test_data/wide.csv            - many columns (numeric & categorical mix)
  test_data/single_column.csv   - edge case: one column
  test_data/large.csv           - 50k rows for performance testing
"""

import csv
import math
import os
import random

random.seed(42)
OUT = "test_data"
os.makedirs(OUT, exist_ok=True)


# ── helpers ─────────────────────────────────────────────────────────────────

def write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows, {len(headers)} cols)")


def maybe_null(value, null_rate=0.0):
    return "" if random.random() < null_rate else value


def normal(mu=0.0, sigma=1.0):
    return random.gauss(mu, sigma)


def lognormal(mu=0.0, sigma=1.0):
    return math.exp(random.gauss(mu, sigma))


def choice(options):
    return random.choice(options)


# ── 1. simple.csv ────────────────────────────────────────────────────────────
# Clean data: numeric columns with clear distributions + a few categoricals.

N = 500
headers = ["age", "salary", "score", "height_cm", "category", "region"]
rows = []
for _ in range(N):
    age    = round(random.uniform(18, 70), 1)
    salary = round(lognormal(10.5, 0.5), 2)
    score  = round(normal(75, 10), 2)
    height = round(normal(170, 10), 1)
    cat    = choice(["A", "B", "C", "D"])
    region = choice(["North", "South", "East", "West"])
    rows.append([age, salary, score, height, cat, region])

write_csv(f"{OUT}/simple.csv", headers, rows)


# ── 2. messy.csv ─────────────────────────────────────────────────────────────
# Missing values, highly skewed numeric, sparse binary flag, low-cardinality cat.

N = 300
headers = ["id", "revenue", "clicks", "label", "flag", "tier"]
rows = []
for i in range(N):
    revenue = maybe_null(round(lognormal(5, 2), 2), null_rate=0.15)
    clicks  = maybe_null(round(random.expovariate(1 / 50)), null_rate=0.20)
    label   = maybe_null(choice(["yes", "no", "maybe"]), null_rate=0.10)
    flag    = maybe_null(choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), null_rate=0.05)  # sparse
    tier    = maybe_null(choice(["free", "pro", "enterprise"]), null_rate=0.08)
    rows.append([i + 1, revenue, clicks, label, flag, tier])

write_csv(f"{OUT}/messy.csv", headers, rows)


# ── 3. wide.csv ──────────────────────────────────────────────────────────────
# 20 numeric + 5 categorical columns to exercise the correlation matrix.

N = 200
num_cols  = [f"x{i}" for i in range(1, 21)]
cat_cols  = ["cat_a", "cat_b", "cat_c", "cat_d", "cat_e"]
headers   = num_cols + cat_cols

rows = []
for _ in range(N):
    base = normal()
    nums = [round(base * random.uniform(0.5, 1.5) + normal(0, 0.3), 4)
            for _ in num_cols]
    cats = [
        choice(["alpha", "beta", "gamma"]),
        choice(["X", "Y"]),
        choice(["low", "mid", "high", "very_high"]),
        choice(["T", "F"]),
        choice(["p", "q", "r", "s", "t"]),
    ]
    rows.append(nums + cats)

write_csv(f"{OUT}/wide.csv", headers, rows)


# ── 4. single_column.csv ─────────────────────────────────────────────────────
# Edge case: exactly one numeric column.

N = 100
headers = ["value"]
rows = [[round(normal(50, 5), 3)] for _ in range(N)]
write_csv(f"{OUT}/single_column.csv", headers, rows)


# ── 5. large.csv ─────────────────────────────────────────────────────────────
# 50k rows for profiling performance.

N = 50_000
headers = ["ts", "amount", "quantity", "price", "status", "country"]
rows = []
for i in range(N):
    amount   = round(lognormal(3, 1), 2)
    quantity = random.randint(1, 100)
    price    = round(random.uniform(0.5, 999.99), 2)
    status   = choice(["pending", "complete", "failed", "refunded"])
    country  = choice(["US", "UK", "DE", "FR", "JP", "BR", "IN", "CA"])
    rows.append([i, amount, quantity, price, status, country])

write_csv(f"{OUT}/large.csv", headers, rows)


print(f"\nAll done. Run with:\n"
      f"  ./target/debug/mcp {OUT}/simple.csv\n"
      f"  ./target/debug/mcp {OUT}/messy.csv\n"
      f"  ./target/debug/mcp {OUT}/wide.csv\n"
      f"  ./target/debug/mcp {OUT}/single_column.csv\n"
      f"  ./target/debug/mcp {OUT}/large.csv")
