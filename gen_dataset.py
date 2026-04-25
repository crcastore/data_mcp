"""Generate an arbitrarily large CSV with nonlinear relationships.

Usage:
    python gen_dataset.py <num_rows> <num_cols> [output.csv]

Dependencies: numpy, scikit-learn (both in the standard scientific stack).
"""

import sys
import csv

import numpy as np
from sklearn.datasets import make_regression


def generate(n_rows: int, n_cols: int, path: str) -> None:
    rng = np.random.default_rng(42)

    # Start with a linear base from make_regression.
    n_informative = max(n_cols // 2, 1)
    X, _ = make_regression(
        n_samples=n_rows,
        n_features=n_informative,
        n_informative=n_informative,
        noise=10.0,
        random_state=42,
    )

    # Apply nonlinear transforms to build the remaining columns.
    cols = [X[:, i] for i in range(n_informative)]

    while len(cols) < n_cols:
        # Pick two existing columns and combine with a random nonlinear op.
        a = cols[rng.integers(len(cols))]
        b = cols[rng.integers(len(cols))]
        op = rng.integers(5)
        if op == 0:
            cols.append(np.sin(a) * b + rng.normal(0, 0.5, n_rows))
        elif op == 1:
            cols.append(a ** 2 + rng.normal(0, 0.5, n_rows))
        elif op == 2:
            cols.append(np.exp(np.clip(a / a.std(), -3, 3)) + rng.normal(0, 0.1, n_rows))
        elif op == 3:
            cols.append(a * b + rng.normal(0, 0.5, n_rows))
        else:
            cols.append(np.tanh(a) + np.sqrt(np.abs(b)) + rng.normal(0, 0.2, n_rows))

    headers = [f"x{i}" for i in range(n_cols)]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row_idx in range(n_rows):
            writer.writerow([f"{cols[c][row_idx]:.6f}" for c in range(n_cols)])

    print(f"Wrote {n_rows} rows × {n_cols} cols → {path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <num_rows> <num_cols> [output.csv]")
        sys.exit(1)

    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "dataset.csv"
    generate(rows, cols, out)
