#!/usr/bin/env python3
"""Generate nonlinear.csv and linear.csv in test_data/."""
import csv, math, random

random.seed(99)
N = 1000

# === nonlinear.csv ===
# Logistic map (chaotic, r=3.99), Henon map, nonlinear transforms
x = 0.1
logistic = []
for _ in range(N):
    x = 3.99 * x * (1 - x)
    logistic.append(x)

hx, hy = 0.1, 0.1
henon = []
for _ in range(N):
    hx_new = 1 - 1.4 * hx**2 + hy
    hy = 0.3 * hx
    hx = hx_new
    henon.append(hx)

with open("test_data/nonlinear.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["logistic_map", "henon_map", "sin_product", "exp_mod", "cubic_mix"])
    for i in range(N):
        t = i / N * 4 * math.pi
        sin_prod = math.sin(t) * math.cos(2.7 * t + logistic[i])
        exp_mod = math.exp(-0.5 * logistic[i]) * math.sin(3 * t)
        cubic = logistic[i]**3 - 0.5 * henon[i]**2 + 0.1 * math.sin(7 * t)
        w.writerow([
            round(logistic[i], 6),
            round(henon[i], 6),
            round(sin_prod, 6),
            round(exp_mod, 6),
            round(cubic, 6),
        ])
print("wrote test_data/nonlinear.csv (1000 rows, chaotic dynamics)")

# === linear.csv ===
# Linear trends, Gaussian noise, simple linear combinations
with open("test_data/linear.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trend", "noisy_trend", "gaussian", "linear_combo", "uniform"])
    for i in range(N):
        trend = 2.0 * i / N + 10.0
        noise = random.gauss(0, 0.3)
        noisy = trend + noise
        gauss = random.gauss(50, 5)
        combo = 0.7 * trend + 0.3 * gauss + random.gauss(0, 0.1)
        unif = random.uniform(0, 100)
        w.writerow([
            round(trend, 6),
            round(noisy, 6),
            round(gauss, 6),
            round(combo, 6),
            round(unif, 6),
        ])
print("wrote test_data/linear.csv (1000 rows, linear/Gaussian)")
