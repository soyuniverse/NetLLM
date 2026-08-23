# Summary Table — A/B/C/D (full 1,698-sample, same checkpoint)

Source: `results/speculative/consolidated/final_table.csv` (results/speculative/20260802T075640Z (full 1,698-sample baseline); results/speculative/20260802T082009Z (full 1,698-sample, 4 speculative configs); results/speculative/20260802T101802Z (full 1,698-sample, Selector x Speculative ablation)).

| config | MAE (deg) | ΔMAE % vs A | latency median (ms) | speedup vs A | avg forward count |
|---|---:|---:|---:|---:|---:|
| A. baseline | 12.799 | +0.00% | 571.7 | 1.00x | 20.00 |
| B. RecentK-2 only | 10.847 | -15.25% | 623.0 | 0.92x | 20.00 |
| C. Speculative only | 12.831 | +0.26% | 124.4 | 4.59x | 4.21 |
| D. RecentK-2 + Speculative | 10.895 | -14.87% | 122.2 | 4.68x | 4.01 |
