# New-Instance Calibration — 2026-08-09

Optional reinforcement per this session's Task 3, run only because
Gate-B (`GATE_A_VERIFICATION.md`) passed. Establishes a baseline point
for future latency comparisons on *this* physical instance/GPU
(RTX 4090, same as before, but a different physical unit than the
2026-08-02 session's) — not a new scientific finding.

## Runs

200-sample subset (first 200 of the 1,698-sample Jin2022 test split,
deterministic ordering — same subset `run_speculative_benchmark.py`
would always produce for `--num-samples 200`), real checkpoint:

| config | MAE | latency median | avg target forwards |
|---|---:|---:|---:|
| A. baseline (no selector, no speculative) | 11.730208 | 673.17 ms | 20.00 |
| D. RecentK-2 + speculative (th=0.35, γ=8) | 10.465869 | 142.06 ms | 4.01 |

Raw output: `results/speculative/20260809T075002Z/` (config A + C),
`results/speculative/20260809T075305Z/` (config B + D).

## MAE direction check: consistent

D improves over A in both this 200-sample check (11.730 → 10.466, −10.8%)
and the full 1,698-sample reference from 2026-08-02
(12.799 → 10.895, −14.9%). Same direction, roughly comparable
magnitude. The absolute MAE values differ from the full-1,698 reference
(11.73 vs. 12.80 for A; 10.47 vs. 10.90 for D) because this is a
different, smaller sample subset (the first 200 of 1,698, not the full
population) — the same effect already documented for the 50-sample
smoke check (11.037 vs. 12.799 full-population baseline MAE) in
`ASSET_RECOVERY_VERIFICATION.md`. Not a discrepancy; expected subset
variance, not evidence of anything wrong with this instance's setup.

## Latency: this instance runs slower in absolute terms

| | this instance (200-sample) | 2026-08-02 instance (full 1,698) | ratio |
|---|---:|---:|---:|
| A latency median | 673.17 ms | 571.7 ms | 1.18x |
| D latency median | 142.06 ms | 122.2 ms | 1.16x |
| A/D speedup ratio | 4.74x | 4.68x | — |

Absolute latency on this instance is ~16–18% higher than the
2026-08-02 instance for both configs — expected for a different
physical GPU/host, not a regression. **Do not compare absolute latency
numbers across instances directly** — different physical hardware,
thermal state, and co-tenancy all affect wall-clock latency
independent of anything this project's code does.

The speedup *ratio* (A/D, 4.74x here vs. 4.68x previously) happens to
be close in this specific comparison, but that should not be read as a
general guarantee that speedup ratios transfer across instances either
— it is a two-point comparison, not a controlled multi-run study of
ratio stability. **The correct general framing: the 4.68x speedup
figure from the 2026-08-02 report is a same-instance relative value
(A vs. D measured back-to-back on identical hardware in one session),
not a portable absolute latency claim.** Any future latency comparison
should be made within a single instance/session, the same way the
original 2026-08-02 comparison was.

## Conclusion

This instance's setup (checkpoint + dataset + base weights) is
confirmed to reproduce accuracy behavior consistent with the prior
instance (MAE direction and rough magnitude both hold). Latency
absolute values are instance-specific and not directly comparable;
future full-scale (1,698-sample) benchmark runs on this instance should
establish their own from-scratch latency baseline rather than citing
the 2026-08-02 latency numbers as if they were expected to reproduce
exactly.
