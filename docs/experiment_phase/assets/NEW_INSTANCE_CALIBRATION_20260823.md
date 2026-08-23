# New-Instance Calibration — 2026-08-23

This instance's own 200-sample A/D latency baseline, run immediately
after `GATE_VERIFICATION_20260823.md` passed. Exists for the same
reason as `NEW_INSTANCE_CALIBRATION.md` (2026-08-09 instance): latency
is not portable across physical instances, so any latency claim this
session makes (including Task 3's adaptive-K latency check) must be
measured against this instance's own baseline, not an older instance's
numbers.

## Runs

200-sample subset (first 200 of the 1,698-sample Jin2022 test split,
same deterministic ordering as every prior instance's 200-sample
check), real checkpoint:

| config | MAE | latency median | avg target forwards |
|---|---:|---:|---:|
| A. baseline (no selector, no speculative) | 11.730207886356089 | 462.69 ms | 20.00 |
| D. RecentK-2 + speculative (th=0.35, γ=8) | 10.465869078778512 | 99.29 ms | 4.00 |

Raw output: `results/speculative/20260823T063954Z/` (config A),
`results/speculative/20260823T064155Z/` (config D).

## Cross-instance consistency check

MAE matches the 2026-08-09 instance's 200-sample check to 6+ significant
figures (A: 11.730208 vs. 11.730207886356089; D: 10.465869 vs.
10.465869078778512) — as close to exact reproduction as fp16 GPU noise
allows. Confirms this instance's asset stack behaves identically to the
two prior instances on accuracy.

## Latency: this instance is the fastest of the three so far

| | this instance (200-sample) | 2026-08-09 instance | 2026-08-02 instance (full 1,698) |
|---|---:|---:|---:|
| A latency median | 462.69 ms | 673.17 ms | 571.7 ms |
| D latency median | 99.29 ms | 142.06 ms | 122.2 ms |
| A/D speedup ratio | 4.66x | 4.74x | 4.68x |

Consistent with `GATE_VERIFICATION_20260823.md`'s 50-sample baseline
(467 ms) — this physical GPU/host runs faster in absolute terms than
either prior instance. **Per the established rule, this instance's
latency numbers are never diffed against the 2026-08-02 or 2026-08-09
instances' absolute latency in any deliverable this session** —
`presentation_20260816/` uses only the 2026-08-02 full-1,698 numbers
(explicitly footnoted per-figure), and Task 3's adaptive-K latency
check (if it proceeds) compares only against the two rows in this table.

## Conclusion

This instance is confirmed trustworthy for GPU work: accuracy
reproduces near-exactly across three independent instances now, latency
is instance-specific as expected and stays correctly siloed by session
in this project's reporting.
