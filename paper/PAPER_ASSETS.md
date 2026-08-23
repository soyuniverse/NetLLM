# Paper Assets — Claims, Tables, Reproducibility, Gaps (2026-08-23)

Companion to `paper/figures/` (figures + captions) and
`docs/final/PAPER_ANALYSIS_CANDIDATES.md` (the earlier, more granular
analysis-candidate table this document builds on). This document is the
paper-writing entry point: what can be claimed, with what evidence, in
LaTeX-ready table form, plus what still can't be claimed.

## 1. Claim → Evidence Mapping

| # | Claim | Figure/Table | Key numbers | Statistical basis | Limitations / counterarguments |
|---|---|---|---|---|---|
| 1 | RecentK-2 selection improves accuracy using *less* history, not more — recency dominates over the full 10-step window. | Fig. 2, Table 1 | MAE 12.799 (A, K=10) → 10.847 (B, K=2), −15.25% | n=1,698, same checkpoint, controlled comparison (only selector changes) | Single checkpoint/dataset; K-sweep between 2 and 10 not exhaustively tested at full scale (only K=2 and K=10 are full-scale — see Fig. of `results/presentation_20260816/module1_token_selection.png`, K=4/6/8 are 50-sample) |
| 2 | An attention-based selector (first-decoder-layer salience) loses to plain recency at every K tested — a negative result for a more "principled" alternative. | `results/presentation_20260816/module1_token_selection.png` | AttentionTopK worse than RecentK by +0.04° to +1.17° across K∈{8,6,4,2} | 50-sample only; gap widens monotonically as K shrinks | **50-sample, not full-scale** — report with explicit sample-size caveat |
| 3 | Block-verified speculative decoding cuts LLM forward evaluations ~4.8x while keeping MAE within tolerance. | Fig. 3(a), Table 1 | 20 → 4.21 avg. forwards; MAE A→C +0.26% | n=1,698; accuracy-preserved criterion MAE ≤ 1.05× baseline | Single (threshold, γ) operating point highlighted; full sweep in claim 4 |
| 4 | Accuracy is insensitive to the acceptance threshold across a 7x range — no cliff found in [0.35, 2.5]. | Fig. 3(b) | MAE 12.831 → 12.929 across threshold 0.35→2.5 | n=1,698, 4 threshold points | Only 4 points sampled; a cliff beyond 2.5 or between sampled points cannot be ruled out |
| 5 | The selector's accuracy shift and speculative decoding's latency shift compose additively, not interactively. | Fig. 4 | Paired diff: `recent_k2_vs_baseline` mean −1.903°, `combined_vs_recent_k2` mean +0.048° (≈ speculative's own standalone cost of +0.033°) | n=1,698, paired per-sample decomposition, two independent methods (CDF overlap + numeric decomposition) agree | N/A — this is one of the strongest, most independently-corroborated claims in the project |
| 6 | Configuration D (RecentK-2 + speculative) is the only configuration achieving both accuracy improvement AND latency reduction simultaneously. | Fig. 2, Table 1 | MAE −14.87%, latency 4.68x speedup | n=1,698, same checkpoint/GPU | Config B (selector only) *increases* latency (0.92x) — selection alone does not reduce compute in this architecture |
| 7 | Per-sample degradation under D (47.1% of samples individually worse) is 100% attributable to the selector, not speculative decoding, in the worst-case tail. | Fig. 5 | 84/84 top-5%-degraded samples: `|diff(selector)| ≥ |diff(speculative)|`, no exceptions | n=1,698 (84-sample tail, exhaustive within that tail, not a sample) | Attribution method (magnitude comparison of paired diffs) is a heuristic, not a causal intervention per sample |
| 8 | Motion speed's relationship to accuracy change is a fan-shaped variance effect, not a simple monotonic one: population-wide correlation is negative, but the worst-case tail concentrates in a high-motion, high-variance regime. | Fig. 5 | Population Spearman ρ=−0.400 (p=2.8e-66); top-5%-worst mean motion speed 2.16x the rest (Mann-Whitney p=3.0e-24) | n=1,698 for correlation; n=84 vs. n=1,614 for group comparison | Two statistics answer different questions (correlation vs. group comparison) — must be presented together, not either alone, to avoid an apparently contradictory claim |
| 9 | An adaptive-K intervention targeting the fan-shaped-variance tail works on its target population but fails at the population level because its velocity-based trigger has poor precision, and this is not fixable by re-tuning the threshold. | Fig. 6, Fig. 7 | Target group MAE −12.8% (n=84) vs. overall MAE +8.53% (n=1,698); only 63/445 (14.2%) widened samples were true positives; 5 history-derived features all fail to separate true/false positives (AUC 0.488–0.555, all p>0.15) | n=1,698 overall; n=63/382 true/false-positive groups; 5 independent Mann-Whitney tests | This instance's own experiment (2026-08-23), not cross-validated on another checkpoint; a learned classifier or different feature family was not tried (explicitly out of scope) |
| 10 | Every headline property (selector gain, speculative tradeoff incl. draft-model accept rate, additive composition) transfers to an unseen dataset (Wu2017) with no meaningful degradation. | Fig. 8 | A→B: −15.25% (in-dist.) vs. −14.39% (unseen); accept rate 71.25%→72.0% (C), 77.75%→77.2% (D) | n=1,698 in-dist., n=300 unseen (evenly-strided subset of 1,395) | **300/1,395 samples, not full-scale** — a real generalization result, not a full-population one |

## 2. Table 1 — Main Ablation (LaTeX, booktabs)

```latex
\begin{table}[t]
  \centering
  \caption{Main ablation, full 1{,}698-sample Jin2022 test split, same fine-tuned checkpoint. Speedup is relative to config A's latency, measured within this session on one GPU.}
  \label{tab:main-ablation}
  \begin{tabular}{lrrrrr}
    \toprule
    Config & MAE (\si{\degree}) & RMSE (\si{\degree}) & Latency (ms) & Fwd/pred. & Speedup \\
    \midrule
    A: Baseline                     & 12.799 & 27.119 & 571.7 & 20.00 & 1.00$\times$ \\
    B: RecentK-2                    & 10.847 & 22.487 & 623.0 & 20.00 & 0.92$\times$ \\
    C: Speculative ($\theta{=}0.35,\gamma{=}8$) & 12.831 & 27.142 & 124.4 & 4.21 & 4.59$\times$ \\
    D: RecentK-2 + Speculative      & \textbf{10.895} & 22.547 & \textbf{122.2} & 4.01 & \textbf{4.68$\times$} \\
    \bottomrule
  \end{tabular}
\end{table}
```

## 3. Table 2 — Adaptive-K and Diagnosis Summary (LaTeX, booktabs)

```latex
\begin{table}[t]
  \centering
  \caption{Adaptive-K full-scale result (n=1{,}698) and the false-positive separation diagnosis (5 history-derived features, n=63 true positive / n=382 false positive). AUC is the probability a random false-positive sample's feature value exceeds a random true-positive sample's; 0.5 = no separation.}
  \label{tab:adaptive-k}
  \begin{tabular}{lrr}
    \toprule
    \multicolumn{3}{l}{\textit{(a) Overall vs. target-group MAE, before (D) vs. after (Adaptive-K)}} \\
    \midrule
    Population & MAE before & MAE after ($\Delta$\%) \\
    \midrule
    Overall (n=1{,}698)              & 10.895 & 11.825 (+8.53\%) \\
    Top-5\% degraded group (n=84)    & 23.915 & 20.864 ($-$12.8\%) \\
    \midrule
    \multicolumn{3}{l}{\textit{(b) Separation diagnosis: true positive (n=63) vs. false positive (n=382)}} \\
    \midrule
    Feature & AUC & $p$ (Mann--Whitney) \\
    \midrule
    Direction reversals   & 0.555 & 0.16 \\
    Velocity std.\         & 0.546 & 0.25 \\
    Velocity CV            & 0.544 & 0.26 \\
    Avg.\ acceleration     & 0.532 & 0.42 \\
    Avg.\ velocity (control) & 0.488 & 0.77 \\
    \bottomrule
  \end{tabular}
\end{table}
```

## 4. Reproducibility Block (drop into an Experimental Setup section)

> **Model.** Llama2-7B (base weights, fp16) with a LoRA adapter
> ($r{=}32$, target modules `q_proj`, `v_proj`, 128 adapter tensors) and
> a non-PLM embedding/task-head stack (linear embedding layer,
> LayerNorm, 1D convolution, 3-DOF regression head), following the
> checkpoint-era `EmbeddingForViewportPrediction` architecture. The
> checkpoint was strict-load verified against the assembled model
> (0 missing/unexpected/mismatched keys, adapter and non-PLM modules
> checked independently) before every session's results.
>
> **Data.** Jin2022 viewport-prediction dataset, 10-step history →
> 20-step future rollout (`his_window=10, fut_window=20, trim_head=30,
> trim_tail=60, frequency=5, step=15`), deterministic test split of
> 1,698 samples. Generalization results use Wu2017 under the identical
> windowing configuration (1,395-sample test split, unseen during this
> checkpoint's fine-tuning), evaluated on 300 evenly-strided samples
> (stride 4, deterministic, no RNG).
>
> **Hyperparameters.** RecentK selector: $K{=}2$ (history length kept).
> Adaptive-K selector: $K \in \{2,4,10\}$, motion-speed thresholds
> $v_{low}{=}2.41$, $v_{high}{=}4.44$ deg/step (25th/75th percentile of
> the historically-degraded group's own motion-speed distribution).
> Speculative block verification: draft $\gamma{=}8$ (drafted
> coordinates per iteration), acceptance threshold $0.35$ (L2 distance
> in the task head's Tanh-bounded normalized output space, not
> degrees), draft model is a parameter-free constant-velocity
> extrapolator (`RecentVelocityDraft`).
>
> **Hardware.** Single NVIDIA RTX 4090 (24GB), fp16 inference.
> **Latency footnote**: absolute latency was measured across four
> distinct physical instances over the course of this project and
> varies by up to $\sim$18\% between them for identical configurations
> (e.g. config A baseline: 571.7ms, 673.2ms, 462.7ms, 459.5ms across
> instances) — a property of physical hardware/host variance, not of
> the method. Every latency comparison in this paper is made
> *within* a single instance/session; absolute latency numbers should
> not be compared across the different runs cited in this paper without
> that caveat.
>
> **Evaluation protocol.** MAE and a rotation-aware corrected RMSE (both
> in degrees) over the full 20-step rollout, per-sample and aggregate;
> latency measured end-to-end per prediction (median of $n$ samples);
> speculative configurations additionally report average target-model
> forward count and average accept rate (accepted drafted coordinates
> per draft-verify iteration, excluding the initial cache-seeding
> forward). A threshold$=0$ equivalence gate (speculative output
> torch.equal-level identical to baseline, atol $1\times10^{-5}$ fp32 /
> $2\times10^{-3}$ fp16) is required to pass before any threshold$>0$
> result from a given implementation is trusted.

## 5. Gap List (priority-ordered, with rough GPU-time estimate)

| Priority | Gap | Why it matters | Est. cost |
|---|---|---|---|
| 1 | Full-scale Wu2017 (currently 300/1,395 evenly-strided) | Generalization claim (#10) would move from "spot-check" to "full-scale", closing the last sample-size caveat in the strongest results set | ~25–30 GPU-min (2 harness runs at full 1,395, same pattern as this session's 300-sample runs) |
| 2 | Threshold sweep beyond 2.5, and finer-grained between existing points | Claim #4 ("no cliff in [0.35, 2.5]") cannot be extended past 2.5 or to gaps between the 4 sampled points without more data | ~10 GPU-min per additional threshold point at full 1,698 scale (speculative pass only, baseline reused) |
| 3 | Iteration-position accept pattern (early/mid/late step within the 20-step rollout) | Flagged as blocked since 2026-08-09 — `accepted_per_iteration` is computed in memory but never persisted; would sharpen the acceptance-ceiling finding (#5 in `PAPER_ANALYSIS_CANDIDATES.md`) into a positional one | Small harness change (persist a per-sample JSON-lines sidecar) + 1 full-1,698 run, ~15 GPU-min total |
| 4 | AdaLoRA-checkpoint integration accuracy | Flagged as unresolved since 2026-08-09 (`TEAM_REPORT_20260809.md`) — no accuracy number exists yet for a non-LoRA adapter variant; the threshold=0 equivalence gate must pass first | Unknown — depends on 하영/teammate's AdaLoRA checkpoint availability, not purely a GPU-time question |
| 5 | Better adaptive-K trigger signal (learned classifier or richer feature, e.g. acceleration variance, future-trajectory shape) | Fig. 7's negative result rules out 5 simple derived statistics but does not rule out a learned or richer-feature approach — the natural next experiment this session's diagnosis motivates | Exploratory — design + implement + validate, not a fixed GPU-time estimate; likely 1+ full session |
| 6 | Multi-checkpoint / multi-seed variance estimate | All results in this paper come from one fine-tuned checkpoint; no estimate exists of how much these numbers would vary with a different training run of the same recipe | Requires re-running the NetLLM fine-tuning procedure itself (training-time cost, not just inference) — out of scope for this project's inference-side focus unless explicitly requested |
| 7 | External baseline comparison (other published VP/LLM-based methods) | Every claim in this paper is a controlled internal comparison (baseline vs. selector vs. speculative vs. combined on the same checkpoint) — no comparison exists against other papers' reported numbers on Jin2022/Wu2017 | Requires either reproducing another method's pipeline or citing its published numbers under matching protocol — significant scoping work, not a quick GPU run |
