# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=200

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline_selector=recent_k:2 | None | None | 10.443956559733337 | 22.13911082247506 | 665.940 | 20.00 | None | False | True |
| threshold=0.35_gamma=8_selector=recent_k:2 | 0.35 | 8 | 10.465869078778512 | 22.21214369341645 | 142.062 | 4.00 | 6.232945091514143 | True | True |
