# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=1698

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline_selector=adaptive_k:2.41:4.44 | None | None | 11.781744255411246 | 24.78262742604094 | 461.129 | 20.00 | None | False | True |
| threshold=0.35_gamma=8_selector=adaptive_k:2.41:4.44 | 0.35 | 8 | 11.82494632773541 | 24.816175649715106 | 98.353 | 4.10 | 5.985773899848255 | True | True |
