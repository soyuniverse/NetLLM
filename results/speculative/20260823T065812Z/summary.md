# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=50

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline_selector=adaptive_k:2.41:4.44 | None | None | 10.57441734717538 | 22.315690645266603 | 456.320 | 20.00 | None | False | True |
| threshold=0.35_gamma=8_selector=adaptive_k:2.41:4.44 | 0.35 | 8 | 10.583967821838955 | 22.31505355162846 | 96.952 | 4.08 | 6.084415584415584 | True | True |
