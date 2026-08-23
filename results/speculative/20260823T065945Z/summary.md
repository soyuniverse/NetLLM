# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=50

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline_selector=recent_k:2 | None | None | 10.017189621763924 | 20.1992407737527 | 514.971 | 20.00 | None | False | True |
| threshold=0.35_gamma=8_selector=recent_k:2 | 0.35 | 8 | 10.018038059212268 | 20.223159471450806 | 98.913 | 4.02 | 6.245033112582782 | True | True |
