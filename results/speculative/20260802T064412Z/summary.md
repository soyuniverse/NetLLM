# Speculative Block Verification Benchmark

dry_run=True, checkpoint_loaded=False, metric_available=False, num_samples=5

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline | None | None | None | None | 573.287 | 20.00 | None | False | True |
| threshold=0.0_gamma=2 | 0.0 | 2 | None | None | 614.860 | 20.00 | 0.0 | False | False |
| threshold=0.0_gamma=4 | 0.0 | 4 | None | None | 617.148 | 20.00 | 0.0 | False | False |
| threshold=0.0_gamma=8 | 0.0 | 8 | None | None | 624.853 | 20.00 | 0.0 | False | False |
| threshold=0.5_gamma=2 | 0.5 | 2 | None | None | 329.858 | 11.00 | 1.8 | True | False |
| threshold=0.5_gamma=4 | 0.5 | 4 | None | None | 213.681 | 7.00 | 3.0 | True | False |
| threshold=0.5_gamma=8 | 0.5 | 8 | None | None | 155.000 | 5.00 | 4.25 | True | False |
| threshold=3.0_gamma=2 | 3.0 | 2 | None | None | 331.922 | 11.00 | 1.9 | True | False |
| threshold=3.0_gamma=4 | 3.0 | 4 | None | None | 180.591 | 6.00 | 3.6 | True | False |
| threshold=3.0_gamma=8 | 3.0 | 8 | None | None | 121.742 | 4.00 | 6.0 | True | False |
