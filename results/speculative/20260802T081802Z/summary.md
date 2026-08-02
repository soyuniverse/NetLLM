# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=50

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline | None | None | 11.036768085417648 | 22.132159381010695 | 569.369 | 20.00 | None | False | True |
| threshold=1.0_gamma=8 | 1.0 | 8 | 11.057485279815893 | 22.171438161914658 | 121.469 | 4.02 | 6.185430463576159 | True | True |
| threshold=1.5_gamma=8 | 1.5 | 8 | 11.063315919337173 | 22.182884303845075 | 121.385 | 4.02 | 6.23841059602649 | True | True |
| threshold=2.5_gamma=8 | 2.5 | 8 | 11.08302636426439 | 22.09694531296065 | 121.607 | 4.02 | 6.2781456953642385 | True | True |
