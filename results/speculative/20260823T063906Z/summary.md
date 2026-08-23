# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=50

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline | None | None | 11.036768085417648 | 22.132159381010695 | 467.170 | 20.00 | None | False | True |
| threshold=0.35_gamma=8 | 0.35 | 8 | 11.031369208392997 | 22.109701547003056 | 96.303 | 4.20 | 5.73125 | True | True |
