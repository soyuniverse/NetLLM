# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=50

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline_selector=adaptive_k:1.93:5.22 | None | None | 10.5922342558677 | 22.089463437497074 | 486.012 | 20.00 | None | False | True |
| threshold=0.35_gamma=8_selector=adaptive_k:1.93:5.22 | 0.35 | 8 | 10.595116833147904 | 22.085122796546287 | 108.129 | 4.06 | 6.104575163398692 | True | True |
