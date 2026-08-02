# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=1698

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline_selector=recent_k:2 | None | None | 10.84686681689948 | 22.486722208730615 | 622.952 | 20.00 | None | False | True |
| threshold=0.35_gamma=8_selector=recent_k:2 | 0.35 | 8 | 10.895102344584037 | 22.54730367655577 | 122.228 | 4.01 | 6.219784524975514 | True | True |
| threshold=0.7_gamma=8_selector=recent_k:2 | 0.7 | 8 | 10.902755948342183 | 22.58030846315392 | 121.899 | 4.00 | 6.303606428851431 | True | True |
