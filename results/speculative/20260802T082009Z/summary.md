# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=1698

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline | None | None | 12.798559396025476 | 27.118723422261244 | 570.467 | 20.00 | None | False | True |
| threshold=0.35_gamma=8 | 0.35 | 8 | 12.831301641086654 | 27.141736128537733 | 124.419 | 4.21 | 5.699743213499633 | True | True |
| threshold=0.7_gamma=8 | 0.7 | 8 | 12.849403701948885 | 27.154655461187122 | 124.005 | 4.08 | 6.028074866310161 | True | True |
| threshold=1.5_gamma=8 | 1.5 | 8 | 12.892513489803276 | 27.236853099700895 | 123.461 | 4.03 | 6.201708406134731 | True | True |
| threshold=2.5_gamma=8 | 2.5 | 8 | 12.929292182552446 | 27.327806831298513 | 123.620 | 4.02 | 6.269673891818004 | True | True |
