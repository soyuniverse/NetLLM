# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=200

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline | None | None | 11.730207886356089 | 25.665130586496794 | 673.168 | 20.00 | None | False | True |
| threshold=0.35_gamma=8 | 0.35 | 8 | 11.742942034595185 | 25.690684853477926 | 144.988 | 4.16 | 5.803486529318542 | True | True |
