# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=300

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline | None | None | 15.895621841192918 | 31.978097424218316 | 459.509 | 20.00 | None | False | True |
| threshold=0.35_gamma=8 | 0.35 | 8 | 15.927836322117027 | 32.02589304001501 | 100.138 | 4.20 | 5.675338189386056 | True | True |
