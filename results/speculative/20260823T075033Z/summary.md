# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=300

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline_selector=recent_k:2 | None | None | 13.606910356814714 | 27.33808227739539 | 458.538 | 20.00 | None | False | True |
| threshold=0.35_gamma=8_selector=recent_k:2 | 0.35 | 8 | 13.64648620796602 | 27.399367898510814 | 98.227 | 4.01 | 6.171650055370986 | True | True |
