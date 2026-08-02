# Speculative Block Verification Benchmark

dry_run=False, checkpoint_loaded=True, metric_available=True, num_samples=50

| config | threshold | gamma | MAE | corrected RMSE | latency median (ms) | avg target forwards | avg accept/iter | speedup_claim_valid | accuracy_preserved |
|---|---|---|---|---|---|---|---|---|---|
| baseline | None | None | 11.036768085417648 | 22.132159381010695 | 566.927 | 20.00 | None | False | True |
| threshold=0.0_gamma=2 | 0.0 | 2 | 11.037794513401886 | 22.13189667904841 | 598.202 | 20.00 | 0.0 | False | True |
| threshold=0.0_gamma=4 | 0.0 | 4 | 11.0370079585885 | 22.130876151063365 | 599.581 | 20.00 | 0.0 | False | True |
| threshold=0.0_gamma=8 | 0.0 | 8 | 11.038046102779607 | 22.132311800466717 | 605.302 | 20.00 | 0.0 | False | True |
| threshold=0.5_gamma=2 | 0.5 | 2 | 11.058236487167576 | 22.18392999663387 | 322.904 | 11.00 | 1.886 | True | True |
| threshold=0.5_gamma=4 | 0.5 | 4 | 11.035251266814768 | 22.12323105458299 | 177.093 | 6.14 | 3.6342412451361867 | True | True |
| threshold=0.5_gamma=8 | 0.5 | 8 | 11.042410857615371 | 22.129414651763103 | 119.462 | 4.14 | 5.885350318471337 | True | True |
| threshold=1.0_gamma=2 | 1.0 | 2 | 11.046280847964187 | 22.16352505292789 | 321.887 | 11.00 | 1.898 | True | True |
| threshold=1.0_gamma=4 | 1.0 | 4 | 11.031559901254873 | 22.153502208449122 | 177.086 | 6.02 | 3.756972111553785 | True | True |
| threshold=1.0_gamma=8 | 1.0 | 8 | 11.057485279815893 | 22.171438161914658 | 119.172 | 4.02 | 6.185430463576159 | True | True |
