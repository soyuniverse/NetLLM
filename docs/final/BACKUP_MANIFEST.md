# Backup Manifest — 2026-08-02

Off-instance backup, created because this instance has lost its
checkpoint/dataset assets twice already this project. Location:
`/root/backup_20260802/` (outside the git repository — never committed;
`.gitignore` now also excludes `*.tar.gz`/`*.tar`/`*.zip` repo-wide as a
safety net, in addition to this directory being outside `/root/NetLLM/`
entirely).

## Contents

| file | source | size | sha256 |
|---|---|---:|---|
| `results_20260802.tar.gz` | `tar czf` of `/root/NetLLM/results/` (includes every per-sample CSV, every run's `results.csv`/`summary.json`/`summary.md`, all figures, the final table, tail-analysis stats) | 1,420,895 B (1.4 MiB) | `729dd63b32fa27a44de8be119b5251e44943c6652011bc235e37fea9931fbd25` |
| `docs_manifests_20260802.tar.gz` | `tar czf` of `/root/NetLLM/docs/` + `/root/NetLLM/manifests/` | 2,783,946 B (2.7 MiB) | `9bfe22db50334c04bac55d2dea38c7a4c7752c9eb8b1c16378014e7dec7f4eb9` |
| `try_llama2_7b.zip` | copy of `/root/NetLLM-assets/staging/try_llama2_7b.zip` (VP checkpoint archive, not re-compressed) | 77,861,701 B (75 MiB) | `57062c71a3e103ae610ccbc499feee22dc46d25e32b8179cac20d6d2e32dec53` |
| `data.zip` | copy of `/root/NetLLM-assets/staging/data.zip` (Jin2022 + Wu2017 dataset archive, not re-compressed) | 3,199,081,523 B (3.0 GiB) | `9c3b700524b63082ab8e85fba72a24d34c81c5b9f782f5a93efd204716476e8d` |

Total: ~3.1 GiB. Checksums also collected in
`/root/backup_20260802/SHA256SUMS.txt` (verify with `sha256sum -c
SHA256SUMS.txt` from inside that directory). The two zip copies were
verified byte-identical to the staging originals (checksums computed on
both, matched exactly) before this manifest was written.

`results/` and `docs/`+`manifests/` are already in the git repository
(pushed) — this backup is a second copy for redundancy given the prior
asset-loss history, not the only copy. The two zips are **not** in git
(by design, per `docs/final/FILE_ORGANIZATION_RULES.md`: models/
checkpoints/datasets never live in the repository) — this backup and
`/root/NetLLM-assets/staging/` are the only two copies of those on this
instance.

## Download

Run from your **local machine**, not from inside this instance. Fill in
your Vast.ai SSH connection details (visible on the instance's Vast.ai
dashboard "Connect" panel) in place of the placeholders below — this
instance's container hostname/internal IP are not usable as an external
address, so an exact pre-filled command can't be produced from inside it:

```bash
scp -P <VAST_SSH_PORT> -r root@<VAST_SSH_HOST>:/root/backup_20260802/ ./netllm_backup_20260802/
```

Or, to fetch just the small git-redundant pieces (skip the 3+ GiB
dataset/checkpoint zips, e.g. if bandwidth is limited and you only want
the results/docs redundancy beyond what's already pushed to GitHub):

```bash
scp -P <VAST_SSH_PORT> \
  root@<VAST_SSH_HOST>:/root/backup_20260802/results_20260802.tar.gz \
  root@<VAST_SSH_HOST>:/root/backup_20260802/docs_manifests_20260802.tar.gz \
  root@<VAST_SSH_HOST>:/root/backup_20260802/SHA256SUMS.txt \
  ./
```

After downloading, verify integrity locally:

```bash
sha256sum -c SHA256SUMS.txt
```
