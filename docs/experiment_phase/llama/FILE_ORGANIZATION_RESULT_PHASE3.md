# File Organization Result — Recovery Phase

## Organization result

All final artifacts are in the requested canonical phase directories. Existing
docs, runtime, logs, manifests, assets, and user work were preserved.

- Existing files overwritten/deleted: none
- Model/checkpoint/dataset/ZIP moved or modified: none
- Duplicate basenames: only intentional per-attempt runtime names
- Broken Python syntax/import: none
- Hard-coded `/workspace` or legacy `/venv/` in new artifacts: none
- New source-tree `__pycache__`/`.pyc`: none
- Temporary validation files/figures: none
- Git-staged ZIP/model/checkpoint/data: none

One newly generated, unreferenced diagnostic directory was moved from
`experiments/vp/llama_vp_technical_smoke_diagnostic_20260726` to
`experiments/vp/llama_vp_technical_smoke/diagnostics/run1_trace_hook`.
All three file SHA-256 values matched before and after:

- `run_status.txt`:
  `0eb4a6485f4abb760d398eb0a377364c2de05f6a22714d81a041469ca0d69380`
- `technical_smoke.log`:
  `ebef80d9343714cff33dd0d3c63966b3c0be3a67076fd4105a8050e40d1aa5b9`
- `technical_smoke_result.json`:
  `6c9b2bc92ff35513f3b17d747b326f3f8e67ed318af556d7e90541eef38a0579`

## Final integrity

| Item | Result |
| --- | --- |
| Current upstream | `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`, clean, empty diff |
| Checkpoint-era source | `ee4d8726898610e4ae7df08bdd26728cafb4701f`, clean, empty diff |
| Current/era source caches | 0 / 0 |
| GPT-2 environment freeze | `731a5031a3fb94909d541db8b66a41299d0977716e09d26eaa682ec3154d0311` |
| GPT-2 artifact fingerprint | `f3fdcf85dd2a8d38b329048ebb0349bcc94e1c6a04aa08ec20a4c0334ed74f14` |
| Llama environment freeze | `8a7e30ccc90703c0810291255f84704a7d5b2e95635ce04d94e80845b5de00f1` |
| Llama base manifest | `28afd48051cd8293a5744eba52d7b21955b273ceadf9209a64a6af227597d3a3` |
| Checkpoint manifest | `44cbaaa6a174207bd98c21030200ad4244f09b5273a3ec0355ece0830519c1c6` |
| Dataset manifest | `4cbc567ebc3783102c996b46fffe965b617815a6a9e487e0a33a59aa4fa17399` |
| Checkpoint ZIP | `57062c71a3e103ae610ccbc499feee22dc46d25e32b8179cac20d6d2e32dec53` |
| Dataset ZIP | `9c3b700524b63082ab8e85fba72a24d34c81c5b9f782f5a93efd204716476e8d` |
| External checksum validation | base 12, checkpoint 4, dataset 48,957 files: pass |
| Llama `pip check` | pass |
| ZIP local ignore/staging | both ignored / none staged |

The checkpoint adapter and non-PLM SHA-256 values were also equal before and
after every load/smoke attempt.
