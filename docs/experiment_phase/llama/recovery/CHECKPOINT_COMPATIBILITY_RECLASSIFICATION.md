# Checkpoint Compatibility Reclassification

## Decision

Reclassified from `D. structurally-incompatible` against the current main path
to **A. exact checkpoint-era compatible**.

## Full non-PLM comparison

All ten checkpoint keys were compared by name, shape, and dtype against both
implementations.

| Comparison target | Exact | Renamed, same shape | Missing | Unexpected | Shape mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current `Pipeline` | 8 | 2 | 2 destination names | 2 source names | 0 |
| Checkpoint-era old pipeline | 10 | 0 | 0 | 0 | 0 |

The only main-path difference remains:

```text
4.task_head.0.weight -> 4.networking_head.0.weight
4.task_head.0.bias   -> 4.networking_head.0.bias
```

In the checkpoint-era implementation, no mapping is needed. The expected state
dict contains the original `task_head` keys with shapes `[3,4096]` and `[3]`,
and every other name/shape also matches.

## Loader decision

The native checkpoint-era `run_old.py` strict loader is selected. Gate 3's
external migration loader is not created because that gate applies only to
classification B. The checkpoint remains unchanged.

Machine-readable matrix:
`experiments/vp/llama_source_recovery/reclassification_matrix.json`.
