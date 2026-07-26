# Llama Identity Equivalence Result

Gate 3 passed on the real Jin2022 test sample `(index=0, video=4,
user=83, timestep=30)`.

| Comparison | Max absolute difference | Exact |
| --- | ---: | --- |
| Original vs disabled | 0.0 | yes |
| Original vs Identity | 0.0 | yes |
| Disabled vs Identity | 0.0 | yes |

All three FP16 predictions have identical SHA-256
`681a39836b75a051338d8c8dce73dad8433e1dcd3fd184bc510a87874d2f5537`.
The output is finite `[1,20,3]`; all paths use sequence lengths 10–29 and 20
PLM forwards without cache reuse. Identity selected indices 0–9, was called
once, and was never applied to feedback.

Tolerance was `atol=1e-6`, `rtol=0`.
