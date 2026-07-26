# Llama VP Non-Multimodal Technical Smoke Result

## Final result

**PASS** — the second preserved run completed the full technical contract.

| Item | Result |
| --- | --- |
| Dataset | upstream Git-tracked Jin2022 test split |
| Sample | index 0, video 4, user 83, timestep 30 |
| History / future | `[1,10,3]` / `[1,20,3]` |
| Prediction | `[1,20,3]`, finite |
| Dtype/device | FP16 / `cuda:0` |
| LoRA | active, adapter `default` |
| Adapter source/loaded keys | 128 / 128 |
| Adapter value mismatches after FP16 cast | 0 |
| Non-PLM missing/unexpected | 0 / 0 |
| Full sequence forwards | 20 |
| Sequence lengths | 10 through 29 |
| Cache reuse | false |
| Inference latency | 1.082 s |
| Peak allocated/reserved | 13,060.0 / 13,090.0 MiB |
| Random VP component | none |

PEFT's internal adapter load result reports 293 missing *base-model* keys
because it loads an adapter state dict into the composite model. This is not an
adapter omission: a separate PEFT adapter-only state extraction verified all
128 source keys, no extra keys, and exact tensor equality after the expected
FP32-to-FP16 cast.

## Validity

```text
technical_smoke_valid=True
quality_metric_valid=False
reproduction_claim_valid=False
```

No quality metric was computed. The remaining limitations are checkpoint
selection provenance and the immutable Llama base revision used during
training, not a load or forward failure.

## Preserved attempts

The initial runtime reached a finite `[1,20,3]` prediction but failed its
post-validation because a trace hook was attached below the PEFT callable and
observed zero forwards. A diagnostic run confirmed the instrumentation error.
Neither runtime was overwritten. The corrected hook was attached to the actual
PEFT wrapper and the successful run is:

`experiments/vp/llama_vp_technical_smoke/run2_20260726/technical_smoke_result.json`.

No training, backward, optimizer, quantization, or offload was used.
