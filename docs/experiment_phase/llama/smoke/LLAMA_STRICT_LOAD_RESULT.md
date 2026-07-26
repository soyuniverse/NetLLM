# Llama Local-Only Strict Load Result

## Result

**PASS** — the base, LoRA adapter, and all checkpoint-era non-PLM modules
loaded without modifying any asset.

| Item | Result |
| --- | --- |
| Model class | `LlamaTaskHeadModel2` |
| PEFT wrapper | `PeftModelForFeatureExtraction` |
| Tokenizer | `LlamaTokenizer` |
| Base revision | `01c7f73d771dfac7d292323805ebc428287df4f9` |
| Dtype/device | FP16 / entire model on `cuda:0` |
| LoRA active | true, adapter `default` |
| LoRA tensors/parameters | 128 / 16,777,216 |
| Restored non-PLM parameters | 4,224,003 |
| Task/networking head | restored |
| Viewport projection | restored |
| Multimodal projection | restored but unused |
| Missing/unexpected non-PLM keys | 0 / 0 |
| Base/model load latency | 4.534 s |
| Peak GPU allocated | 13,061.1 MiB |
| Peak GPU reserved | 13,086.0 MiB |

The native checkpoint-era model and state-dict layout were used. No key
migration, `strict=False`, quantization, CPU offload, or alternate model was
used. The FP32 non-PLM checkpoint was strictly restored and the complete
inference pipeline was then converted to FP16 to match the base model.

Checkpoint file SHA-256 values were identical before and after load:

- Adapter:
  `8ae9c330971240cd276fbfe66f93281ded4166df72cb87aae12697e56896e63d`
- Non-PLM:
  `49c9473cd186b53eff929a03caec016e95f4d4795d710ffdfd920d08575ffb39`

No tokenizer token was added by the smoke runner. All VP-relevant trainable
components were restored; the next technical smoke performs an additional
adapter tensor equality check before inference.

Runtime:
`experiments/vp/llama_strict_load/strict_load_result.json`.
