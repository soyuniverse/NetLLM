# Checkpoint와 current upstream compatibility

- upstream: `/root/NetLLM-source`
- commit: `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`
- Gate 2 분류: **D. structurally-incompatible**

## 현재 save/load contract

`run_plm.py`의 LoRA checkpoint save/load는 다음 contract를 사용한다.

```text
model.plm.save_pretrained(checkpoint)
torch.save(model.modules_except_plm.state_dict(), modules_except_plm.bin)

model.plm.load_adapter(checkpoint, adapter_name="default")
model.modules_except_plm.load_state_dict(torch.load(modules_except_plm.bin))
```

두 번째 `load_state_dict()`에는 `strict=False`가 없으므로 strict load가 기본이다.

Current `Pipeline.modules_except_plm` 순서:

```text
0 embed_vp
1 embed_multimodal
2 embed_ln
3 conv1d
4 plm.networking_head
```

## Module compatibility matrix

| Module | Key 판정 | Shape | 결론 |
|---|---|---|---|
| LoRA q/v projection | PEFT adapter format match | match | compatible |
| viewport Conv1d | exact match | match | compatible |
| viewport projection | exact match | match | compatible |
| LayerNorm | exact match | match | compatible |
| networking head | **renamed module mismatch** | match | incompatible with strict loader |
| multimodal projection | exact match | match | usage undetermined |
| image/features | checkpoint에 저장 안 됨 | N/A | external runtime dependency |

Networking head:

```text
checkpoint:
  4.task_head.0.weight
  4.task_head.0.bias

current upstream expected:
  4.networking_head.0.weight
  4.networking_head.0.bias
```

Strict load 결과로 예상되는 차이:

```text
missing:
  4.networking_head.0.weight
  4.networking_head.0.bias

unexpected:
  4.task_head.0.weight
  4.task_head.0.bias
```

Weight/bias shape는 `[3,4096]`, `[3]`으로 동일하지만 module attribute rename은 단순
prefix-only mismatch가 아니다. 원본 checkpoint key를 변경하지 않았고 silent
`strict=False`도 사용하지 않았다.

## LoRA compatibility

Current `models/low_rank.py`와 checkpoint config는 다음 항목이 일치한다.

- target modules: q/v projection
- rank: 32
- alpha: 32
- dropout: 0.05
- bias: none
- task type: FEATURE_EXTRACTION
- Llama hidden/layer contract: 4096 / 32

Adapter의 saved key prefix는 PEFT adapter serialization contract와 일치한다.

## using_multimodal 판정

판정: **undetermined**

확정 사실:

1. Checkpoint에 명시적 `using_multimodal` config가 없다.
2. `[4096,768]` multimodal projection tensor는 존재한다.
3. Current Pipeline은 `using_multimodal=False`에서도 projection을 항상 생성·저장한다.
4. Current training optimizer 구성도 flag와 무관하게 projection을 포함한다.
5. Image/feature tensor 자체는 checkpoint에 저장되지 않는다.

따라서 projection 존재 또는 shape만으로 multimodal/non-multimodal을 증명하거나 강하게
추론할 수 없다.

## Gate 판정

선택: **D. structurally-incompatible**

이유는 networking head의 renamed-module mismatch다. Gate 5는 분류 B의 prefix-only
mapping만 허용하므로 external compatibility loader를 구현하지 않는다. Gate 4 environment와
Gate 6/7 load/smoke도 시작하지 않는다.

Machine-readable matrix:
`experiments/vp/llama_compatibility/compatibility_matrix.json`
