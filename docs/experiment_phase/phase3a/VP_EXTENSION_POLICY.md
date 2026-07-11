# VP extension policy

## 1. 목적

NetLLM upstream을 read-only로 유지하면서 VP initial temporal sequence에 외부 selector를 삽입한다. 기본 selector 상태는 disabled이며 IdentitySelector는 원본 output을 보존해야 한다.

## 2. extension 방식

`SelectablePipeline`은 upstream `Pipeline`을 composition으로 보유하고 다음 module instance를 그대로 재사용한다.

- `plm`
- `conv1d`
- `embed_vp`
- `embed_ln`
- `networking_head`

원본 class, method 또는 module을 patch/monkey-patch하지 않는다.

Phase 3A에서 외부 class에 재구현한 최소 범위는 `Pipeline.auto_regressive()`의 orchestration이다. Selector 호출을 `embed_ln` 직후 삽입하려면 원본 method 내부에 injection hook이 없기 때문이다. Tensor 연산 순서와 feedback loop는 원본 contract를 유지한다.

## 3. selector 삽입 정책

```text
normalized history [1,10,3]
→ upstream conv1d/embed_vp per timestep
→ concatenate [1,10,1024]
→ upstream embed_ln
→ selector once
→ selected embeddings + attention mask
→ GPT-2 autoregressive loop
```

Selector는 initial history sequence에만 한 번 호출한다.

- autoregressive feedback에는 호출하지 않는다.
- feedback embedding을 선택하거나 제거하지 않는다.
- IdentitySelector는 embeddings와 mask object를 그대로 반환한다.
- token 순서와 selected indices는 시간 순서를 유지한다.

## 4. 현재 보존

### Batch size 1

upstream의 `conv1d(...).view(1,256)` contract를 그대로 사용한다. Batch 확장 수정은 equivalence baseline을 변경하므로 Phase 3A에서 다루지 않는다.

### Feedback LayerNorm 미적용

upstream은 initial history sequence에만 `embed_ln`을 적용하고 feedback embedding은 그대로 append한다. 분포 통일 여부와 무관하게 원본 output 보존을 위해 유지한다.

### KV cache 미사용

각 GPT-2 호출이 cache를 반환하더라도 다음 호출에 `past_key_values`를 전달하지 않는다. Cache 최적화는 계산 순서와 numerical behavior를 변경할 수 있어 제외한다.

### 마지막 unused feedback 계산

Step 19 prediction 후에도 coordinate feedback embedding을 계산하고 length 30 sequence를 구성한다. 사용되지 않더라도 원본 call/module execution contract를 보존한다.

### Full-sequence autoregression

GPT-2 input length `10,11,...,29`를 매 step 전체 처리한다. Incremental decoding으로 바꾸지 않는다.

### Non-multimodal

Phase 3A extension은 `using_multimodal=False`만 허용하고 multimodal Pipeline을 받으면 명시적으로 거부한다.

## 5. 현재 구현하지 않음

- 실제 token pruning
- Recent-K
- ScoreTopK 또는 learned scorer
- Patch/Frame/Image-Feature Selection
- selector training
- LoRA 및 adaptation
- cache optimization
- feedback normalization 변경
- batch-size 확장

## 6. 후속 검토

다음 항목은 Identity equivalence가 성공한 뒤 각각 독립 실험으로 검토한다.

1. Batch size 확장
2. Feedback normalization 통일
3. KV cache 최적화
4. 마지막 feedback 계산 제거
5. Multimodal selector
6. Temporal token pruning

각 변경은 Identity baseline과 분리하여 latency, memory, output 차이와 accuracy 영향을 기록해야 한다.

## 7. Phase 3A 현재 상태

IdentitySelector 자체의 shape/dtype/device/mask/order/no-in-place unit contract는 통과했다. 그러나 equivalence 결과 보고 helper의 Torch API compatibility 오류로 Original/disabled/Identity maximum difference가 기록되지 않았다.

따라서 extension policy는 결정됐지만 Identity equivalence baseline은 아직 확정되지 않았다. Reporting-only 수정과 별도 retry runtime은 사용자 승인 후 수행한다.
