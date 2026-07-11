# VP GPT-2 runtime tensor contract

## 1. 검증 범위

이 문서는 NetLLM commit `105bcf070f2bec808f7b14f8f5a953de6e4e6e54`의 VP GPT-2 non-multimodal autoregressive inference를 실제 Jin2022 test sample과 batch size 1에서 측정한 runtime contract다.

```text
B=1
H=10
F=20
E=1024
coordinate channels=3
device=cuda:0
dtype=float32
multimodal=False
```

full numeric statistics는 `experiments/vp/phase2b_runtime/phase2b_tensor_trace.json`을 기준으로 한다.

## 2. dataset contract

```text
create_dataset(
    dataset="Jin2022",
    his_window=10,
    fut_window=20,
    trim_head=30,
    trim_tail=60,
    frequency=5,
    step=15,
    include=["test"],
)
```

default collate 결과:

| 값 | shape | dtype | device |
|---|---|---|---|
| history | `[1,10,3]` | `torch.float32` | CPU |
| future | `[1,20,3]` | `torch.float32` | CPU |
| video/user/timestep | 각각 `[1]` | `torch.int64` | CPU |

측정 sample은 `(video=4, user=83, timestep=30)`이다.

## 3. normalization contract

channel 순서는 Roll, Pitch, Yaw다.

```text
Roll  / 180
Pitch / 90
Yaw   / 180
```

| tensor | shape | min | max | mean | finite |
|---|---|---:|---:|---:|---|
| raw history | `[1,10,3]` | `-0.770342` | `144.435684` | `46.234268` | True |
| normalized history | `[1,10,3]` | `-0.008559` | `0.802420` | `0.259934` | True |

## 4. initial viewport embedding contract

한 history timestep에 다음 연산이 적용된다.

```text
x[:,i,:] [1,3]
→ Conv1d(1,256,kernel_size=3)
→ LeakyReLU
→ Flatten
→ view(1,256)
→ Linear(256,1024)
→ unsqueeze(1)
```

| 단계 | runtime shape | 비고 |
|---|---|---|
| timestep input | `[1,3]` | Conv1d의 unbatched input으로 해석 |
| raw Conv1d output | `[256,1]` | batch dimension 없음 |
| sequential output | `[256,1]` | `Flatten(start_dim=1)`이 shape를 유지 |
| explicit view | `[1,256]` | source의 batch-size 1 고정점 |
| projected timestep | `[1,1,1024]` | temporal token-like embedding 1개 |
| concatenated history | `[1,10,1024]` | 10개 timestep 연결 |
| LayerNorm output | `[1,10,1024]` | GPT-2 첫 input |

initial sequence의 모든 tensor는 `float32`, `cuda:0`, finite, `requires_grad=False`였다.

## 5. GPT-2 autoregressive contract

각 step `s∈[0,19]`에서:

```text
input length L = H + s = 10 + s
inputs_embeds: [1,L,1024]
attention_mask: [1,L], int64, all ones
final hidden states: [1,L,1024]
NetworkingHead full input: [1,L,1024]
selected last hidden: [1,1,1024]
Linear output: [1,1,3]
Tanh output: [1,1,3]
```

대표 step:

| tensor | step 0 | step 1 | step 19 |
|---|---|---|---|
| GPT-2 inputs_embeds | `[1,10,1024]` | `[1,11,1024]` | `[1,29,1024]` |
| attention mask | `[1,10]` | `[1,11]` | `[1,29]` |
| final hidden | `[1,10,1024]` | `[1,11,1024]` | `[1,29,1024]` |
| selected hidden | `[1,1,1024]` | `[1,1,1024]` | `[1,1,1024]` |
| coordinate output | `[1,1,3]` | `[1,1,3]` | `[1,1,3]` |
| cache key/value layer 0 | `[1,16,10,64]` | `[1,16,11,64]` | `[1,16,29,64]` |

모든 대표 tensor는 finite, `requires_grad=False`였다.

## 6. feedback embedding contract

각 coordinate output은 다음과 같이 sequence 끝에 추가된다.

```text
coordinate [1,1,3]
→ Conv1d(1,256,3) [1,256,1]
→ LeakyReLU + Flatten [1,256]
→ Linear(256,1024) [1,1024]
→ unsqueeze(1) [1,1,1024]
→ cat with current sequence
```

history embedding과 달리 feedback input은 batched 3D Conv1d input이다. feedback embedding에는 `embed_ln`이 다시 적용되지 않는다.

GPT-2에서 실제 사용된 sequence length:

```text
10, 11, 12, ..., 29
```

append 이후 length:

```text
11, 12, 13, ..., 30
```

마지막 length 30 sequence는 생성되지만 GPT-2에 입력되지 않는다.

## 7. cache contract

GPT-2는 각 step에서 다음 cache를 반환한다.

```text
layers=24
key/value=[1,16,L,64]
```

그러나 `Pipeline.auto_regressive()`의 다음 호출에는 `past_key_values` argument가 전달되지 않는다.

```text
cache returned: 20/20 steps
cache passed to next step: 0/20 steps
cache reused: False
```

따라서 현재 contract는 incremental decoding이 아니라 매 step 전체 sequence 재계산이다.

## 8. output contract

20개 `[1,1,3]` output을 dim 1로 concatenate한다.

```text
prediction: [1,20,3]
dtype: torch.float32
device: cuda:0
finite: True
requires_grad: False
range: [-0.9917044639587402, 0.9857870936393738]
```

NetworkingHead의 `Tanh` 때문에 normalized output은 `[-1,1]` 안에 있다. 현재 head와 viewport embedding은 학습되지 않았으므로 이 값은 accuracy contract가 아니다.

## 9. gradient 및 mode contract

```text
plm.training=False
pipeline.training=False
torch grad enabled=False
execution context=torch.inference_mode()
prediction.requires_grad=False
```

backward, optimizer, scheduler, adaptation 및 LoRA는 존재하지 않는다.

## 10. 확정된 제약

- current initial viewport embedding은 `.view(1,256)` 때문에 `B=1` contract다.
- initial history와 feedback의 Conv1d rank가 다르다.
- history sequence만 LayerNorm을 통과한다.
- cache는 반환되지만 재사용되지 않는다.
- final feedback embedding은 계산되지만 소비되지 않는다.
- multimodal tensor는 이번 contract에 포함되지 않는다.
