# NetLLM × LiteVLM 연구 방향

## 1. 연구 배경

NetLLM은 네트워크 데이터를 Large Language Model에 적응시켜 다양한 네트워크 태스크를 처리하는 구조를 제안한다. 주요 태스크는 다음과 같다.

- VP: Viewport Prediction
- ABR: Adaptive Bitrate Streaming
- CJS: Cluster Job Scheduling

각 태스크는 입력 데이터 표현과 출력 목표가 다르다.

- VP는 과거 viewport 시계열과 선택적인 이미지 정보를 이용해 미래의 Roll, Pitch, Yaw를 예측한다.
- ABR은 네트워크 처리량, 버퍼 상태, 비디오 청크 정보 등을 이용해 다음 비트레이트를 결정한다.
- CJS는 작업 간 의존성과 컴퓨팅 자원을 고려해 실행할 작업 단계와 자원 배분을 결정한다.

본 연구는 NetLLM 구조에 LiteVLM의 경량화 아이디어를 연결하여, 네트워크 태스크에서 입력 토큰 수와 연산량을 줄일 수 있는지 검토하는 것을 목표로 한다.

---

## 2. 장기 연구 범위

장기적으로는 VP, ABR, CJS를 모두 고려한다.

각 태스크에 대응하는 입력 구조는 다음과 같다.

| Task | Input structure | Possible selection target |
|---|---|---|
| VP | Viewport time-series + image/image feature | Temporal token, frame feature, visual token |
| ABR | Throughput sequence + buffer/scalar state | State token, temporal token |
| CJS | Job dependency graph + resource state | Graph node token, stage token |

최종적으로는 태스크별 입력 구조에 대응할 수 있는 공통 selection interface를 설계하고, 태스크별 scoring 방식만 분리하는 방향을 지향한다.

다만 세 태스크의 구현 환경과 데이터 파이프라인이 서로 다르므로, 모든 태스크를 완전히 재현한 후에 모듈 구현을 시작하지 않는다.

---

## 3. 최초 구현 대상

최초 reference task는 VP로 한다.

VP를 먼저 선택하는 이유는 다음과 같다.

1. 시계열 입력과 이미지 modality를 동시에 다룰 수 있다.
2. Patch/Frame Selection과 Token Selection을 함께 검토할 수 있다.
3. 미래 좌표를 autoregressive하게 예측하므로 decoding 경량화의 적용 가능성도 분석할 수 있다.
4. VP에서 검증된 selector interface를 ABR과 CJS로 확장할 수 있다.

따라서 연구 전체 범위는 VP, ABR, CJS이지만, 최초 코드 구현과 실험은 VP에 집중한다.

---

## 4. 현재 단계의 목표

현재 단계에서는 새로운 Transformer 구조를 설계하거나 기존 논문 성능을 크게 개선하는 것을 우선 목표로 두지 않는다.

우선순위는 다음과 같다.

1. 원본 NetLLM VP 실행 환경 검증
2. 원본 baseline 및 최소 PLM smoke test
3. 데이터 흐름과 tensor shape 분석
4. 비침습적인 extension point 확인
5. LiteVLM 모듈의 최소 동작 구현
6. 실행 환경과 사용 라이브러리 문서화
7. 모듈 적용 전후의 성능 및 효율 비교

성능이 기존보다 반드시 향상될 필요는 없다. 모듈이 구조적으로 연결되고, 학습 또는 추론이 정상적으로 동작하며, 실험 결과를 재현할 수 있으면 1차 목표를 달성한 것으로 본다.

---

## 5. 적용 대상 모듈

### 5.1 Token Selection

Token Selection을 가장 먼저 구현한다.

VP의 시계열 입력은 여러 temporal token으로 표현될 가능성이 있으므로, 과거 시점의 token 중 일부만 유지하는 방식으로 연산량을 줄일 수 있다.

최초 구현 순서는 다음과 같다.

1. IdentitySelector
2. RecentKSelector
3. ScoreTopKSelector

IdentitySelector는 원본과 동일한 결과를 보장하기 위한 기준 구현이다.

RecentKSelector는 가장 최근의 K개 temporal token만 유지한다.

ScoreTopKSelector는 각 token의 중요도 점수를 계산한 후 상위 K개 또는 일정 비율의 token만 유지한다.

초기 scorer는 복잡한 attention 구조보다 작은 position-wise MLP 또는 Linear-MLP를 사용한다.

Restricted Self-Attention, 새로운 FFN 구조, 복잡한 scoring function은 후속 실험으로 둔다.

---

### 5.2 Patch Selection

LiteVLM의 Patch Selection을 NetLLM VP에 그대로 적용할 수 있는지는 실제 데이터 구조를 확인한 후 결정한다.

NetLLM VP가 raw image patch token을 사용하는 것이 아니라, 사전에 추출된 단일 image feature 또는 frame-level feature를 사용하는 경우 실제 patch selection은 바로 적용할 수 없다.

따라서 1차 구현은 다음과 같이 정의한다.

- 여러 frame feature가 존재하면 Frame Feature Selection을 구현한다.
- 여러 visual token이 존재하면 Visual Token Selection을 구현한다.
- sample당 하나의 image vector만 존재하면 selection 자체가 의미가 없음을 문서화한다.
- raw image와 ViT patch pipeline을 새로 구성해야 한다면 후속 작업으로 분리한다.

Frame Selection을 구현한 경우 이를 Patch Selection이라고 부르지 않는다. LiteVLM 원형과 VP용 변형을 문서에서 명확하게 구분한다.

---

### 5.3 Speculative Decoding

LiteVLM의 Speculative Decoding은 일반적으로 작은 draft model이 language token 후보를 생성하고, 큰 model이 후보를 검증하는 구조다.

NetLLM VP는 일반적인 vocabulary token이 아니라 연속적인 viewport coordinate를 출력할 가능성이 높다.

따라서 다음 항목을 먼저 분석한다.

1. VP의 autoregressive prediction 구조
2. networking head의 출력 형태
3. 예측값의 feedback 방식
4. KV cache 사용 가능 여부
5. 후보 좌표의 acceptance 또는 verification 기준

직접 호환되지 않는 경우 LiteVLM의 Speculative Decoding을 그대로 구현하지 않는다.

대신 다음과 같은 별도 개념으로 분석한다.

- Draft trajectory prediction
- Main model verification
- Multi-step coordinate proposal
- Error threshold 기반 acceptance

이 방식은 LiteVLM의 원래 speculative decoding과 동일한 방법이라고 표현하지 않는다.

---

## 6. 원본 코드 보호 원칙

NetLLM 원본 디렉터리는 read-only upstream으로 취급한다.

금지 사항은 다음과 같다.

- NetLLM 원본 source file 수정
- 원본 파일명 변경
- 원본 파일 삭제
- 원본 코드 formatting
- 원본 commit 임의 변경
- 사용자 작업 삭제
- `git reset --hard`
- `git clean -fd`
- checkpoint, dataset, model weight Git commit

신규 코드는 다음 경로에만 생성한다.

```text
src/netllm_litevlm/
experiments/vp/
scripts/
tests/
configs/
docs/
patches/