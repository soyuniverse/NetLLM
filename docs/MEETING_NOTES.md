
---

# `docs/MEETING_NOTES.md`

```markdown
# NetLLM × LiteVLM 구현 회의록

## 1. 회의 목적

NetLLM에 LiteVLM의 경량화 모듈을 연결하기 위한 대상 태스크, 구현 범위, 우선순위 및 업무 분담을 결정하였다.

논의 대상 모듈은 다음 세 가지다.

- Patch Selection
- Token Selection
- Speculative Decoding

이번 구현의 목적은 새로운 Transformer를 처음부터 설계하는 것이 아니라, 기존 NetLLM 구조에 LiteVLM의 경량화 아이디어를 연결하고 정상 동작을 확인하는 것이다.

---

## 2. 연구 전체 범위

NetLLM의 주요 태스크는 다음과 같다.r

- VP: Viewport Prediction
- ABR: Adaptive Bitrate Streaming
- CJS: Cluster Job Scheduling

연구 전체 범위에서는 세 태스크를 모두 고려한다.

다만 세 태스크의 데이터 표현과 실행환경이 서로 다르기 때문에, 모든 태스크의 전체 학습과 성능 재현을 먼저 완료한 뒤 모듈 구현을 시작하는 방식은 비효율적이라고 판단하였다.

따라서 다음과 같이 범위를 분리한다.

### 전체 프로젝트 범위

- VP, ABR, CJS의 환경 및 구조 파악
- 태스크별 input encoder와 output head 위치 정리
- 공통 selector interface 설계
- 향후 태스크별 확장 가능성 검토

### 최초 구현 범위

- VP baseline 실행
- VP data flow 및 tensor shape 분석
- VP에 LiteVLM 모듈의 최소 구현 연결
- 적용 전후 성능 및 효율 기록

---

## 3. VP를 최초 대상으로 선정한 이유

VP는 이미지 정보와 시계열 viewport 정보를 함께 사용할 수 있다.

따라서 다음 세 모듈을 검토하기에 가장 적합하다.

- 시각 정보에 대한 Patch 또는 Frame Selection
- 시계열 및 multimodal 입력에 대한 Token Selection
- 미래 좌표 예측에 대한 speculative prediction 가능성

ABR과 CJS는 이미지 patch가 없기 때문에 LiteVLM의 Patch Selection을 그대로 적용하기 어렵다.

VP에서 공통 selection interface를 검증한 후 ABR의 state token과 CJS의 graph token으로 확장하는 방향이 합리적이라고 판단하였다.

---

## 4. 기본 구현 원칙

다음 원칙을 기준으로 구현한다.

1. 기존 NetLLM 원본 코드는 수정하지 않는다.
2. 새로운 코드는 외부 wrapper 또는 adapter 방식으로 작성한다.
3. 원본 baseline이 정상 실행된 후에만 모듈을 연결한다.
4. 성능 향상은 필수 조건이 아니다.
5. 모듈이 정상적으로 연결되고 forward 또는 학습이 동작하는 것을 1차 목표로 한다.
6. 논문에 직접 제공되지 않은 구현은 공개 라이브러리 또는 유사 구현을 조사하여 활용할 수 있다.
7. 외부 코드 사용 시 출처, commit, license, 수정 내용을 기록한다.
8. 새로운 attention 또는 FFN 변형은 기본 모듈 연결 후 후속 과제로 둔다.
9. 모듈의 기본값은 OFF로 설정하고, OFF 상태에서 원본 동작을 유지한다.
10. 각자 다른 서버를 사용할 수 있으므로 library와 environment를 문서화한다.

---

## 5. Patch Selection 논의

LiteVLM의 Patch Selection은 이미지 patch 또는 camera view 중 중요한 입력만 선택하는 방식이다.

다만 NetLLM VP가 실제로 raw image patch token을 사용하는지는 코드와 데이터 구조를 확인해야 한다.

가능한 경우는 다음과 같다.

### Case A: 여러 frame feature가 존재

여러 시점의 image feature 중 중요한 frame을 선택한다.

이 경우 구현 명칭은 `FrameFeatureSelector` 또는 `TemporalFrameSelector`로 한다.

### Case B: 여러 patch token이 존재

Patch-level importance score를 계산하고 일부 patch만 유지한다.

이 경우 실제 Patch Selection을 구현할 수 있다.

### Case C: sample당 image vector가 하나만 존재

선택 대상이 없으므로 Patch Selection은 직접 적용할 수 없다.

이 경우 raw image와 ViT patch embedding을 포함하는 별도 pipeline이 필요하며, 현재 단계에서는 설계 문서만 작성한다.

회의에서는 Patch Selection이 세 모듈 중 데이터 구조 분석과 pipeline 수정이 가장 많이 필요할 가능성이 높다고 판단하였다.

따라서 실제 patch 단위 구현은 후순위로 두고, 가능한 경우 frame 또는 image-feature selection부터 구현한다.

---

## 6. Token Selection 논의

Token Selection은 VP에 가장 직접적으로 연결할 수 있는 모듈로 판단하였다.

VP의 과거 viewport 시계열이 temporal token으로 변환된다면, 다음과 같은 방식으로 입력 길이를 줄일 수 있다.

- 최근 시점의 token만 유지
- 중요도 점수가 높은 token만 유지
- 시간적으로 너무 먼 token 제거
- image token을 보존하면서 temporal token만 선택

최초 구현 순서는 다음과 같다.

1. IdentitySelector
2. RecentKSelector
3. ScoreTopKSelector

ScoreTopKSelector의 초기 scorer는 position-wise MLP 또는 작은 Linear-MLP로 구현한다.

Restricted Self-Attention과 복잡한 구조는 초기 구현 범위에서 제외한다.

회의 결과 Token Selection을 가장 먼저 구현하는 것이 적절하다고 판단하였다.

---

## 7. Speculative Decoding 논의

LiteVLM의 Speculative Decoding은 작은 draft model이 token 후보를 생성하고 큰 model이 이를 검증하는 방식이다.

NetLLM VP는 일반적인 language token이 아니라 Roll, Pitch, Yaw와 같은 연속 좌표를 출력할 가능성이 있다.

따라서 기존 speculative decoding을 그대로 적용할 수 있는지는 불확실하다.

우선 다음 내용을 분석한다.

- VP가 미래 시점을 어떤 방식으로 순차 생성하는지
- networking head가 연속값을 출력하는지
- 예측값이 다음 입력으로 어떻게 사용되는지
- draft output을 main model이 검증할 수 있는 기준이 있는지
- KV cache 기반 병렬 검증이 가능한지

직접 호환되지 않는 경우 다음과 같이 별도 구조를 검토한다.

- Lightweight trajectory draft head
- Multi-step coordinate proposal
- Main NetLLM verification
- Error threshold 기반 accept/reject

이 구조는 LiteVLM의 speculative decoding과 동일한 구현이라고 주장하지 않는다.

회의 결과 다음 주에는 직접 구현보다 호환성 분석과 최소 설계를 우선한다.

---

## 8. 추가 구조 아이디어

다음 아이디어가 논의되었다.

- 첫 번째 layer representation을 이용한 score 계산
- Restricted Self-Attention
- Position-wise FFN 기반 importance scoring
- 시간적으로 먼 token 제거
- 시계열 데이터의 positional 특성 반영
- Patch 또는 token selection score의 태스크별 변경

다만 이러한 변형은 현재 우선순위가 아니다.

이번 단계에서는 기존에 공개된 모듈 또는 단순한 구조를 사용해 정상 동작을 확인한다.

---

## 9. 업무 분담 방향

### Patch 또는 Frame Selection 담당

- VP image 및 image-feature 데이터 구조 분석
- raw image, precomputed feature, patch token 존재 여부 확인
- image encoder와 multimodal projection 위치 확인
- 가능한 경우 FrameFeatureSelector 구현
- 선택 전후 feature 개수와 shape 기록
- 실제 Patch Selection 확장 계획 작성

### Token Selection 담당

- viewport embedding과 LLM input sequence 분석
- temporal token 수와 attention mask 확인
- IdentitySelector 구현
- RecentKSelector 구현
- ScoreTopKSelector 구현
- token 수, latency, memory, MAE 비교

### Speculative Decoding 담당

- LiteVLM speculative decoding 구조 분석
- VP networking head와의 호환성 검토
- continuous-coordinate prediction과 language token generation 차이 정리
- draft-verifier prototype 설계
- 적용이 부적절하면 기술적 근거 문서화

### 통합 지원

- 공통 selector interface 검토
- shape mismatch 해결
- config와 CLI 통합
- branch merge 지원
- 환경 및 library version 통합

---

## 10. 서버 및 환경 관리

각자 별도의 서버를 사용할 수 있으므로 다음 내용을 반드시 기록한다.

- GPU model
- VRAM
- CUDA driver
- CUDA toolkit 또는 PyTorch CUDA runtime
- Python version
- PyTorch version
- Transformers version
- PEFT version
- 추가 설치 package
- NetLLM commit
- Project commit
- Dataset path
- 실행 명령
- 사용한 외부 코드 출처
- 발생한 error와 해결 과정

재현을 위해 다음 파일을 유지한다.

- `setup.sh`
- `requirements-vp.txt`
- `environment.yml`
- `scripts/check_environment.sh`
- 실험 config
- 실행 log
- result JSON 또는 CSV

---

## 11. 다음 단계

현재 완료된 상태는 다음과 같다.

- Git 연결
- `setup.sh` 기반 기본 환경 세팅
- NetLLM 원본 저장소 확보

아직 검증되지 않은 내용은 다음과 같다.

- VP baseline 정상 실행
- VP multimodal 입력 실행
- Jin2022 및 Wu2017 데이터 존재 여부
- image 및 image-feature 데이터 존재 여부
- 실제 tensor flow
- 실제 patch token 존재 여부
- LiteVLM 모듈 삽입 지점

따라서 다음 순서로 진행한다.

1. Repository audit
2. Upstream integrity check
3. Environment 및 dependency 확인
4. VP repository map 작성
5. Dataset/checkpoint 확인
6. VP baseline smoke test
7. Tensor shape 추적
8. IdentitySelector 구현
9. Token Selection 구현
10. Frame/Image-Feature Selection 구현
11. Speculative Decoding 호환성 분석
12. 통합 실험 및 문서화

---

## 12. 최종 회의 결정사항

1. 연구 전체 범위는 VP, ABR, CJS로 유지한다.
2. 최초 실제 구현 대상은 VP로 한다.
3. 모든 태스크를 완전히 재현한 뒤 시작하지 않는다.
4. 전체 태스크의 환경과 interface는 문서화하되 실제 학습과 모듈 검증은 VP부터 수행한다.
5. 원본 NetLLM 코드는 수정하지 않는다.
6. Token Selection을 최우선으로 구현한다.
7. Patch Selection은 실제 데이터 구조에 따라 Frame/Image-Feature Selection으로 변환할 수 있다.
8. 실제 patch token이 없으면 가짜 Patch Selection을 구현하지 않는다.
9. Speculative Decoding은 VP와의 직접 호환성을 먼저 검증한다.
10. 성능 향상보다 정상 동작과 재현 가능성을 우선한다.
11. 사용 라이브러리와 외부 코드 출처를 반드시 문서화한다.
12. VP에서 검증된 공통 selector interface를 향후 ABR과 CJS로 확장한다.

---

## 13. 최종 방향

NetLLM 전체 태스크를 고려한 공통 개발 기반을 준비하되, 최초 경량화 모듈 구현과 검증은 VP에 집중한다.

VP에서 Token Selection과 Frame/Image-Feature Selection을 먼저 구현하고, Speculative Decoding은 호환성 분석 후 구현 여부를 결정한다.

모든 신규 기능은 외부 wrapper 또는 adapter 형태로 작성하고 원본 NetLLM은 read-only upstream으로 유지한다.