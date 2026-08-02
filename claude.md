# CLAUDE.md — NetLLM × LiteVLM 프로젝트 컨텍스트

## 프로젝트 목표
NetLLM(SIGCOMM'24)의 Viewport Prediction(VP) 파이프라인에 LiteVLM(NVIDIA)의
경량화 기법 2가지를 이식한다:
1. Speculative Decoding (Eagle-2 스타일 draft-and-verify, 연속 좌표용으로 변형)
2. Token Selection Module (FastV 스타일, LLM 1번째 decoder layer attention 기반)

## 절대 규칙
- 기존 NetLLM 원본 소스는 수정 금지. 모든 신규 기능은 wrapper/신규 파일로 추가.
- 기존 완성 코드 경로:
  - src/netllm_litevlm/vp/llama_old_selectable_pipeline.py (Selector 파이프라인)
  - src/netllm_litevlm/selectors/ (IdentitySelector, RecentKSelector)
  - scripts/experiment_phase/llama/benchmark/ (벤치마크 스크립트)
- 모델: Llama2-7B checkpoint (strict load 검증 완료, missing/unexpected key 0/0)
- VP 태스크: 과거 10개 시점 → 미래 20 step의 (roll, pitch, yaw) autoregressive 예측
  - Prediction shape: [1, 20, 3], 회귀 문제 (MAE/RMSE/Loss, 낮을수록 좋음)
- 검증 데이터: test 1,698 samples. 빠른 반복은 50-sample smoke test 먼저.

## 현재 상태 (8.2 기준)
- Continuous VP Draft-and-Verify prototype 존재: draft(velocity extrapolation)와
  검증 제어 흐름은 동작하나, target forward가 여전히 step당 1회 = 총 20회.
  → 계산량 감소 없음 (speedup_claim_valid=False). 원인: target_predictor가
  검증 전에 항상 전체 20-step baseline을 완주하고 사후에 draft prefix를
  이어붙이는 구조라 target forward를 줄일 여지 자체가 없음
  (docs/experiment_phase/speculative/PHASE_A_DESIGN.md §2).
- Block verification 설계/구현 진행 중: src/netllm_litevlm/speculative/block_verify.py
  (설계 근거: docs/experiment_phase/speculative/PHASE_A_DESIGN.md).
- **인스턴스 재생성으로 자산 유실 확인 (8.2)**:
  - transformers/peft 미설치 상태였음 → transformers==4.34.1, peft==0.6.2,
    accelerate==0.24.1 재설치 완료 (torch는 기존 2.2.0 유지, pip check 통과).
  - transformers 4.34.1은 legacy tuple 기반 past_key_values 사용
    (`transformers.cache_utils` 모듈 자체가 없음). 실측 결과 incremental
    cache 디코딩과 full recompute가 fp16/GPU에서도 torch.equal 수준으로
    완전히 동일 — block verification의 threshold=0 정합성 게이트가
    설계상 성립함을 확인.
  - **VP fine-tuned checkpoint(LoRA adapter + modules_except_plm.bin)와
    Jin2022 데이터셋이 파일시스템 전체에서 발견되지 않음** (NetLLM-assets/checkpoints,
    NetLLM-assets/datasets 모두 비어있음; staging 포함 전수 검색 완료).
    Base Llama2-7b 가중치는 존재 (단, /root/NetLLM 루트에 잘못 위치 —
    /root/NetLLM/*.safetensors, config.json 등; /root/NetLLM-assets/llama/base로
    이동 예정이었으나 아직 미이동).
  - 사용자 결정: 체크포인트/데이터셋 복구 전까지 축소 범위로 진행
    — base Llama2-7b 가중치 + 랜덤 초기화 SimpleLinearTaskHead + 합성 입력으로
    block verification의 **정합성 게이트만** 검증 (threshold=0 동등성,
    target forward count). 실제 VP 예측 정확도/실데이터 accept율은 미검증 상태로
    남아있음 — 체크포인트/데이터셋 복구 후 재검증 필요.
- 다음 핵심 작업: 여러 draft step을 한 번의 target forward로 검증하는
  **block verification** 구현 + target forward 수 실측 감소 확인.

## 정합성 게이트 (모든 speculative 구현이 통과해야 함)
1. acceptance_threshold=0 → 모든 draft 거부 → 출력이 baseline과 torch.equal 수준
   동일 + target forward 정확히 20회.
2. threshold>0 → target forward < 20 실측 확인 (계측 counter 필수).
3. OOM 없음, non-finite 출력 없음.

## 하드웨어
Vast.ai 단일 GPU, fp16 inference. Llama2-7B 로드 기준 GPU 메모리 ~13GB 사용 중.