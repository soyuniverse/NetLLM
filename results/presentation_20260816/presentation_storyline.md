# 발표 구성안 — NetLLM VP × LiteVLM 경량화 (10분 대면 발표)

전체 원칙: 이번 발표는 **새 실험이 아니라 기존에 검증된 결과의 모듈 단위
재구성**이다. 모든 수치는 2026-08-02 인스턴스의 full-1,698-sample 결과를
기본으로 하고, AttentionTopK 비교만 50-sample 기준(그 방식으로만 측정된
적이 있어서)임을 매 슬라이드에서 명시한다. 각 그림 하단 각주에 원본 run
디렉토리가 있으니, 발표 중 수치를 추궁받으면 그 경로를 그대로 보여주면 된다.

## 슬라이드 순서 / 시간 배분 (총 10분)

| # | 슬라이드 | 시간 | 핵심 그림/표 |
|---|---|---:|---|
| 1 | 문제 제기 | 1.0분 | (텍스트만) |
| 2 | 모듈① Token Selection | 2.0분 | `module1_token_selection.png` |
| 3 | 모듈② Speculative Decoding | 2.5분 | `module2_speculative.png` |
| 4 | 조합 (모듈①+②) | 2.0분 | `module3_combination.png` + `summary_table.png` |
| 5 | 심층 분석 (tail, 상위 2개) | 2.0분 | `../speculative/consolidated/tail_velocity_vs_diff.png` + `accept_rate_histogram.png` |
| 6 | 결론 및 향후 | 0.5분 | `summary_table.png` (재사용) |

합계 10.0분. 리허설 시 슬라이드 4~5가 넘치기 쉬우니, 시간이 부족하면
슬라이드 5의 두 번째 항목(acceptance ceiling)을 한 문장으로 축약하고
슬라이드 6으로 붙인다.

---

### 슬라이드 1 — 문제 제기 (1.0분)

- Llama2-7B 기반 VP(Viewport Prediction) 파이프라인은 baseline이 예측
  1회당 LLM forward 20회, latency 중앙값 571.7ms — 실시간 서비스에 부적합.
- LiteVLM의 경량화 기법 2가지(Token Selection, Speculative Decoding)를
  이식해 정확도를 지키면서 latency를 줄일 수 있는지가 질문.
- 텍스트만, 그림 없음.

**예상 질문**: "왜 하필 이 두 기법을 골랐나?"
**답변 근거**: `claude.md` 프로젝트 목표 절 (LiteVLM 2대 경량화 기법 정의).

---

### 슬라이드 2 — 모듈① Token Selection (2.0분)

그림: `module1_token_selection.png`

- 발언 요지: "이력 선택에서는 최근성이 attention 중요도를 이긴다." RecentK가
  모든 K에서 AttentionTopK보다 낮은 MAE. K=2에서 RecentK는 baseline 대비
  MAE가 오히려 개선(더 짧은 이력이 더 정확) — negative result:
  attention 기반 선택이 이론적으로는 더 정교해 보이지만 실측에서 진다.
- 표본 크기를 반드시 언급: 곡선 자체는 50-sample 기준(AttentionTopK가
  그 규모로만 측정됐으므로 공정 비교를 위해 RecentK도 동일 50-sample로
  맞춤), K=2 지점만 별표로 full 1,698-sample 확인값(10.847)을 병기.
- Negative result를 감추지 말고 명시: "더 똑똑해 보이는 방법이 진 것도
  하나의 결과다."

**예상 질문**: "AttentionTopK가 진 이유가 뭔가? attention이 틀렸다는 뜻인가?"
**답변 근거**: `docs/final/FINAL_RESULTS_SUMMARY.md` "recency dominates
narrative" 절 (3중 증거) — attention은 첫 decoder layer 기준이라
장기 의존성을 아직 반영 못했을 가능성, RecentK는 이 태스크의 관성 구조와
자연히 맞는다는 설명.

---

### 슬라이드 3 — 모듈② Speculative Decoding (2.5분)

그림: `module2_speculative.png` (2-패널)

- 좌 패널: 예측당 LLM forward 수 20 → 4.2, 약 4.8배 감소.
- 우 패널: threshold를 0.35~2.5까지 7배 넓혀도 MAE는 12.83~12.93 사이
  (y축을 12.7~13.0으로 확대해 둔감성을 시각적으로 강조) — **동등성
  게이트가 이미 통과된 상태에서** threshold를 크게 흔들어도 정확도가
  거의 변하지 않는다는 뜻.
- 동등성 게이트 언급 필수: threshold=0 → baseline과 완전 동일 (atol=1e-5
  fp32 / 2e-3 fp16), target forward 정확히 20회 — 이 게이트를 통과한
  구현이라야 threshold>0 결과를 신뢰할 수 있다.

**예상 질문**: "threshold를 더 낮추면 (예: 0.1) forward 수가 더 줄어드는가?"
**답변 근거**: `results/speculative/consolidated/threshold_vs_forward_count.png`
+ `docs/experiment_phase/analysis/TAIL_ANALYSIS.md` acceptance mechanism
절 — accept rate가 이미 근접 천장(mean 6.22/8)이라 낮은 threshold에서도
forward 수 감소 여지가 크지 않음.

---

### 슬라이드 4 — 조합 (2.0분)

그림: `module3_combination.png` + `summary_table.png`

- CDF 그림: baseline ≈ +Speculative (거의 겹침), +RecentK-2 ≈ +둘 다
  (거의 겹침) — 두 쌍이 각각 겹친다는 것 자체가 **가산적 결합**의
  시각적 증거: 정확도 이동은 selector가 담당, latency 감소는 speculative가
  담당, 서로 거의 간섭하지 않는다.
- 표: D(RecentK-2+Speculative)가 유일하게 정확도 개선(-14.87%)과 latency
  감소(4.68배)를 동시에 달성. B(selector만)는 latency가 오히려 소폭
  증가(0.92배) — selector 자체는 계산량을 줄이지 않는다는 점을 짚는다.

**예상 질문**: "가산적이라는 걸 어떻게 통계적으로 확인했나, 그림만으로는
정성적 아닌가?"
**답변 근거**: `results/speculative/consolidated/paired_stats_combined_vs_baseline.json`
+ `docs/experiment_phase/speculative/PHASE_B_REAL_RESULTS.md` §2 (per-sample
paired diff 분해, diff(B,A)와 diff(D,B)를 분리 계산).

---

### 슬라이드 5 — 심층 분석: tail 발견 중 상위 2개 (2.0분)

**선별 근거** (전체 tail 발견 6~7개 중 이 2개만 고른 이유): 10분 발표에서
tail 전체를 다루면 산만해진다. 아래 두 개는 (a) 가장 반직관적이라
청중의 흥미를 끌고, (b) 다음 슬라이드의 "향후 계획(adaptive-K)"으로 바로
연결되는 실행 가능한 결론을 제공한다는 공통점으로 골랐다. 나머지
(acceptance 세부 iteration 패턴, Wu2017 일반화 spot-check 등)는 논문
후보 문서(`docs/final/PAPER_ANALYSIS_CANDIDATES.md`)에는 남기되 발표에서는
생략한다.

1. **팬 형태 분산 + 100% selector 귀속** — 그림
   `../speculative/consolidated/tail_velocity_vs_diff.png`. 전체
   1,698-sample에서는 history motion speed와 정확도 변화의 상관관계가
   **음수**(rho=-0.40, p=2.8e-66) — 즉 모션이 빠를수록 대체로 개선된다.
   하지만 최악 5%(85개 샘플)는 평균보다 2.16배 빠른 모션 구간에
   집중(p=3.0e-24) — "빠른 모션 = 나쁨"이 아니라 "빠른 모션 = 결과 분산
   증가, 그 분산의 꼬리가 최악 사례"라는 재해석. 그리고 이 85개 전부
   (100%) 오차의 지배적 원인이 RecentK-2 selector이지 speculative
   decoding이 아니다.
2. **Acceptance rate 근접 천장** — 그림
   `../speculative/consolidated/accept_rate_histogram.png`. 전체
   1,698-sample의 accept rate 평균 6.22/8(=77.75%), 중앙값 6.33 —
   ~99%가 히스토그램 한 구간에 몰려 있어 threshold/gamma를 더 튜닝해도
   얻을 여지가 거의 없다. 이것이 왜 "다음 단계는 selector 쪽(adaptive-K)이지
   speculative 쪽이 아니다"라는 결론의 실증적 근거다.

**예상 질문**: "그러면 adaptive-K가 이 문제를 실제로 풀었나?"
**답변 근거**: `docs/experiment_phase/analysis/ADAPTIVE_K_RESULTS.md`
(이번 세션 Task 3 결과 — 성공/실패 여부와 관계없이 수치 그대로 기재됨;
Task 3가 게이트 실패 등으로 미완료였다면 이 문서 자체가 없을 수 있으니
발표 전 파일 존재를 확인할 것).

---

### 슬라이드 6 — 결론 및 향후 (0.5분)

- 한 문장 결론: "RecentK-2 + Speculative Decoding(D)이 유일하게 정확도
  개선(-14.87%)과 latency 감소(4.68배)를 동시에 달성하는 설정이다."
- 향후 방향 한 줄: 고분산 tail 구간에 한해 이력 길이를 넓히는
  adaptive-K selector가 다음으로 시도할 만한, 데이터가 뒷받침하는 유일한
  좁은 범위의 다음 단계다 (adaptive-threshold가 아니라 adaptive-K인 이유는
  슬라이드 5의 두 근거 그대로).
- Task 3(adaptive-K)이 이번 세션에서 성공했다면 결과 한 문장을 이 슬라이드
  마지막에 추가한다 (아래 "Task 3 성공 시 추가 문장" 참고 — 이 문서
  자체를 그 문장으로 갱신할 것).

**예상 질문**: "실서비스에 바로 적용 가능한 상태인가?"
**답변 근거**: `docs/final/TEAM_REPORT_20260809.md` (e) 다음 확인 사항 —
AdaLoRA 통합 미검증, latency는 인스턴스별로 새로 측정 필요 등 남은 리스크
목록.

---

## Task 3(adaptive-K) 성공 시 추가할 문장 (자리 표시자)

adaptive-K가 성공적으로 검증되면 슬라이드 6 마지막에 다음 형태로 한 문장을
추가한다 (본 세션에서 실제 결과가 나오면 이 자리를 그 수치로 교체할 것):

> "실제로 adaptive-K selector를 적용하면, 기존 열화 상위 5% 샘플군에서
> [before MAE] → [after MAE]로 개선되면서도 전체 MAE는 유지된다
> (`ADAPTIVE_K_RESULTS.md`)."

실패했거나 미착수라면 이 절 자체를 삭제하고 슬라이드 6은 위 원안 그대로
유지한다.
