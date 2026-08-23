# 발표 대본 — NetLLM VP × LiteVLM 경량화 (10분)

`results/presentation_20260816/presentation_storyline.md`의 슬라이드
순서·시간 배분을 그대로 따른 실제 발화 원고. 한국어 구어체, 전체
약 1,570자(10분 분량). 톤: 과장 없이 수치 중심. 두 서사 축 —
① "속도를 위해 정확도를 희생하지 않았다"(열화 100% selector 귀속),
② "실패한 실험에서 무엇을 배웠는가"(adaptive-K) — 를 각각 조합
슬라이드와 심층분석 슬라이드에서 명시적으로 짚는다.

---

## 슬라이드 1 — 문제 제기 (1.0분)

**원고**:
"지금 저희 VP 파이프라인은 예측 한 번에 LLM forward를 20번 돌립니다.
latency 570밀리초, 실시간 서비스엔 너무 느립니다. 오늘은 두 경량화
기법 — 이력을 줄이는 Token Selection과, forward 횟수를 줄이는
Speculative Decoding — 을 붙였을 때 속도뿐 아니라 **정확도까지
지켰는지**를 보여드리겠습니다."

- 그림: 없음
- 강조 수치: forward 20회, latency 571.7ms

---

## 슬라이드 2 — 모듈① Token Selection (2.0분)

**원고**:
"이력 10스텝 중 일부만 남기는 selector 두 개를 비교했습니다. attention
점수로 고르는 AttentionTopK, 최근 것만 남기는 RecentK입니다. 결과는
반전이었어요 — 더 정교해 보이는 attention 기반이 모든 K에서 졌습니다.
RecentK는 K를 2까지 줄여도, 즉 이력을 10개에서 2개로 줄여도 오히려
정확도가 올라갑니다. 이 곡선은 50-sample 기준이고, K=2 지점만 전체
1,698개에서 재확인했는데 결과는 똑같았습니다."

- 그림: `module1_token_selection.png`
- 강조 수치: RecentK K=2 MAE 10.847(full-1698), AttentionTopK가 모든
  K에서 열세

---

## 슬라이드 3 — 모듈② Speculative Decoding (2.5분)

**원고**:
"speculative decoding은 draft 모델이 여러 스텝을 미리 예측하고, target
모델이 한 번의 forward로 한꺼번에 검증합니다. forward 20번이 4.2번으로
줄었습니다. threshold를 0.35에서 2.5까지 7배 넓혀도 MAE는 12.83에서
12.93 사이에서만 움직여요 — 절벽이 없습니다. threshold=0에서 baseline과
완전히 똑같다는 동등성 게이트를 통과한 구현이라 믿을 수 있는
결과입니다."

- 그림: `module2_speculative.png`
- 강조 수치: forward 20→4.2, threshold 7배 확대해도 MAE 변화 0.1° 이내

---

## 슬라이드 4 — 조합 (1.5분)

**원고**:
"두 모듈을 합치면? 이 CDF 그림에서 baseline과 speculative-only 곡선이
거의 겹치고, RecentK-2와 둘 다 합친 곡선도 거의 겹칩니다. 정확도는
selector가, latency는 speculative가 각자 담당하고 서로 거의 안
건드린다는 뜻이에요. 그래서 RecentK-2와 speculative를 같이 쓰면
정확도 −14.9%, latency 4.7배 감소 — 이 프로젝트에서 유일하게 둘 다
잡은 조합입니다."

- 그림: `module3_combination.png`, `summary_table.png`
- 강조 수치: MAE −14.87%, latency 4.68배 감소

---

## 슬라이드 5 — 심층 분석: tail과 adaptive-K (2.5분)

**원고**:
"방금 본 조합, 평균은 좋아졌지만 개별 샘플의 47%는 오히려 악화됩니다.
살펴보니 — 전체로는 모션이 빠를수록 대체로 좋아지는데(상관계수
−0.40), 최악 5%는 그 빠른 모션 구간에 몰려 있어요. 빠른 모션이
결과를 더 좋게도 나쁘게도 만드는 '분산'의 문제였고, 이 오차는 100%
selector 탓이었습니다.

그래서 모션이 빠른 구간에서만 이력을 넓히는 adaptive-K를
시도했습니다. 목표로 삼은 최악 5% 그룹은 실제로 −12.8% 개선됐어요.
그런데 전체 1,698개로는 오히려 8.5% 악화됐습니다. 넓힌 445개 중 진짜
목표 그룹은 63개뿐, 나머지 382개는 평균 속도만 보고 잘못 넓혀 평균
4.8도씩 나빠졌습니다. 5개 지표로 더 파봤지만 이 둘을 가르는 신호가
10스텝 이력 안엔 없었습니다. 실패했지만 왜인지까지 짚은 실험입니다."

- 그림: `../../results/speculative/consolidated/tail_velocity_vs_diff.png`,
  `../../results/presentation_20260816/adaptive_k_negative_result.png`
- 강조 수치: 목표군 −12.8%, 전체 +8.5%, false positive 382/445(86%)

---

## 슬라이드 6 — 결론 및 향후 (0.5분)

**원고**:
"정리하면, RecentK-2와 speculative를 같이 쓰는 게 정확도·속도를 동시에
잡는 유일한 조합이었고, 그 정확도 개선은 100% selector 덕분이었습니다.
adaptive-K는 방향은 맞았지만 평균 속도만으로는 트리거를 못 만든다는
걸 확인했고, 다음 단계는 다른 정보원을 찾는 겁니다. 이상입니다."

- 그림: `summary_table.png`
- 강조 수치: −14.87% / 4.68배 (한 번 더 반복)

---

## 예상 질문 5개

| # | 예상 질문 | 답변 요지 | 근거 문서 |
|---|---|---|---|
| 1 | AttentionTopK가 진 이유가 뭔가? attention이 틀렸다는 뜻인가? | attention은 첫 decoder layer 기준이라 장기 의존성을 아직 반영 못했을 가능성. RecentK는 이 태스크의 관성(inertia) 구조와 자연히 맞는다. | `docs/final/FINAL_RESULTS_SUMMARY.md` "recency dominates narrative" 절 |
| 2 | threshold를 더 낮추면 forward 수가 더 줄어드는가? | accept rate가 이미 근접 천장(평균 6.22/8)이라 낮은 threshold에서 얻을 여지가 크지 않다. | `docs/experiment_phase/analysis/TAIL_ANALYSIS.md` acceptance mechanism 절 |
| 3 | 가산적 결합이라는 걸 통계적으로 어떻게 확인했나? | per-sample paired diff를 분해해서 `diff(B,A)`와 `diff(D,B)`를 따로 계산 — CDF의 두 쌍 겹침과 수치가 서로 검증. | `results/speculative/consolidated/paired_stats_combined_vs_baseline.json` |
| 4 | 실패한 실험(adaptive-K)을 왜 발표에 넣었나? | 모집단 음의 상관(rho=-0.40) 예측을 개입 실험으로 검증했고 그대로 재현됐다. 실패 원인도 "임계값이 나빴다"가 아니라 "이 정보원으로는 분리 자체가 불가능하다"는 수준까지 진단했다. | `docs/experiment_phase/analysis/ADAPTIVE_K_RESULTS.md` "후속 진단" 절 |
| 5 | 실서비스에 바로 적용 가능한 상태인가? | AdaLoRA 통합 미검증, latency는 인스턴스마다 새로 측정해야 함 등 남은 리스크가 있다. | `docs/final/TEAM_REPORT_20260809.md` (e) 다음 확인 사항 |
