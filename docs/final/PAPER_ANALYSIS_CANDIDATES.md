# Paper Analysis Candidates — 2026-08-23

지금까지 축적된 분석 결과를 논문 작성을 염두에 두고 후보 목록으로 정리한다.
과장하지 않는다: 50-sample 기반 결과는 명시적으로 표시하고, "본문에 쓸 수
있는 강한 증거"와 "부록/발표에만 어울리는 보조 증거"를 구분한다. 각 항목의
출처는 이미 `docs/final/FINAL_RESULTS_SUMMARY.md` /
`docs/experiment_phase/analysis/TAIL_ANALYSIS.md` /
`docs/experiment_phase/speculative/PHASE_B_REAL_RESULTS.md`에 근거가 있고,
이 문서는 그것들을 논문 배치 관점에서 재정렬한 것일 뿐 새로운 주장을
추가하지 않는다.

| # | 항목 | (a) 한 문장 요지 | (b) 근거 수치 | (c) 뒷받침 그림 | (d) 논문 배치 제안 | (e) 강도 평가 |
|---|---|---|---|---|---|---|
| 1 | **가산적 결합 분해** | RecentK-2+Speculative 결합의 정확도 변화는 거의 전부 selector 기여이고, speculative decoding은 그 위에 거의 간섭 없이 더해진다. | n=1,698 paired diff: `combined_vs_recent_k2` mean +0.048°(speculative 단독 비용 +0.033°와 거의 일치) vs. `recent_k2_vs_baseline` mean −1.903°(전체 이동의 대부분) | `results/speculative/consolidated/mae_cdf.png`, `paired_stats_combined_vs_baseline.json` | **본문** — 두 모듈을 독립적으로 설계·분석할 수 있다는 방법론적 근거이자 핵심 결과 | 강함 — full-scale, paired 통계, 두 독립 경로(CDF 시각 + 수치 분해)가 일치 |
| 2 | **정확도+latency 동시 개선 (config D)** | RecentK-2+Speculative(D)는 이 프로젝트에서 유일하게 정확도 개선과 latency 감소를 동시에 달성한 설정이다. | MAE 12.799→10.895(−14.87%), latency 571.7ms→122.2ms(4.68배), forward 20→4.01 | `results/speculative/consolidated/ablation_bars.png`, `final_table.{csv,md}` | **본문** — headline result | 강함 — full-scale, 동일 checkpoint/GPU 통제 비교 |
| 3 | **Tail 귀속 (selector vs. speculative)** | 정확도가 악화된 상위 5% 샘플 전부(100%)에서 오차의 지배적 원인이 RecentK-2 selector이지 speculative decoding이 아니다. | 85/85 샘플에서 `|diff(B,A)| >= |diff(D,B)|`; 예: sample 116 selector +22.01°, speculative −0.07° | `results/speculative/consolidated/tail_velocity_vs_diff.png`, `tail_analysis_stats.json` | **본문** (한계 분석 절) — "왜 47%가 개별적으로는 악화되는가"에 대한 직접 답 | 강함 — full-scale, 예외 없는 100% 귀속 |
| 4 | **고분산 극단치 (fan-shaped variance)** | 전체 모집단에서는 모션 속도가 빠를수록 오히려 개선되는 경향(음의 상관)이지만, 최악 5%는 평균보다 2.16배 빠른 고모션 구간에 집중돼 있다 — "고모션=나쁨"이 아니라 "고모션=결과 분산 증가". | population Spearman rho=−0.400(p=2.8e-66); top-5% mean motion speed 3.49 vs 1.62 deg/step(Mann-Whitney p=3.0e-24) | `results/speculative/consolidated/tail_velocity_vs_diff.png` | **본문** — 반직관적이고 논문의 discussion을 풍부하게 만드는 결과, adaptive-K 제안의 직접 근거 | 강함 — full-scale, 두 통계 검정(상관+집단비교)이 상호보완적으로 같은 이야기를 함 |
| 5 | **Acceptance 천장 (accept rate near-ceiling)** | draft(constant-velocity) 모델이 거의 모든 곳에서 근접 천장 수준으로 수락되어, threshold/gamma를 더 튜닝해도 얻을 여지가 거의 없다. | mean accept rate 6.22/8(77.75%), median 6.33, std 0.22, ~99%가 단일 히스토그램 구간 | `results/speculative/consolidated/accept_rate_histogram.png` | **본문** (근거 절) 또는 **부록** — "다음 단계는 selector 쪽"이라는 스코프 결정의 실증 근거로 본문에 넣을 가치가 있으나, 그 자체로 headline은 아님 | 중간 — full-scale이지만 하나의 요약통계(narrow distribution)에 의존, iteration-position 세부 분해는 미확보 |
| 6 | **Threshold 둔감성** | acceptance threshold를 0.35→2.5로 7배 넓혀도 MAE는 12.83~12.93 사이에서만 움직인다 — 급격한 정확도 절벽이 이 범위에 없다. | 4개 threshold 지점, full 1,698-sample, MAE 12.831/12.849/12.893/12.929 | `results/speculative/consolidated/threshold_vs_mae.png` | **본문** (강건성 절) 또는 **부록** | 중간-강함 — full-scale이지만 4개 지점뿐이라 "절벽이 아예 없다"는 주장은 이 범위 밖에서는 미검증 |
| 7 | **AttentionTopK negative result** | 첫 decoder layer attention 기반 선택이 이론적으로는 더 정교해 보이지만, 모든 K에서 단순 RecentK보다 진다. | K=8: +0.038°, K=6: +0.352°, K=4: +0.939°, K=2: +1.172° (RecentK 대비, 50-sample) | `experiments/vp/attention_topk_7b_smoke/smoke_result.json`, `results/presentation_20260816/module1_token_selection.png` | **본문** (negative result로 명시) 또는 **부록** | **약함-중간 — 50-sample 한정.** 격차가 K가 작아질수록 커지는 일관된 방향성은 있으나 표본이 작아 본문에 쓸 경우 반드시 "50-sample" 표기 필수 |
| 8 | **Recency 지배 3중 증거** | (i) 순수 recency 기반 draft 모델도 실제 target 모델과 자주 일치, (ii) RecentK-2가 전체 10-step 이력을 쓰는 것보다 오히려 더 정확, (iii) attention 기반 선택이 항상 짐 — 세 독립 경로가 모두 같은 방향을 가리킴. | (ii) MAE 10.847(K=2, full-1698) vs 12.799(K=10=baseline, full-1698); (iii) 항목 7과 동일 | `results/presentation_20260816/module1_token_selection.png` | **본문** — 세 증거 중 (i)(ii)는 full-scale, (iii)만 50-sample이므로 논문에서는 "세 가지 중 하나는 50-sample 보조 증거"로 구분해서 서술 | 강함(부분) — (i)(ii)는 강함, (iii)는 항목 7과 동일하게 50-sample 한계 있음 |
| 9 | **Threshold=0 동등성 게이트** | speculative decoding 구현이 threshold=0에서 baseline과 완전히 동일한 출력을 낸다는 정합성 검증. | tiny model atol=1e-5(fp32)/2e-3(fp16); 7B random-head max diff 0.00122; 7B+RecentK-2 max diff 0.00146 | (수치만, 그림 없음) | **부록** (방법론 검증 절) — 논문 headline이 아니라 구현 신뢰성의 전제조건 | 강함이지만 성격이 다름 — 이것은 "결과"가 아니라 "결과를 믿을 수 있는 근거"이므로 본문 최상단이 아닌 방법론/부록에 위치해야 함 |
| 10 | **Wu2017 분포 이동 일반화 spot-check** | Jin2022로 fine-tune된 checkpoint가 unseen인 Wu2017에서도 정확도 개선과 latency 감소를 모두 유지한다. | A: MAE 15.476→D: 13.050, latency 567.0ms→121.0ms(~4.7배) | (표만, 그림 없음) | **부록** — 본문 결과의 일반화 가능성을 보강하는 보조 증거 | **중간 — 200/1,395 samples(spot-check), full-scale 아님.** 방향은 일관되지만 이 규모로 "일반화된다"는 강한 주장은 과함 |

## 요약: 강도별 재분류

- **본문에 바로 쓸 수 있는 full-scale 강한 결과**: #1, #2, #3, #4 (모두
  1,698-sample, 통계적으로 뒷받침됨, 서로 다른 각도에서 하나의 일관된
  이야기 — "selector가 정확도를, speculative가 latency를 각자 담당하고
  그 결합이 가산적이다").
- **본문 보강/강건성 절에 쓸 수 있는 중간 강도**: #5, #6 (full-scale이지만
  단일 요약통계 또는 좁은 sweep 범위).
- **50-sample 또는 spot-check로 명시적으로 구분해야 하는 보조 증거**:
  #7, #8(부분), #10 — 논문에 넣더라도 표본 크기를 반드시 병기.
- **방법론/부록 전용**: #9 (검증 게이트, headline 아님).

## 발표에만 어울리고 논문에는 부적합한 항목

없음 — 위 10개 전부 논문 어딘가(본문 또는 부록)에 배치 가능하다고 판단.
다만 `results/presentation_20260816/presentation_storyline.md`가 시간
제약상 발표에서는 이 중 4개만(모듈①의 #7/#8, 모듈②의 #6, 심층분석의
#3/#4/#5)만 다룬다 — 발표 선별과 논문 배치는 별개 결정이다.
