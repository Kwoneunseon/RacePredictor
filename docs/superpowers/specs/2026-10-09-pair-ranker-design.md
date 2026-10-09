# 복연승 페어 랭커(3차): rank:pairwise 전환 설계

## 배경

1차(튜닝 없는 XGBoost 분류기, 10개 피처): test 적중률 38.5%.
2차(경주-상대 피처 3개 추가): 39.2%(+0.7%p). 둘 다 기존 오프셋 로지스틱
결과 40.0%에 못 미쳤다(메모리 [[project_multi_bet_probability_model]]
"2026-10-09 복연승 페어 분류기(XGBoost) 결과"/"2026-10-09 복연승 페어
분류기 2차(경주-상대 피처) 결과" 절 참고).

2차에서 `p_indep_rank_in_race`(경주 내 순위)가 피처 중요도 1위(0.4311)로
올라서며 "학습(독립 이진분류)과 평가(경주 내 최고픽) 목표 불일치"
가설이 맞았음을 확인했지만, 손으로 만든 상대 피처만으로는 간극을 완전히
못 메웠다(0.8%p 부족).

이번 3차는 그 간극을 피처 추가가 아니라 **모델의 목적함수 자체**로
메운다. XGBoost의 `rank:pairwise` 목적함수는 같은 그룹(경주) 안에서
"어느 쪽이 더 높아야 하는가"를 직접 학습 목표로 삼는다 — 평가 지표
(경주 내 최고픽 적중률)와 학습 목표가 구조적으로 일치한다.

## 목표

같은 피처 13개, 같은 하이퍼파라미터 규모로 목적함수만 `rank:pairwise`로
바꿔서, 2차(39.2%)를 넘고 오프셋 로지스틱(40.0%)에 도달하는지 확인한다.

## 결정된 사항

- **기존 `train_and_evaluate`(분류기 버전, 1~2차에서 완성)는 수정하지
  않는다.** 새 함수 `train_and_evaluate_ranker`를 추가하고, CLI에
  `--model {classifier,ranker}` 플래그를 추가해 기본값(`classifier`)은
  기존 동작을 그대로 유지한다. 1~2차 결과의 재현성을 깨지 않는다.
- **`build_classifier_pair_table`/`add_race_relative_features`는 수정
  하지 않는다.** `train_and_evaluate_ranker`도 이 둘을 그대로 호출해
  같은 `pair_df`(13개 피처 포함)를 만든다.
- **피처 13개, 하이퍼파라미터 규모 전부 1~2차와 동일하게 유지한다**
  (`n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8, random_state=42`, `objective`만
  `"rank:pairwise"`로 교체). 바뀌는 변수를 하나로 제한해, 이번 결과가
  "목적함수 전환의 효과"임을 명확히 한다.
- **튜닝(Optuna)·앙상블·프로덕션 연결·ROI 계산 없음**(1~2차와 동일한
  범위 외 항목).
- **dev/test 창도 동일하게 유지**: dev≤2026-04-30, test
  2026-05-01~06-25(408경주).

## 기술적 제약: XGBRanker의 qid 요구사항

`xgboost.XGBRanker.fit(X, y, qid=...)`는 **같은 qid(경주)에 속한 행이
데이터 안에서 서로 붙어있어야 한다**(sklearn API 공식 요구사항,
설치된 xgboost 2.1.1에서 확인됨). `predict()`는 이 제약이 없다(행별로
독립적으로 점수를 내므로 순서 무관).

따라서:
- **학습(`fit`) 직전에만** 데이터를 `race_id`로 정렬한다
  (`df.sort_values("race_id").reset_index(drop=True)`).
- **예측(`predict`)은 정렬하지 않고 그대로 쓴다.**
- GroupKFold로 나눈 fold의 train/val 서브셋도 `fit`에 넘기기 전에
  각각 `race_id`로 정렬한다.

## 평가 방식 변경

랭커는 확률을 출력하지 않으므로(`predict()`가 임의 스케일의 relevance
점수를 반환) 기존 `accuracy_score`/`log_loss`(이진분류 지표)는 쓸 수
없다. 대신 **매 fold마다, 그리고 최종 test 평가에서, 1~2차와 동일한
"경주별 최고 점수 페어가 label==1인 비율"(최고픽 적중률)을 직접
계산한다** — 평가 지표를 학습 목표와 통일하는 것이 이번 설계의 핵심
이므로, fold 간 비교 지표도 동일한 기준으로 맞춘다.

```python
def _pick_accuracy(df: pd.DataFrame, score_col: str = "p_joint") -> float:
    """경주별로 score_col이 가장 높은 행을 골라 label==1 비율을 낸다."""
    best_idx = df.groupby("race_id")[score_col].idxmax()
    return df.loc[best_idx]["label"].mean()
```

## 신규 함수

`scripts/diagnose_pair_classifier.py`에 `train_and_evaluate` 함수 뒤에
추가:

- `_pick_accuracy(df, score_col="p_joint") -> float` (위 코드, 순수 함수,
  단위테스트 대상)
- `train_and_evaluate_ranker(load_start, dev_end, test_start, test_end)`:
  `train_and_evaluate`와 데이터 로드/분리/리포트 구조는 동일하되, 모델을
  `XGBClassifier` 대신 `XGBRanker(objective="rank:pairwise", ...)`로
  바꾸고, fold/최종 학습 전 `race_id` 정렬, fold 평가를
  `_pick_accuracy`로 교체한다. `capture_report("pair_ranker")`로 별도
  리포트 접두사를 쓴다(1~2차 `"pair_classifier"` 리포트와 파일명이
  섞이지 않도록).

CLI(`if __name__ == "__main__":` 블록)에 `--model` 인자를 추가해
`"classifier"`(기존 동작, 기본값)와 `"ranker"`(신규) 중 선택하도록
분기한다.

## 에러 처리/엣지케이스

- 1~2차와 동일한 가드(모델/데이터 없음, dev/test 빈 경우 조기 반환)를
  그대로 따른다 — `_load_predictions_with_probs`, `build_classifier_pair_table`,
  `add_race_relative_features`를 그대로 재사용하므로 이 부분은 이미
  검증돼 있다.
- `_pick_accuracy`가 빈 DataFrame을 받으면(이론상 `pair_df`가 비었을 때)
  `groupby("race_id").idxmax()`가 빈 Series를 반환하고 `.mean()`은
  `NaN`을 낸다 — 다만 `train_and_evaluate_ranker`는 그 전에
  `pair_df.empty`/`dev_df.empty`/`test_df.empty` 체크로 조기 반환하므로
  `_pick_accuracy`가 실제로 빈 입력을 받는 경로는 없다. 함수 자체의
  동작은 문서화만 하고 추가 가드는 넣지 않는다(YAGNI).

## 테스트 방침

`_pick_accuracy`는 순수 함수이므로 합성 데이터로 단위테스트를 추가한다:
경주별로 점수가 다른 페어들 중 최고점이 label==1인 경주 비율이 올바르게
계산되는지, 여러 경주가 섞여도 경주별로 올바르게 분리되는지.

`train_and_evaluate_ranker`는 DB/모델 의존이라 기존 관례대로 단위테스트
대상이 아니다. 실행+결과 해석으로 검증한다.

## 판정 기준

- **test 최고 페어 픽 적중률이 2차(39.2%)보다 오르고 40.0%(오프셋
  로지스틱)에 도달하거나 넘으면** → 목적함수 전환이 남은 간극을 메운
  것. 이 모델링 경로가 오프셋 로지스틱과 동등하거나 그 이상이라는
  근거 확보.
- **39.2%와 40.0% 사이면** → 또 부분 개선. 이 시점에서는 모델링 축
  (피처 추가·목적함수 전환)으로 3연속 "부분 개선"만 나온 것이므로,
  추가 투자보다 오프셋 로지스틱(40.0%)을 실무 상한으로 받아들이고 ROI/
  배당 데이터 쪽으로 넘어가는 것을 권고안으로 보고한다.
- **39.2% 이하면** → 목적함수 전환이 오히려 도움이 안 된 것(피처
  엔지니어링이 이 방향에서는 더 유효했다는 뜻). 모델링 축을 종료하고
  오프셋 로지스틱(40.0%)을 최종 상한으로 기록, ROI/배당 데이터 쪽으로
  넘어가는 것을 권고안으로 보고한다.
- 세 경우 모두 피처 중요도(1~2차와 나란히)와 fold별 최고픽 적중률을
  함께 보고한다.

## 범위 외 (이번에 하지 않음)

- 하이퍼파라미터 튜닝(Optuna), `rank:ndcg`/`rank:map` 등 다른 랭킹
  목적함수와의 비교
- 앙상블
- `core/predictor.py`/`core/ev_calculator.py`/`core/investor.py` 연결
- ROI/EV 계산
- `train_and_evaluate`/`build_classifier_pair_table`/
  `add_race_relative_features` 자체 수정
