"""Harville(Plackett-Luce) 순차 소거 방식으로 경주 내 순위 확률분포를 유도.

각 말의 강도 점수(이미 캘리브레이션된 win_probability, 경주 내 합=1)를
입력받아 1~3등 배정 확률(연승)과 특정 두 말이 동시에 1~3등에 들어갈
확률(복연승)을 계산한다.

방법론 및 롤아웃 원칙은 [[project_multi_bet_probability_model]]
(docs/superpowers/specs/2026-07-25-multi-bet-probability-model-design.md
Phase 2) 참고.

핵심 아이디어: 정식 Plackett-Luce는 전체 순열(n!)에 대한 확률을 정의하지만,
연승/복연승은 상위 3자리 배정만 필요하므로 길이 3인 순서쌍(ordered triple)
n(n-1)(n-2)가지만 열거하면 충분하다(출전마 수 n<=14 기준 최대 2184가지 —
비용 무시 가능). 이 순서쌍의 확률은 Harville 순차 소거 공식을 그대로 쓴다:

    P(1등=i, 2등=j, 3등=k) = s_i/S * s_j/(S-s_i) * s_k/(S-s_i-s_j)

여기서 S는 전체 강도 합(정규화 후 1).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "top3_order_probabilities",
    "top3_marginal_probabilities",
    "pair_top3_probabilities",
]


def _normalize(strengths) -> np.ndarray:
    s = np.asarray(strengths, dtype=float)
    if s.ndim != 1:
        raise ValueError("strengths는 1차원 배열이어야 합니다")
    if len(s) < 3:
        raise ValueError("출전마 수가 3 미만이면 top3 정의가 불가능합니다")
    if np.any(s < 0):
        raise ValueError("강도 점수는 음수일 수 없습니다")
    total = s.sum()
    if total <= 0:
        raise ValueError("strengths 합이 0 이하입니다")
    return s / total


def top3_order_probabilities(strengths) -> dict[tuple[int, int, int], float]:
    """모든 (1등,2등,3등) 순서쌍(인덱스 기준)의 Harville 확률을 계산.

    강도 점수는 내부에서 정규화(합=1)하므로 호출자가 미리 정규화하지
    않아도 된다. 반환값 dict의 합은 1.0이 되어야 한다(전체 확률질량 보존).
    """
    s = _normalize(strengths)
    n = len(s)

    probs: dict[tuple[int, int, int], float] = {}
    total = 1.0
    for i in range(n):
        p_i = s[i] / total
        rem_i = total - s[i]
        if rem_i <= 0:
            continue
        for j in range(n):
            if j == i:
                continue
            p_j = s[j] / rem_i
            rem_ij = rem_i - s[j]
            if rem_ij <= 0:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                p_k = s[k] / rem_ij
                probs[(i, j, k)] = p_i * p_j * p_k
    return probs


def top3_marginal_probabilities(strengths) -> np.ndarray:
    """각 말이 1~3등 중 어디든 들어갈 확률(연승확률). 전체 합은 3.0이어야 정상."""
    s = _normalize(strengths)
    n = len(s)
    order_probs = top3_order_probabilities(s)
    marg = np.zeros(n)
    for (i, j, k), p in order_probs.items():
        marg[i] += p
        marg[j] += p
        marg[k] += p
    return marg


def pair_top3_probabilities(strengths) -> np.ndarray:
    """모든 말 쌍(i, j)에 대해 두 말이 동시에 1~3등에 들어갈 확률(복연승) 행렬.

    n x n 대칭 행렬, 대각선은 0. 상위 삼각합(각 쌍 1회)은
    C(3,2)=3 * (경주 내 top3 조합 수) 형태로 이론상 3이 되어야 한다
    (연승 확률 합이 3인 것과 동일한 이유 — 매 순서쌍이 3개의 쌍을 기여).
    """
    s = _normalize(strengths)
    n = len(s)
    order_probs = top3_order_probabilities(s)
    pair = np.zeros((n, n))
    for (i, j, k), p in order_probs.items():
        for a, b in ((i, j), (i, k), (j, k)):
            pair[a, b] += p
            pair[b, a] += p
    return pair
