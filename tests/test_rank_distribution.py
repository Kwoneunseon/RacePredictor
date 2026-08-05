import itertools

import numpy as np
import pytest

from core.rank_distribution import (
    top3_order_probabilities,
    top3_marginal_probabilities,
    pair_top3_probabilities,
)


def _brute_force_top3_marginals(strengths) -> np.ndarray:
    """전체 순열(n!)을 직접 열거하는 Plackett-Luce 순차소거 방식의
    독립적인 대조 구현. top3_marginal_probabilities()가 쓰는 '길이 3
    순서쌍만 열거하는' 지름길과는 다른 경로로 계산해 결과를 대조한다."""
    s = np.asarray(strengths, dtype=float)
    s = s / s.sum()
    n = len(s)
    marg = np.zeros(n)
    for perm in itertools.permutations(range(n)):
        total = 1.0
        p = 1.0
        for idx in perm:
            p *= s[idx] / total
            total -= s[idx]
        for pos in perm[:3]:
            marg[pos] += p
    return marg


def _brute_force_pair_top3(strengths) -> np.ndarray:
    s = np.asarray(strengths, dtype=float)
    s = s / s.sum()
    n = len(s)
    pair = np.zeros((n, n))
    for perm in itertools.permutations(range(n)):
        total = 1.0
        p = 1.0
        for idx in perm:
            p *= s[idx] / total
            total -= s[idx]
        top3 = perm[:3]
        for a, b in itertools.combinations(top3, 2):
            pair[a, b] += p
            pair[b, a] += p
    return pair


# ──────────────────────────────────
# 확률질량 보존(합=1, 합=3) 검증
# ──────────────────────────────────

def test_top3_order_probabilities_sum_to_one():
    np.random.seed(0)
    strengths = np.random.dirichlet(np.ones(8))
    order_probs = top3_order_probabilities(strengths)
    assert sum(order_probs.values()) == pytest.approx(1.0, abs=1e-9)


def test_top3_order_probabilities_count():
    """n두 경주에서 (1,2,3등) 순서쌍 개수는 n*(n-1)*(n-2)개."""
    n = 6
    strengths = np.full(n, 1.0 / n)
    order_probs = top3_order_probabilities(strengths)
    assert len(order_probs) == n * (n - 1) * (n - 2)


def test_top3_marginal_sums_to_three():
    np.random.seed(1)
    for n in (5, 8, 12, 14):
        strengths = np.random.dirichlet(np.ones(n))
        marg = top3_marginal_probabilities(strengths)
        assert marg.sum() == pytest.approx(3.0, abs=1e-9)
        assert (marg >= 0).all() and (marg <= 1 + 1e-9).all()


def test_pair_top3_upper_triangle_sums_to_three():
    np.random.seed(2)
    for n in (5, 8, 12):
        strengths = np.random.dirichlet(np.ones(n))
        pair = pair_top3_probabilities(strengths)
        iu = np.triu_indices(n, k=1)
        assert pair[iu].sum() == pytest.approx(3.0, abs=1e-9)


def test_pair_top3_matrix_symmetric_and_diag_zero():
    np.random.seed(3)
    n = 7
    strengths = np.random.dirichlet(np.ones(n))
    pair = pair_top3_probabilities(strengths)
    assert np.allclose(pair, pair.T)
    assert np.allclose(np.diag(pair), 0.0)


def test_equal_strengths_uniform_marginal():
    """모든 말의 강도가 동일하면 연승확률은 모두 3/n으로 균등해야 한다."""
    n = 10
    strengths = np.full(n, 1.0 / n)
    marg = top3_marginal_probabilities(strengths)
    assert np.allclose(marg, 3.0 / n, atol=1e-9)


def test_scale_invariance():
    """입력이 정규화되어 있지 않아도(합!=1) 동일한 결과를 내야 한다."""
    np.random.seed(4)
    n = 6
    raw = np.random.uniform(1, 100, n)
    marg_scaled = top3_marginal_probabilities(raw)
    marg_normalized = top3_marginal_probabilities(raw / raw.sum())
    assert np.allclose(marg_scaled, marg_normalized)


# ──────────────────────────────────
# 완전열거(brute-force) 대조 검증 — 소규모 출전마수
# ──────────────────────────────────

@pytest.mark.parametrize("n", [4, 5])
def test_marginal_matches_brute_force(n):
    np.random.seed(10 + n)
    strengths = np.random.dirichlet(np.ones(n))
    expected = _brute_force_top3_marginals(strengths)
    actual = top3_marginal_probabilities(strengths)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


@pytest.mark.parametrize("n", [4, 5])
def test_pair_matches_brute_force(n):
    np.random.seed(20 + n)
    strengths = np.random.dirichlet(np.ones(n))
    expected = _brute_force_pair_top3(strengths)
    actual = pair_top3_probabilities(strengths)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_marginal_matches_brute_force_skewed_strengths():
    """한 말이 압도적으로 강한 비대칭 케이스도 대조 검증."""
    strengths = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
    expected = _brute_force_top3_marginals(strengths)
    actual = top3_marginal_probabilities(strengths)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


# ──────────────────────────────────
# 예외 처리
# ──────────────────────────────────

def test_raises_when_fewer_than_three_horses():
    with pytest.raises(ValueError):
        top3_marginal_probabilities(np.array([0.5, 0.5]))


def test_raises_on_negative_strength():
    with pytest.raises(ValueError):
        top3_marginal_probabilities(np.array([0.5, -0.1, 0.6]))


def test_raises_on_zero_sum():
    with pytest.raises(ValueError):
        top3_marginal_probabilities(np.array([0.0, 0.0, 0.0]))
