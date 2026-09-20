"""유효 착순(1 <= final_rank < NON_FINISH_RANK_MIN) 판정 헬퍼 테스트."""
import pandas as pd

from config import NON_FINISH_RANK_MIN
from Utils.ranks import valid_rank_mask


def test_non_finish_rank_min_is_90():
    assert NON_FINISH_RANK_MIN == 90


def test_valid_rank_mask_keeps_only_finishing_ranks():
    s = pd.Series([0, 1, 3, 16, 89, 90, 91, 92, 93, 94, 95, 99])
    assert valid_rank_mask(s).tolist() == [
        False, True, True, True, True, False, False, False, False, False, False, False
    ]


def test_valid_rank_mask_treats_null_and_non_numeric_as_invalid():
    s = pd.Series([None, "2", "x", 1.0, float("nan")], dtype=object)
    assert valid_rank_mask(s).tolist() == [False, True, False, True, False]
