"""유효 착순 판정 헬퍼. final_rank의 0(미기입)·90번대(비착순 상태 코드)를 걸러낸다."""
import pandas as pd

from config import NON_FINISH_RANK_MIN


def valid_rank_mask(final_rank: pd.Series) -> pd.Series:
    """1 <= final_rank < NON_FINISH_RANK_MIN 인 행이면 True. 결측/비숫자는 False."""
    ranks = pd.to_numeric(final_rank, errors="coerce")
    return (ranks >= 1) & (ranks < NON_FINISH_RANK_MIN)
