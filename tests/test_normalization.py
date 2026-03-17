"""Tests for percentile_rank() in bor_risk.utils.

Covers: boundary anchoring, ties, constant reference, degenerate cases,
clipping behaviour (below min, above max).
"""

import pytest

from bor_risk.utils import percentile_rank


class TestPercentileRankBoundaries:
    def test_min_value_returns_zero(self):
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile_rank(1.0, ref) == 0.0

    def test_max_value_returns_one(self):
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile_rank(5.0, ref) == 1.0

    def test_below_min_clips_to_zero(self):
        ref = [2.0, 3.0, 4.0]
        assert percentile_rank(0.0, ref) == 0.0

    def test_above_max_clips_to_one(self):
        ref = [2.0, 3.0, 4.0]
        assert percentile_rank(10.0, ref) == 1.0

    def test_midpoint_of_five_element_ref(self):
        ref = [0.0, 1.0, 2.0, 3.0, 4.0]
        # raw=2.0 is the middle value (index 2 of 5, 0-indexed) → (2+2)/2 / 4 = 0.5
        result = percentile_rank(2.0, ref)
        assert result == pytest.approx(0.5)

    def test_interior_value_between_elements(self):
        ref = [0.0, 2.0, 4.0, 6.0]
        # raw=3.0 lies between index 1 (2.0) and index 2 (4.0)
        # bisect_left=2, bisect_right=2; mid_rank=(2+2-1)/2 = 1.5; result = 1.5/3
        result = percentile_rank(3.0, ref)
        assert result == pytest.approx(1.5 / 3.0)


class TestPercentileRankTies:
    def test_tied_values_share_average_rank(self):
        ref = [0.0, 1.0, 1.0, 1.0, 2.0]
        # All three 1.0 entries: bisect_left=1, bisect_right=4
        # mid_rank = (1+4-1)/2 = 2.0; result = 2.0/(5-1) = 0.5
        assert percentile_rank(1.0, ref) == pytest.approx(0.5)

    def test_all_tied_except_extremes(self):
        ref = [0.0, 5.0, 5.0, 5.0, 10.0]
        r1 = percentile_rank(5.0, ref)
        assert 0.0 < r1 < 1.0

    def test_two_equal_interior_values(self):
        ref = [1.0, 3.0, 3.0, 5.0]
        r = percentile_rank(3.0, ref)
        # bisect_left=1, bisect_right=3; mid_rank=(1+3-1)/2=1.5; /3 = 0.5
        assert r == pytest.approx(1.5 / 3.0)


class TestPercentileRankConstantReference:
    def test_constant_reference_returns_half(self):
        ref = [5.0, 5.0, 5.0]
        assert percentile_rank(0.0, ref) == 0.5
        assert percentile_rank(5.0, ref) == 0.5
        assert percentile_rank(100.0, ref) == 0.5

    def test_single_element_returns_half(self):
        assert percentile_rank(42.0, [10.0]) == 0.5

    def test_empty_reference_returns_half(self):
        assert percentile_rank(42.0, []) == 0.5
