"""Tests for the bradley_terry module"""

# Copyright Contributors to the Climbing Ratings project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy.typing import NDArray
import unittest
from ..bradley_terry import (
    BradleyTerry,
    expand_to_slices,
    expand_to_slices_sparse,
    _get_bt_summation_terms,
    _sum,
)
from .assertions import assert_close_get


_Array = NDArray[np.float_]


class TestBradleyTerryFunctions(unittest.TestCase):
    """Tests for functions in the bradley_terry module"""

    def setUp(self) -> None:
        np.seterr(all="raise")
        self.assert_close = assert_close_get(self, self.__class__)

    def test_expand_to_slices(self) -> None:
        """Test expand_to_slices()"""
        slices = [(0, 2), (2, 5)]
        values: _Array = np.array([1.0, 10.0])
        expanded = np.empty([5])
        expand_to_slices(values, slices, expanded)
        self.assertSequenceEqual([1.0, 1.0, 10.0, 10.0, 10.0], expanded.tolist())

    def test_expand_to_slices_sparse(self) -> None:
        """Test expand_to_slices_sparse()"""
        slices = [(1, 2), (3, 4)]
        values: _Array = np.array([1.0, 10.0])
        expanded = expand_to_slices_sparse(values, slices, 5)
        self.assertSequenceEqual([0.0, 1.0, 0.0, 10.0, 0.0], expanded.tolist())

    def test_get_bt_summation_terms(self) -> None:
        """Test _get_bt_summation_terms()"""
        win: _Array = np.array([1.0, 1.0])
        player = np.log([1.0, 2.0])
        adversary = np.log([1.0, 2.0])
        d1, d2 = _get_bt_summation_terms(win, player, adversary)
        self.assert_close([0.5, 0.5], d1, "d1")
        self.assert_close([-0.25, -0.25], d2, "d2")

    def test_sum(self) -> None:
        """Test sum()"""
        x: NDArray[np.longdouble] = np.array([1.0, 2.0, 4.0, 8.0], dtype="longdouble")
        self.assertEqual(15.0, _sum(x, 0, 4))
        self.assertEqual(0.0, _sum(x, 0, 0))
        self.assertEqual(6.0, _sum(x, 1, 3))
        self.assertEqual(7.0, _sum(x, 0, 3))

    def test_sum_error(self) -> None:
        """Test sum() error compensation with extended precision"""
        # These values should detect uncompensated error for both double
        # and extended precision floats.
        x: NDArray[np.longdouble] = np.full([10], 0.1, dtype="longdouble")
        self.assertEqual(1.0, _sum(x, 0, 10))
        x = np.array([1e100, -1.0, -1e100, 1.0], dtype="longdouble")
        self.assertEqual(0.0, _sum(x, 0, 4))
        x = np.array([1e100, 1.0, -1e100, 1.0], dtype="longdouble")
        self.assertEqual(2.0, _sum(x, 0, 4))


class TestBradleyTerry(unittest.TestCase):
    """Tests for the BradleyTerry class"""

    def setUp(self) -> None:
        np.seterr(all="raise")
        self.assert_close = assert_close_get(self, self.__class__)

    def test_get_derivatives_single_win(self) -> None:
        """Test get_derivatives() with a single win"""
        bt = BradleyTerry(1, 1)
        slices = [(0, 1)]
        win: _Array = np.array([1.0])
        bt.ratings[:] = np.log([1.0])
        adversary = np.log([1.0])
        d1, d2 = bt.get_derivatives(slices, win, adversary)
        self.assert_close([0.5], d1, "d1")
        self.assert_close([-0.25], d2, "d2")

    def test_get_derivatives_single_loss(self) -> None:
        """Test get_derivatives() with a single loss"""
        bt = BradleyTerry(1, 1)
        slices = [(0, 1)]
        win: _Array = np.array([-1.0])
        bt.ratings[:] = np.log([1.0])
        adversary = np.log([1.0])
        d1, d2 = bt.get_derivatives(slices, win, adversary)
        self.assert_close([-0.5], d1, "d1")
        self.assert_close([-0.25], d2, "d2")

    def test_get_derivatives_four_losses(self) -> None:
        """Test get_derivatives() with four losses"""
        bt = BradleyTerry(4, 1)
        slices = [(0, 4)]
        win: _Array = np.array([-1.0, -1.0, -1.0, -1.0])
        bt.ratings[:] = np.log([4.0, 4.0, 4.0, 4.0])
        adversary = np.log([1.0, 1.0, 1.0, 1.0])
        d1, d2 = bt.get_derivatives(slices, win, adversary)
        self.assert_close([-3.2], d1, "d1")
        self.assert_close([-0.64], d2, "d2")

    def test_get_derivatives_no_ascents(self) -> None:
        """Test get_derivatives() with no ascents"""
        bt = BradleyTerry(0, 1)
        slices = [(0, 0)]
        win = np.ones([0])
        bt.ratings[:] = np.log([])
        adversary = np.log([])
        d1, d2 = bt.get_derivatives(slices, win, adversary)
        self.assert_close([0.0], d1, "d1")
        self.assert_close([0.0], d2, "d2")

    def test_get_derivatives(self) -> None:
        """Test get_derivatives() with multiple slices"""
        bt = BradleyTerry(4, 2)
        slices = [(0, 1), (1, 4)]
        win: _Array = np.array([1.0, 1.0, 1.0, -1.0])
        bt.ratings[:] = np.log([6.0, 4.0, 4.0, 4.0])
        adversary = np.log([6.0, 4.0, 12.0, 12.0])
        d1, d2 = bt.get_derivatives(slices, win, adversary)
        self.assert_close([0.5, 1.0], d1, "d1")
        self.assert_close([-0.25, -0.625], d2, "d2")

    def test_get_derivatives_simple(self) -> None:
        """Test get_derivatives() with "simple" data"""
        bt = BradleyTerry(5, 2)
        slices = [(0, 3), (3, 5)]
        win: _Array = np.array([1.0, 1.0, 1.0, -1.0, 1.0])
        bt.ratings[:] = np.zeros(5)
        adversary: _Array = np.zeros(5)
        d1, d2 = bt.get_derivatives(slices, win, adversary)
        self.assert_close([1.5, 0.0], d1, "d1")
        self.assert_close([-0.75, -0.5], d2, "d2")
