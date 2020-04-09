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
import unittest
from ..bradley_terry import (
    expand_to_slices,
    get_bt_summation_terms,
    get_bt_derivatives,
    sum,
)
from .assertions import assert_close


class TestBradleyTerryFunctions(unittest.TestCase):
    """Tests for functions in the bradley_terry module"""

    def setUp(self):
        np.seterr(all="raise")
        self.assert_close = assert_close.__get__(self, self.__class__)

    def test_expand_to_slices(self):
        """Test expand_to_slices()"""
        slices = [(0, 2), (2, 5)]
        values = np.array([1.0, 10.0])
        expanded = expand_to_slices(values, slices)
        self.assertSequenceEqual([1.0, 1.0, 10.0, 10.0, 10.0], expanded.tolist())

    def test_get_bt_summation_terms(self):
        """Test get_bt_summation_terms()"""
        gamma = np.array([1.0, 2.0])
        adversary_gamma = np.array([1.0, 2.0])
        d1, d2 = get_bt_summation_terms(gamma, adversary_gamma)
        self.assert_close([0.5, 0.5], d1, "d1")
        self.assert_close([0.25, 0.25], d2, "d2")

    def test_sum(self):
        """Test sum()"""
        x = np.array([1.0, 2.0, 4.0, 8.0])
        self.assertEqual(15.0, sum(x, 0, 4))
        self.assertEqual(0.0, sum(x, 0, 0))
        self.assertEqual(6.0, sum(x, 1, 3))
        self.assertEqual(7.0, sum(x, 0, 3))

    def test_sum(self):
        """Test sum() error compensation"""
        x = np.full([10], 0.1)
        self.assertEqual(1.0, sum(x, 0, 10))
        x = np.array([1e100, -1.0, -1e100, 1.0])
        self.assertEqual(0.0, sum(x, 0, 4))
        x = np.array([1e100, 1.0, -1e100, 1.0])
        self.assertEqual(2.0, sum(x, 0, 4))

    def test_get_bt_derivatives_single_win(self):
        """Test get_bt_derivatives() with a single win"""
        slices = [(0, 1)]
        wins = np.array([1.0])
        gamma = np.array([1.0])
        adversary_gamma = np.array([1.0])
        d1, d2 = get_bt_derivatives(slices, wins, gamma, adversary_gamma)
        self.assert_close([0.5], d1, "d1")
        self.assert_close([-0.25], d2, "d2")

    def test_get_bt_derivatives_single_loss(self):
        """Test get_bt_derivatives() with a single loss"""
        slices = [(0, 1)]
        wins = np.array([0.0])
        gamma = np.array([1.0])
        adversary_gamma = np.array([1.0])
        d1, d2 = get_bt_derivatives(slices, wins, gamma, adversary_gamma)
        self.assert_close([-0.5], d1, "d1")
        self.assert_close([-0.25], d2, "d2")

    def test_get_bt_derivatives_four_losses(self):
        """Test get_bt_derivatives() with four losses"""
        slices = [(0, 4)]
        wins = np.array([0.0])
        gamma = np.array([4.0, 4.0, 4.0, 4.0])
        adversary_gamma = np.array([1.0, 1.0, 1.0, 1.0])
        d1, d2 = get_bt_derivatives(slices, wins, gamma, adversary_gamma)
        self.assert_close([-3.2], d1, "d1")
        self.assert_close([-0.64], d2, "d2")

    def test_get_bt_derivatives_no_ascents(self):
        """Test get_bt_derivatives() with no ascents"""
        slices = [(0, 0)]
        wins = np.array([])
        gamma = np.array([])
        adversary_gamma = np.array([])
        d1, d2 = get_bt_derivatives(slices, wins, gamma, adversary_gamma)
        self.assert_close([0.0], d1, "d1")
        self.assert_close([0.0], d2, "d2")

    def test_get_bt_derivatives(self):
        """Test get_bt_derivatives() with multiple slices"""
        slices = [(0, 1), (1, 4)]
        wins = np.array([1.0, 2.0])
        gamma = np.array([6.0, 4.0, 4.0, 4.0])
        adversary_gamma = np.array([6.0, 4.0, 12.0, 12.0])
        d1, d2 = get_bt_derivatives(slices, wins, gamma, adversary_gamma)
        self.assert_close([0.5, 1.0], d1, "d1")
        self.assert_close([-0.25, -0.625], d2, "d2")
