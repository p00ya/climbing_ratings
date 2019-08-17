"""Tests for the climber_helpers extension"""

# Copyright 2019 Dean Scarff
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
from .. import climber_helpers
from .assertions import assert_close


class TestClimberHelpers(unittest.TestCase):
    """Tests for the climber_helpers extension."""

    def setUp(self):
        np.seterr(all="raise")
        self.assert_close = assert_close.__get__(self, self.__class__)

    def test_solve_lu_d(self):
        """Test solve_lu_d()"""
        c = np.array([0.0, 1.0, -3.0])
        hd = np.array([1.0, 2.0, 11.0])
        climber_helpers.solve_lu_d(c, hd)
        d = hd  # output parameter
        self.assert_close([1.0, 3.0, 10.0], d, "d")

    def test_solve_ul_d(self):
        """Test solve_lu_d()"""
        c = np.array([0.0, 1.0, -3.0])
        hd = np.array([1.0, 2.0, 11.0])
        climber_helpers.solve_ul_d(c, hd)
        d = hd  # output parameter
        self.assert_close([1.0, 247.0 / 118.0, 118.0 / 11.0], d, "d")

    def test_solve_y(self):
        """Test solve_y()"""
        g = np.array([10.0, 5.0, 32.0])
        a = np.array([0.0, -0.1, 2.0])
        climber_helpers.solve_y(g, a)
        y = a  # output parameter
        self.assert_close([10.0, 4.0, 40.0], y, "y")

    def test_solve_x(self):
        """Test solve_x()"""
        b = np.array([-2, 1.0])
        d = np.array([1.0, 3.0, 10.0])
        y = np.array([10.0, 4.0, 40.0])
        climber_helpers.solve_x(b, d, y)
        x = y  # output parameter
        self.assert_close([10.0, 0.0, 4.0], x, "x")
