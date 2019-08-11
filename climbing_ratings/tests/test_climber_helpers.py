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
import climbing_ratings.climber_helpers as climber_helpers
from climbing_ratings.tests.assertions import assert_close


class TestClimberHelpers(unittest.TestCase):
    """Tests for the climber_helpers extension."""
    def setUp(self):
        np.seterr(all='raise')
        self.assert_close = assert_close.__get__(self, self.__class__)
        
    def test_lu_decomposition_helper(self):
        """Test the lu_decomposition_helper function"""
        c = np.array([0., 1., -3.])
        hd = np.array([1., 2., 11.])
        climber_helpers.lu_decomposition_helper(c, hd)
        d = hd  # output parameter
        self.assert_close([1., 3., 10.], d, 'd')

    def test_ly_helper(self):
        """Test the ly_helper function"""
        g = np.array([10., 5., 32.])
        a = np.array([0., -0.1, 2.])
        climber_helpers.ly_helper(g, a)
        y = a  # output parameter
        self.assert_close([10., 4., 40.], y, 'y')

    def test_ux_helper(self):
        """Test the ux_helper function"""
        b = np.array([-2, 1.])
        d = np.array([1., 3., 10.])
        y = np.array([10., 4., 40.])
        climber_helpers.ux_helper(b, d, y)
        x = y  # output parameter
        self.assert_close([10., 0., 4.], x, 'x')
