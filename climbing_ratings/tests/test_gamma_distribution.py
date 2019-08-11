"""Tests for the gamma_distribution module"""

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
from ..gamma_distribution import GammaDistribution
from .assertions import assert_close


class TestGammaDistribution(unittest.TestCase):
    """Tests for the GammaDistribution class"""

    def setUp(self):
        np.seterr(all='raise')
        self.assert_close = assert_close.__get__(self, self.__class__)
        
    def test_get_derivatives_mode_1(self):
        """Test GammaDistribution(1).get_derivatives()"""
        mu = np.array([1., 1., 1.])
        x = np.array([1., 2., 3.])
        g = GammaDistribution(mu)
        d1, d2 = g.get_derivatives(x)
        self.assert_close([0., -1., -2.], d1, 'd1')
        self.assert_close([-1., -2., -3.], d2, 'd2')

    def test_get_derivatives_mode_2(self):
        """Test GammaDistribution(2).get_derivatives()"""
        mu = np.array([2., 2., 2.])
        x = np.array([1., 2., 3.])
        g = GammaDistribution(mu)
        d1, d2 = g.get_derivatives(x)
        self.assert_close([0.5, 0., -0.5], d1, 'd1')
        self.assert_close([-0.5, -1., -1.5], d2, 'd2')
