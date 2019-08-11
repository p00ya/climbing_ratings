"""Tests for the climber module"""

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

import unittest
import numpy as np
import climbing_ratings.climber as climber
from climbing_ratings.tests.assertions import assert_close


class TestClimberFunctions(unittest.TestCase):
    """Tests for functions in the climber module"""
    
    def setUp(self):
        np.seterr(all='raise')
        self.assert_close = assert_close.__get__(self, self.__class__)
        
    def test_lu_decomposition(self):
        """Test that for (L, U) = lu_decomposition(M), LU = M"""
        m = np.array([
            1., -2., 0.,
            0.5, 2., 1.,
            0., 3., 11.
        ]).reshape((3, 3))
        md = np.diag(m)
        mu = np.diag(m, 1)
        ml = np.diag(m, -1)
        tri_diagonal = climber.TriDiagonal(md, mu, ml)
        lu = climber.lu_decomposition(tri_diagonal)

        # Reconstruct the L and U matrices.
        u_matrix = np.diagflat(lu.d) + np.diagflat(lu.b, 1)
        l_matrix = np.eye(3) + np.diagflat(lu.a, -1)
        lu_matrix = np.dot(l_matrix, u_matrix)
        # Test that M = LU.
        self.assert_close(m, lu_matrix, 'M')

    def test_invert_h_dot_g(self):
        """Test that for X = invert_h_dot_g(LU, G), LU X = G"""
        g = np.array([10., 5., 32.])
        d = np.array([1., 3., 10.])
        b = np.array([-2, 1.])
        a = np.array([0.1, -2.])
        lu = climber.TriDiagonalLU(d, b, a)
        x = climber.invert_h_dot_g(lu, g)

        u_matrix = np.diagflat(d) + np.diagflat(b, 1)
        l_matrix = np.eye(3) + np.diagflat(a, -1)
        lu_matrix = np.dot(l_matrix, u_matrix)

        # Test that LU X = G.
        lux = np.dot(lu_matrix, x)
        self.assert_close(g, lux, 'LU X')


class TestClimber(unittest.TestCase):
    """Tests for the Climber class"""
    
    def setUp(self):
        np.seterr(all='raise')
        self.assert_close = assert_close.__get__(self, self.__class__)

    def test_one_on_sigma_sq(self):
        """Test Climber.one_on_sigma_sq"""
        gaps = np.array([1., 2.])
        climber.Climber.wiener_variance = 10.
        c = climber.Climber(gaps)
        one_on_sigma_sq = c.one_on_sigma_sq
        self.assert_close([0.1, 0.05], one_on_sigma_sq, 'one_on_sigma_sq')

    def test_get_ratings_adjustment(self):
        """Test Climber.get_ratings_adjustment"""
        gaps = np.array([1.])
        ratings = np.array([6., 4.])
        bt_d1 = np.array([0.5, 1.])
        bt_d2 = np.array([-0.25, -0.625])
        c = climber.Climber(gaps)
        delta = c.get_ratings_adjustment(ratings, bt_d1, bt_d2)
        self.assert_close([0.68422919, 0.0551964], delta, 'delta')
