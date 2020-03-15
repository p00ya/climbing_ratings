"""Tests for the climber module"""

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

import unittest
import numpy as np
from .. import climber
from ..normal_distribution import NormalDistribution
from .assertions import assert_close


class TestClimberFunctions(unittest.TestCase):
    """Tests for functions in the climber module"""

    m = np.array([1.0, -2.0, 0.0, 0.5, 2.0, 1.0, 0.0, 3.0, 11.0]).reshape((3, 3))

    def setUp(self):
        np.seterr(all="raise")
        self.assert_close = assert_close.__get__(self, self.__class__)

    def test_lu_decompose(self):
        """Test that for (L, U) = lu_decompose(M), LU = M"""
        m = self.__class__.m
        md = np.diag(m)
        mu = np.diag(m, 1)
        ml = np.diag(m, -1)
        tri_diagonal = climber.TriDiagonal(md, mu, ml)
        lu = climber.lu_decompose(tri_diagonal)

        # Reconstruct the L and U matrices.
        u_matrix = np.diagflat(lu.d) + np.diagflat(lu.b, 1)
        l_matrix = np.eye(3) + np.diagflat(lu.a, -1)
        lu_matrix = np.dot(l_matrix, u_matrix)
        # Test that M = LU.
        self.assert_close(m, lu_matrix, "M")

    def test_ul_decompose(self):
        """Test that for (U', L') = ul_decompose(M), U'L' = M"""
        m = self.__class__.m
        md = np.diag(m)
        mu = np.diag(m, 1)
        ml = np.diag(m, -1)
        tri_diagonal = climber.TriDiagonal(md, mu, ml)
        ul = climber.ul_decompose(tri_diagonal)

        # Reconstruct the L and U matrices.
        u_matrix = np.eye(3) + np.diagflat(ul.a, 1)
        l_matrix = np.diagflat(ul.d) + np.diagflat(ul.b, -1)
        ul_matrix = np.dot(u_matrix, l_matrix)
        # Test that M = U'L'
        self.assert_close(m, ul_matrix, "M")

    def test_invert_lu_dot_g(self):
        """Test that for X = invert_lu_dot_g(LU, G), LU X = G"""
        g = np.array([10.0, 5.0, 32.0])
        d = np.array([1.0, 3.0, 10.0])
        b = np.array([-2, 1.0])
        a = np.array([0.1, -2.0])
        lu = climber.TriDiagonalLU(d, b, a)
        x = climber.invert_lu_dot_g(lu, g)

        u_matrix = np.diagflat(d) + np.diagflat(b, 1)
        l_matrix = np.eye(3) + np.diagflat(a, -1)
        lu_matrix = np.dot(l_matrix, u_matrix)

        # Test that LU X = G.
        lux = np.dot(lu_matrix, x)
        self.assert_close(g, lux, "LU X")

    def test_invert_lu(self):
        """Test invert_lu(LU, U'L') M = -I"""
        m = self.__class__.m
        md = np.diag(m)
        mu = np.diag(m, 1)
        ml = np.diag(m, -1)
        tri_diagonal = climber.TriDiagonal(md, mu, ml)
        lu = climber.lu_decompose(tri_diagonal)
        ul = climber.ul_decompose(tri_diagonal)

        d = np.empty([3])
        ld = np.empty([2])
        climber.invert_lu(lu, ul, d, ld)

        # Test that M^-1 M = -I for the two diagonals
        expected = -np.linalg.inv(m)
        self.assert_close(np.diag(expected), d, "d")
        self.assert_close(np.diag(expected, -1), ld, "ld")


class TestClimber(unittest.TestCase):
    """Tests for the Climber class"""

    def setUp(self):
        np.seterr(all="raise")
        self.assert_close = assert_close.__get__(self, self.__class__)
        climber.Climber.wiener_variance = 1.0
        self.initial_prior = NormalDistribution(0.0, 1.0)

    def test_init(self):
        """Test Climber initializes one_on_sigma_sq and wiener_d2"""
        gaps = np.array([1.0, 2.0])
        climber.Climber.wiener_variance = 10.0
        c = climber.Climber(self.initial_prior, gaps)
        self.assert_close([0.1, 0.05], c._one_on_sigma_sq, "one_on_sigma_sq")
        self.assert_close([-0.1, -0.15, -0.05], c._wiener_d2, "wiener_d2")

    def test_get_ratings_adjustment(self):
        """Test Climber.get_ratings_adjustment"""
        gaps = np.array([1.0])
        ratings = np.array([6.0, 4.0])
        bt_d1 = np.array([0.5, 1.0])
        bt_d2 = np.array([-0.25, -0.625])
        c = climber.Climber(self.initial_prior, gaps)
        delta = c.get_ratings_adjustment(ratings, bt_d1, bt_d2)
        self.assert_close([0.50918582, -0.55155649], delta, "delta")

    def test_get_covariance(self):
        """Test Climber.get_covariance"""
        gaps = np.array([1.0])
        ratings = np.array([6.0, 4.0])
        bt_d1 = np.array([0.5, 1.0])
        bt_d2 = np.array([-0.25, -0.625])
        c = climber.Climber(self.initial_prior, gaps)
        var = np.empty(2)
        cov = np.empty(1)
        c.get_covariance(ratings, bt_d1, bt_d2, var, cov)
        self.assert_close([0.61176471, 0.84705882], var, "var")
        self.assert_close([0.37647059], cov, "cov")
