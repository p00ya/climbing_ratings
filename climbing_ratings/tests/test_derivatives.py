"""Tests for the process_helpers extension"""

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

import copy
import numpy as np
from numpy.typing import NDArray
import unittest
from .. import derivatives
from ..slices import Slices
from .assertions import assert_close_get


_Array = NDArray[np.float64]


class TestDerivativesFunctions(unittest.TestCase):
    """Tests for the matrix functions in the derivatives extension"""

    # fmt: off
    m: _Array = np.array([
        1.0, -2.0,  0.0,
        0.5,  2.0,  1.0,
        0.0,  3.0, 11.0,
    ]).reshape((3, 3))
    # fmt: on

    @staticmethod
    def tri_diagonal(m: _Array) -> derivatives.TriDiagonal:
        """Returns m as a TriDiagonal"""
        h = derivatives.TriDiagonal(np.empty(3), np.empty(3), np.empty(3))
        hd, hu, hl = h.as_tuple()
        np.copyto(hd, np.diag(m))
        np.copyto(hu[:-1], np.diag(m, 1))
        np.copyto(hl[:-1], np.diag(m, -1))
        return h

    def setUp(self) -> None:
        np.seterr(all="raise")
        self.assert_close = assert_close_get(self, self.__class__)

    def test_lu_decompose(self) -> None:
        """Test that for (L, U) = lu_decompose(M), LU = M"""
        m = self.__class__.m
        h = self.__class__.tri_diagonal(m)
        lu = derivatives.TriDiagonalLU(np.empty(3), np.empty(3), np.empty(3))

        derivatives.lu_decompose(h, lu, 0, 3)

        # Reconstruct the L and U matrices.
        d, b, a = lu.as_tuple()
        u_matrix = np.diagflat(d) + np.diagflat(b[:-1], 1)
        l_matrix = np.eye(3) + np.diagflat(a[:-1], -1)
        lu_matrix = np.dot(l_matrix, u_matrix)
        # Test that M = LU.
        self.assert_close(m, lu_matrix, "LU")

    def test_ul_decompose(self) -> None:
        """Test that for (U', L') = ul_decompose(M), U'L' = M"""
        m = self.__class__.m
        h = self.__class__.tri_diagonal(m)
        ul = derivatives.TriDiagonalLU(np.empty(3), np.empty(3), np.empty(3))

        derivatives.ul_decompose(h, ul, 0, 3)

        # Reconstruct the L and U matrices.
        d, b, a = ul.as_tuple()
        u_matrix = np.eye(3) + np.diagflat(a[:-1], 1)
        l_matrix = np.diagflat(d) + np.diagflat(b[:-1], -1)
        ul_matrix = np.dot(u_matrix, l_matrix)
        # Test that M = U'L'
        self.assert_close(m, ul_matrix, "U'L'")

    def test_solve_y(self) -> None:
        """Test that for Y = solve_y(G, L), LY = G"""
        g: _Array = np.array([10.0, 5.0, 32.0])
        a: _Array = np.array([0.1, -2.0])
        y = np.empty([3])
        derivatives.solve_y(g, a, y, 0, 3)
        # Test that LY = G.
        l_matrix = np.eye(3) + np.diagflat(a, -1)
        ly = np.dot(l_matrix, y)
        self.assert_close(g, ly, "LY")
        self.assert_close([10.0, 4.0, 40.0], y, "y")

    def test_solve_x(self) -> None:
        """Test solve_x()"""
        b: _Array = np.array([-2, 1.0])
        d: _Array = np.array([1.0, 3.0, 10.0])
        y: _Array = np.array([10.0, 4.0, 40.0])
        x = y.copy()
        derivatives.solve_x(b, d, x, 0, 3)
        # Test that UX = Y.
        u_matrix = np.diagflat(d) + np.diagflat(b, 1)
        ux = np.dot(u_matrix, x)
        self.assert_close(y, ux, "UX")
        self.assert_close([10.0, 0.0, 4.0], x, "x")

    def test_invert_lu_dot_g(self) -> None:
        """Test that for X = invert_lu_dot_g(LU, G), LU X = G"""
        g: _Array = np.array([10.0, 5.0, 32.0])
        d: _Array = np.array([1.0, 3.0, 10.0])
        b: _Array = np.array([-2.0, 1.0, 0.0])
        a: _Array = np.array([0.1, -2.0, 0.0])
        lu = derivatives.TriDiagonalLU(d, b, a)
        x: _Array = np.empty(3)
        derivatives.invert_lu_dot_g(lu, g, x, 0, 3)

        u_matrix = np.diagflat(d) + np.diagflat(b[:-1], 1)
        l_matrix = np.eye(3) + np.diagflat(a[:-1], -1)
        lu_matrix = np.dot(l_matrix, u_matrix)

        # Test that LU X = G.
        lux = np.dot(lu_matrix, x)
        self.assert_close(g, lux, "LU X")

    def test_invert_lu(self) -> None:
        """Test invert_lu(LU, U'L') M = -I"""
        m: _Array = self.__class__.m
        lu = derivatives.TriDiagonalLU(
            np.array([1.0, 3.0, 10.0]),
            np.array([-2, 1.0, 0.0]),
            np.array([0.5, 1.0, 0.0]),
        )
        ul = derivatives.TriDiagonalLU(
            np.array([30.0 / 19.0, 19.0 / 11.0, 11.0]),
            np.array([0.5, 3.0, 0.0]),
            np.array([-22.0 / 19.0, 1.0 / 11.0, 0.0]),
        )

        d = np.empty([3])
        ld = np.empty([3])
        derivatives.invert_lu(lu, ul, d, ld, 0, 3)

        # Test that M^-1 M = -I for the two diagonals
        expected = -np.linalg.inv(m)
        self.assert_close(np.diag(expected), d, "d")
        self.assert_close(np.diag(expected, -1), ld[:-1], "ld")


class TestNormalDistribution(unittest.TestCase):
    """Tests for the NormalDistribution class"""

    def setUp(self) -> None:
        np.seterr(all="raise")
        self.assert_close = assert_close_get(self, self.__class__)

    def test_var_1(self) -> None:
        """Test NormalDistribution(0, 1)"""
        dist = derivatives.NormalDistribution(0.0, 1.0)
        d1 = np.zeros(3)
        self.assertEqual(0.0, dist.gradient(0.0), "gradient")
        self.assertEqual(1.0, dist.gradient(-1.0), "gradient")
        self.assertEqual(2.0, dist.gradient(-2.0), "gradient")
        self.assertEqual(-1.0, dist.d2(), "d2")

    def test_var_2(self) -> None:
        """Test NormalDistribution(0, 2)"""
        dist = derivatives.NormalDistribution(0.0, 2.0)
        d1 = np.zeros(3)
        self.assertEqual(0.0, dist.gradient(0.0), "gradient")
        self.assertEqual(-0.5, dist.gradient(1.0), "gradient")
        self.assertEqual(-1.0, dist.gradient(2.0), "gradient")
        self.assertEqual(-0.5, dist.d2(), "d2")


class TestMultiNormalDistribution(unittest.TestCase):
    """Tests for the NormalDistribution class"""

    def setUp(self) -> None:
        np.seterr(all="raise")
        self.assert_close = assert_close_get(self, self.__class__)

    def test_var_1(self) -> None:
        """Test MultiNormalDistribution with variance=1"""
        mu = np.array([0.0, 1.0, 2.0])
        dist = derivatives.MultiNormalDistribution(mu, 1.0)
        d1 = np.zeros(3)
        dist.add_gradient(np.array([0.0, 0.0, 0.0]), d1)
        self.assert_close([0.0, 1.0, 2.0], d1, "d1")
        self.assertEqual(-1.0, dist.d2(), "d2")


class TestWienerProcess(unittest.TestCase):
    """Tests for the WienerProcess class"""

    def setUp(self) -> None:
        np.seterr(all="raise")
        self.assert_close = assert_close_get(self, self.__class__)

    def test_init(self) -> None:
        """Test WienerProcess initializes one_on_sigma_sq and d2"""
        gaps: _Array = np.array([1.0, 0.0])
        slices = Slices([(0, 2)])
        w = derivatives.WienerProcess(gaps, slices, 10.0)
        one_on_sigma_sq, d2 = w.as_tuple()
        self.assert_close([0.1, 0.0], one_on_sigma_sq, "one_on_sigma_sq")
        self.assert_close([-0.1, -0.1], d2, "d2")

    def test_init_two_slices(self) -> None:
        """Test WienerProcess initialization with 2 slices"""
        gaps: _Array = np.array([1.0, 0.0, 2.0, 0.0])
        slices = Slices([(0, 2), (2, 4)])
        w = derivatives.WienerProcess(gaps, slices, 10.0)
        one_on_sigma_sq, d2 = w.as_tuple()
        self.assert_close(
            [0.1, 0.0, 0.05, 0.0],
            one_on_sigma_sq,
            "one_on_sigma_sq",
        )
        self.assert_close([-0.1, -0.1, -0.05, -0.05], d2, "d2")

    def test_add_gradient(self) -> None:
        """Test add_gradient()"""
        gaps: _Array = np.array([1.0, 0.0])
        slices = Slices([(0, 2)])
        w = derivatives.WienerProcess(gaps, slices, 1.0)

        gradient = np.zeros(2)
        ratings: _Array = np.array([1.0, 2.0])
        w._add_gradient(ratings, gradient)
        self.assert_close([1.0, -1.0], gradient, "gradient")


class TestPageInvariants(unittest.TestCase):
    """Tests for the PageInvariants class"""

    def setUp(self) -> None:
        np.seterr(all="raise")

    def test_init(self) -> None:
        """Test len(model)"""
        initial = derivatives.NormalDistribution(0.0, 1.0)
        gaps: _Array = np.array([1.0, 1.0])
        slices = Slices([(0, 2)])
        w = derivatives.WienerProcess(gaps, slices, 1.0)
        model = derivatives.PageInvariants(initial, w, slices)


class TestPageModel(unittest.TestCase):
    """Tests for the PageModel class"""

    def setUp(self) -> None:
        np.seterr(all="raise")
        self.assert_close = assert_close_get(self, self.__class__)

        initial = derivatives.NormalDistribution(0.0, 1.0)
        gaps: _Array = np.array([1.0, 0.0])
        slices = Slices([(0, 2)])
        w = derivatives.WienerProcess(gaps, slices, 1.0)
        self.invariants = derivatives.PageInvariants(initial, w, slices)

    def test_update_ratings(self) -> None:
        """Test update_ratings"""
        # TODO: set ratings inside model
        ratings = np.log([6.0, 4.0])
        bt_d1: _Array = np.array([0.5, 1.0])
        bt_d2: _Array = np.array([-0.25, -0.625])
        model = derivatives.PageModel(self.invariants, ratings)
        model.update_ratings(bt_d1, bt_d2)
        self.assert_close(
            [0.50918582, -0.55155649],
            ratings - model.ratings,
            "ratings delta",
        )

    def test_update_covariance(self) -> None:
        """Test get_covariance"""
        ratings = np.log([6.0, 4.0])
        bt_d1: _Array = np.array([0.5, 1.0])
        bt_d2: _Array = np.array([-0.25, -0.625])
        model = derivatives.PageModel(self.invariants, ratings)
        var, cov = model.update_covariance(bt_d1, bt_d2)
        self.assert_close([0.61176471, 0.84705882], model.var, "var")
        self.assert_close([0.37647059, 0.0], model.cov, "cov")

    def test_copy(self) -> None:
        ratings = np.log([6.0, 4.0])
        model = derivatives.PageModel(self.invariants, ratings)
        model_copy = copy.copy(model)
        self.assertIsNot(model.ratings, model_copy.ratings)
        self.assertIsNot(model.var, model_copy.var, "var")
        self.assertIsNot(model.cov, model_copy.cov, "cov")
        self.assert_close(model.ratings, model_copy.ratings, "ratings")


class TestPageModelMultipleProcesses(unittest.TestCase):
    """Tests for the PageModel class with multiple processes"""

    def setUp(self) -> None:
        np.seterr(all="raise")
        self.assert_close = assert_close_get(self, self.__class__)

        initial = derivatives.NormalDistribution(0.0, 1.0)
        gaps: _Array = np.array([1.0, 0.0, 1.0, 0.0])
        slices = Slices([(0, 2), (2, 4)])
        w = derivatives.WienerProcess(gaps, slices, 1.0)
        self.invariants = derivatives.PageInvariants(initial, w, slices)

    def test_update_ratings(self) -> None:
        """Test update_ratings"""
        ratings = np.log([6.0, 4.0, 6.0, 4.0])
        bt_d1: _Array = np.array([0.5, 1.0, 0.5, 1.0])
        bt_d2: _Array = np.array([-0.25, -0.625, -0.25, -0.625])
        model = derivatives.PageModel(self.invariants, ratings)
        model.update_ratings(bt_d1, bt_d2)
        self.assert_close(
            [0.50918582, -0.55155649, 0.50918582, -0.55155649],
            ratings - model.ratings,
            "ratings delta",
        )

    def test_update_covariance(self) -> None:
        """Test get_covariance"""
        ratings = np.log([6.0, 4.0, 6.0, 4.0])
        bt_d1: _Array = np.array([0.5, 1.0, 0.5, 1.0])
        bt_d2: _Array = np.array([-0.25, -0.625, -0.25, -0.625])
        model = derivatives.PageModel(self.invariants, ratings)
        var, cov = model.update_covariance(bt_d1, bt_d2)
        self.assert_close(
            [0.61176471, 0.84705882, 0.61176471, 0.84705882],
            var,
            "var",
        )
        self.assert_close([0.37647059, 0.0, 0.37647059, 0.0], cov, "cov")


class TestPageModelSimple(unittest.TestCase):
    """Tests for the PageModel class with "simple" data.

    Uses the same inputs as the first climber from the data in
    tests/testdata/simple.  These inputs aren't actually that "simple"
    compared to the trivial cases in the other unit tests, but they're simple
    compared to real data.
    """

    def setUp(self) -> None:
        np.seterr(all="raise")
        self.assert_close = assert_close_get(self, self.__class__)

        initial = derivatives.NormalDistribution(0.0, 4.0)
        gaps: _Array = np.array([58060800.0, 0.0])
        slices = Slices([(0, 2)])
        w = derivatives.WienerProcess(gaps, slices, 1.0 / 86400.0 / 364.0)
        self.invariants = derivatives.PageInvariants(initial, w, slices)

    def test_update_derivatives(self) -> None:
        """Test _update_derivatives"""
        ratings: _Array = np.zeros(2)
        bt_d1: _Array = np.array([1.5, 0.0])
        bt_d2: _Array = np.array([-0.75, -0.5])
        model = derivatives.PageModel(self.invariants, ratings)
        model._update_derivatives(bt_d1, bt_d2)
        self.assert_close([1.5, 0.0], model._gradient, "gradient")
        hd, hu, hl = model._hessian
        self.assert_close([0.54166667, 0.0], hu, "Hessian upper sub-diagonal")
        self.assert_close([0.54166667, 0.0], hl, "Hessian lower sub-diagonal")
        self.assert_close([-1.54166667, -1.04166667], hd, "Hessian diagonal")

    def test_update_ratings(self) -> None:
        """Test update_ratings"""
        ratings: _Array = np.zeros(2)
        bt_d1: _Array = np.array([1.5, 0.0])
        bt_d2: _Array = np.array([-0.75, -0.5])
        model = derivatives.PageModel(self.invariants, ratings)
        model.update_ratings(bt_d1, bt_d2)
        self.assert_close([1.19047619, 0.61904762], model.ratings, "ratings")
