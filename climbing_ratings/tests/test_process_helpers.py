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

import numpy as np
from numpy.typing import NDArray
import unittest
from .. import process_helpers
from .assertions import assert_close_get


_Array = NDArray[np.float_]


class TestProcessHelpers(unittest.TestCase):
    """Tests for the process_helpers extension."""

    # fmt: off
    m: _Array = np.array([
        1.0, -2.0,  0.0,
        0.5,  2.0,  1.0,
        0.0,  3.0, 11.0,
    ]).reshape((3, 3))
    # fmt: on

    def setUp(self) -> None:
        np.seterr(all="raise")
        self.assert_close = assert_close_get(self, self.__class__)

    def test_lu_decompose(self) -> None:
        """Test that for (L, U) = lu_decompose(M), LU = M"""
        m = self.__class__.m
        md: _Array = np.diag(m).copy()
        mu: _Array = np.diag(m, 1).copy()
        ml: _Array = np.diag(m, -1).copy()
        tri_diagonal = process_helpers.TriDiagonal(md, mu, ml)
        lu = process_helpers.lu_decompose(tri_diagonal)

        # Reconstruct the L and U matrices.
        u_matrix = np.diagflat(lu.d) + np.diagflat(lu.b, 1)
        l_matrix = np.eye(3) + np.diagflat(lu.a, -1)
        lu_matrix = np.dot(l_matrix, u_matrix)
        # Test that M = LU.
        self.assert_close(m, lu_matrix, "M")

    def test_ul_decompose(self) -> None:
        """Test that for (U', L') = ul_decompose(M), U'L' = M"""
        m = self.__class__.m
        md: _Array = np.diag(m).copy()
        mu: _Array = np.diag(m, 1).copy()
        ml: _Array = np.diag(m, -1).copy()
        tri_diagonal = process_helpers.TriDiagonal(md, mu, ml)
        ul = process_helpers.ul_decompose(tri_diagonal)

        # Reconstruct the L and U matrices.
        u_matrix = np.eye(3) + np.diagflat(ul.a, 1)
        l_matrix = np.diagflat(ul.d) + np.diagflat(ul.b, -1)
        ul_matrix = np.dot(u_matrix, l_matrix)
        # Test that M = U'L'
        self.assert_close(m, ul_matrix, "M")

    def test_add_wiener_gradient(self) -> None:
        """Test add_wiener_gradient()"""
        one_on_sigma_sq: _Array = np.array([1.0])
        ratings: _Array = np.array([1.0, 2.0])
        gradient = np.zeros(2)
        process_helpers.add_wiener_gradient(one_on_sigma_sq, ratings, gradient)
        self.assert_close([1.0, -1.0], gradient, "gradient")

    def test_solve_y(self) -> None:
        """Test solve_y()"""
        g: _Array = np.array([10.0, 5.0, 32.0])
        a: _Array = np.array([0.0, -0.1, 2.0])
        process_helpers.solve_y(g, a)
        y = a  # output parameter
        self.assert_close([10.0, 4.0, 40.0], y, "y")

    def test_solve_x(self) -> None:
        """Test solve_x()"""
        b: _Array = np.array([-2, 1.0])
        d: _Array = np.array([1.0, 3.0, 10.0])
        y: _Array = np.array([10.0, 4.0, 40.0])
        process_helpers.solve_x(b, d, y)
        x = y  # output parameter
        self.assert_close([10.0, 0.0, 4.0], x, "x")
