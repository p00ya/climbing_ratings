"""Tests for the csum module"""

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
from ..csum import _csum


_Array = NDArray[np.float64]


class TestCsumFunctions(unittest.TestCase):
    """Tests for functions in the csum module"""

    def setUp(self) -> None:
        np.seterr(all="raise")

    def test_csum(self) -> None:
        """Test csum()"""
        x: NDArray[np.longdouble] = np.array([1.0, 2.0, 4.0, 8.0], dtype="longdouble")
        self.assertEqual(15.0, _csum(x, 0, 4))
        self.assertEqual(0.0, _csum(x, 0, 0))
        self.assertEqual(6.0, _csum(x, 1, 3))
        self.assertEqual(7.0, _csum(x, 0, 3))

    def test_csum_error(self) -> None:
        """Test csum() error compensation with extended precision"""
        # These values should detect uncompensated error for both double
        # and extended precision floats.
        x: NDArray[np.longdouble] = np.full([10], 0.1, dtype="longdouble")
        self.assertEqual(1.0, _csum(x, 0, 10))
        x = np.array([1e100, -1.0, -1e100, 1.0], dtype="longdouble")
        self.assertEqual(0.0, _csum(x, 0, 4))
        x = np.array([1e100, 1.0, -1e100, 1.0], dtype="longdouble")
        self.assertEqual(2.0, _csum(x, 0, 4))
