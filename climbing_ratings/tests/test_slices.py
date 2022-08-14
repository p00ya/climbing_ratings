"""Tests for the slices module"""

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
from .. import slices


class TestSlices(unittest.TestCase):
    """Tests for the Slices class"""

    def test_len(self) -> None:
        """Test len()"""
        x = slices.Slices([(0, 2)])
        self.assertEqual(1, len(x))

    def test_getitem(self) -> None:
        """Test getitem()"""
        x = slices.Slices([(0, 2)])
        self.assertEqual((0, 2), x[0])

    def test_iter(self) -> None:
        """Test iter()"""
        x = slices.Slices([(0, 2), (2, 5)])
        i = iter(x)
        self.assertEqual((0, 2), next(i))
        self.assertEqual((2, 5), next(i))
        with self.assertRaises(StopIteration):
            next(i)
