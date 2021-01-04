"""Unit tests for the climbing_ratings.__main__ module"""

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

import contextlib
import inspect
import os
import tempfile
import unittest
from ..__main__ import TableReader, guess_iterations


class TestTableReader(unittest.TestCase):
    """Tests for the TableReader class"""

    def setUp(self):
        with contextlib.ExitStack() as stack:
            self._tmpdir = tempfile.TemporaryDirectory()
            stack.enter_context(self._tmpdir)
            self.addCleanup(stack.pop_all().close)

    def _get_file(self):
        """Returns a temporary file with the given name"""
        name = inspect.stack()[0][3]
        return os.path.join(self._tmpdir.name, f"{name}.csv")

    def test_read_all(self):
        """Test TableReader.read() with expected columns in order"""
        with open(self._get_file(), "w") as f:
            f.write("a,b,c\n1,2.0,foo\n")

        reader = TableReader(
            [
                ("a", int, None),
                ("b", float, None),
                ("c", str, None),
            ]
        )
        data = reader.read(self._get_file())

        self.assertEqual(data[0], [1])
        self.assertEqual(data[1], [2.0])
        self.assertEqual(data[2], ["foo"])

    def test_read_reorder(self):
        """Test TableReader.read() with expected columns out-of-order"""
        with open(self._get_file(), "w") as f:
            f.write("c,b,a\nfoo,2.0,1\n")

        reader = TableReader(
            [
                ("a", int, None),
                ("b", float, None),
                ("c", str, None),
            ]
        )
        data = reader.read(self._get_file())

        self.assertEqual(data[0], [1])
        self.assertEqual(data[1], [2.0])
        self.assertEqual(data[2], ["foo"])

    def test_read_default(self):
        """Test TableReader.read() with expected columns out-of-order"""
        with open(self._get_file(), "w") as f:
            f.write("a,b\n1,2.0\n")

        reader = TableReader(
            [
                ("a", int, None),
                ("b", float, None),
                ("c", str, "foo"),
            ]
        )
        data = reader.read(self._get_file())

        self.assertEqual(data[0], [1])
        self.assertEqual(data[1], [2.0])
        self.assertEqual(data[2], ["foo"])

    def test_read_no_default(self):
        """Test TableReader.read() with expected columns out-of-order"""
        with open(self._get_file(), "w") as f:
            f.write("a,b\n1,2.0\n")

        reader = TableReader(
            [
                ("a", int, None),
                ("b", float, None),
                ("c", str, None),
            ]
        )
        with self.assertRaises(ValueError) as cm:
            data = reader.read(self._get_file())


class TestMainFunctions(unittest.TestCase):
    """Tests for functions in the __main__ module"""

    def test_guess_iterations(self):
        """Test guess_iterations()"""
        self.assertEqual(guess_iterations(1), 64)
        self.assertEqual(guess_iterations(1024), 64)
        self.assertEqual(guess_iterations(1025), 64)
        self.assertEqual(guess_iterations(1089), 66)
        self.assertEqual(guess_iterations(4096), 128)
