#!/usr/bin/env python3

"""Goldens test for 02-run_estimation.py"""

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
import csv
import os.path
import shutil
import subprocess
import sys
import tempfile
import unittest


class TestRunEstimation(unittest.TestCase):
    """Goldens test for 02-run_estimation.py"""

    def setUp(self):
        with contextlib.ExitStack() as stack:
            self._tmpdir = tempfile.TemporaryDirectory()
            stack.enter_context(self._tmpdir)
            self.addCleanup(stack.pop_all().close)
        shutil.copy(self.get_golden("ascents.csv"), self._tmpdir.name)
        shutil.copy(self.get_golden("pages.csv"), self._tmpdir.name)
        shutil.copy(self.get_golden("routes.csv"), self._tmpdir.name)

    def get_golden(self, filename):
        """Get the path to the golden copy of the given file."""
        return os.path.join("tests", "testdata", filename)

    def assert_matches_golden(self, filename):
        """Raise an exception if the golden and generated CSVs do not match.

        Assumes the first row and first column can be compared as strings,
        while all other cells are compared approximately as floating points.
        """
        try:
            fp_actual = open(
                os.path.join(self._tmpdir.name, filename), newline=""
            )
            fp = open(self.get_golden(filename), newline="")
            reader_actual = iter(csv.reader(fp_actual))
            reader = iter(csv.reader(fp))

            # Check header rows.
            self.assertEqual(next(reader_actual), next(reader))

            for line, goldens in enumerate(reader):
                actuals = next(reader_actual)

                for i, _ in enumerate(goldens):
                    msg = "did not match golden at %s:%d" % (filename, line + 1)
                    if i == 0:
                        self.assertEqual(actuals[i], goldens[i], msg)
                    else:
                        # Tolerate small differences in floating point values.
                        self.assertAlmostEqual(
                            float(actuals[i]), float(goldens[i]),
                            msg=msg
                        )

        finally:
            fp.close()
            fp_actual.close()

    def test_run_estimation(self):
        """Test run_estimation"""
        cmd = [sys.executable, "02-run_estimation.py", self._tmpdir.name]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        self.assert_matches_golden("page_ratings.csv")
        self.assert_matches_golden("route_ratings.csv")


if __name__ == "__main__":
    unittest.main()
