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


class TestClimbingRatings(unittest.TestCase):
    """Goldens test for climbing_ratings' main script"""

    def copy_inputdata(self, testdata: str) -> str:
        with contextlib.ExitStack() as stack:
            ctx = tempfile.TemporaryDirectory()
            stack.enter_context(ctx)
            self.addCleanup(stack.pop_all().close)
        tmpdir = ctx.name
        shutil.copy(self.get_golden(testdata, "ascents.csv"), tmpdir)
        shutil.copy(self.get_golden(testdata, "pages.csv"), tmpdir)
        shutil.copy(self.get_golden(testdata, "style_pages.csv"), tmpdir)
        shutil.copy(self.get_golden(testdata, "routes.csv"), tmpdir)
        return tmpdir

    def get_golden(self, testdata: str, filename: str) -> str:
        """Get the path to the golden copy of the given file."""
        return os.path.join(testdata, filename)

    def assert_matches_golden(self, tmpdir: str, testdata: str, filename: str) -> None:
        """Raise an exception if the golden and generated CSVs do not match.

        Assumes the first row and first column can be compared as strings,
        while all other cells are compared approximately as floating points.
        """
        try:
            fp_actual = open(os.path.join(tmpdir, filename), newline="")
            fp = open(self.get_golden(testdata, filename), newline="")
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
                            float(actuals[i]), float(goldens[i]), msg=msg
                        )

        finally:
            fp.close()
            fp_actual.close()

    def test_main(self) -> None:
        """Test climbing_ratings"""
        testdata = os.path.join("tests", "testdata")
        tmpdir = self.copy_inputdata(testdata)
        cmd = [sys.executable, "-m", "climbing_ratings", tmpdir]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        self.assert_matches_golden(tmpdir, testdata, "page_ratings.csv")
        self.assert_matches_golden(tmpdir, testdata, "route_ratings.csv")
        self.assert_matches_golden(tmpdir, testdata, "style_page_ratings.csv")

    def test_main_v3(self) -> None:
        """Test climbing_ratings backwards-compatibility with v3 data"""
        v3_goldens = os.path.join("tests", "testdata", "v3.0")
        cmd = [
            sys.executable,
            "-m",
            "climbing_ratings",
            "-n",
            "--max-iterations",
            "1",
            v3_goldens,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    def test_run_estimation(self) -> None:
        """Test legacy 02-run_estimation.py stub"""
        cmd = [
            sys.executable,
            "02-run_estimation.py",
            "-n",
            "--max-iterations",
            "1",
            os.path.join("tests", "testdata"),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
