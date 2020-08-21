"""Additional assertions for use in unit tests"""

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


# Show the calling frames when assertions fail, instead of the helper function.
__unittest = True


def assert_close(test_case, expected, actual, name):
    """Raise an exception if expected does not equal actual.

    Equality is checked "approximately".

    Parameters
    ----------
    test_case : unittest.TestCase
        The test case to raise an assertion on.
    expected : list or ndarray
        The expected value.
    actual : ndarray
        The actual value.
    name : string
        The name of the value for use in the failure message.
    """
    expected = np.array(expected)
    if np.allclose(expected, actual):
        return
    msg = "expected {} = {}, got {}".format(name, expected, actual)
    raise test_case.failureException(msg)
