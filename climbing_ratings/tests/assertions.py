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
import unittest
from numpy.typing import ArrayLike, NDArray
from typing import Any, Callable


# Show the calling frames when assertions fail, instead of the helper function.
__unittest = True


def assert_close(
    test_case: unittest.TestCase,
    expected: ArrayLike,
    actual: NDArray[np.floating[Any]],
    name: str,
) -> None:
    """Raise an exception if expected does not equal actual.

    Equality is checked "approximately".

    Parameters
    ----------
    test_case
        The test case to raise an assertion on.
    expected
        The expected value.
    actual
        The actual value.
    name
        The name of the value for use in the failure message.
    """
    expected = np.array(expected)
    if expected.shape == actual.shape and np.allclose(expected, actual):
        return
    msg = "expected {} = {}, got {}".format(name, expected, actual)
    raise test_case.failureException(msg)


def assert_close_get(
    test_case: unittest.TestCase, owner: type
) -> Callable[[ArrayLike, NDArray[np.floating[Any]], str], None]:
    """Returns assert_close bound to a particular test case.

    Parameters
    ----------
    test_case
        The instance to bind to.
    owner
        The owner class.

    Returns
    -------
        A callable for assert_close that is bound to the given test case.
    """
    # This is a workaround for the mypy error
    # 'Callable... has no attribute "__get__"'', because mypy can't distinguish
    # between functions (which are guaranteed to have a __get__ method, see
    # https://docs.python.org/3.8/howto/descriptor.html#functions-and-methods)
    # and other callables (like C functions).
    return assert_close.__get__(test_case, owner)  # type: ignore
