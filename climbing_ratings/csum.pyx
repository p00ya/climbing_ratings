"""Error-compensated sum"""

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

from libc.math cimport fabsl


cdef long double csum(
    const long double *x,
    Py_ssize_t start,
    Py_ssize_t end
) noexcept nogil:
    """Compute the sum of x[start:end], with error compensation."""
    # Neumaier's improved Kahan–Babuška summation algorithm.  To be effective,
    # this must be compiled with clang's "-fno-associative-math" or equivalent.
    cdef long double s = 0.0
    cdef Py_ssize_t i

    cdef long double c = 0.0
    cdef long double t
    for i in range(start, end):
        t = s + x[i]
        if fabsl(s) >= fabsl(x[i]):
            c += (s - t) + x[i]
        else:
            c += (x[i] - t) + s

        s = t
    return s + c


def _csum(const long double[::1] x, Py_ssize_t start, Py_ssize_t end):
    """Wraps csum() for testing"""
    return csum(&x[0], start, end)
