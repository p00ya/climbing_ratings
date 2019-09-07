"""Cython helpers for the climber module.

WHR uses some specialized matrix algorithms that are not easily vectorizable.
These arise in relation to the pages of a climber.

By compiling these algorithms via Cython, the high overhead of single-element
access to numpy arrays is avoided.
"""

# Copyright 2019 Dean Scarff
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

from libc.math cimport log


def add_wiener_gradient(
    double[::1] one_on_sigma_sq,
    double[::1] ratings,
    double[::1] gradient,
):
    """Add terms from the Wiener prior to the gradient.

    Parameters
    ----------
    one_on_sigma_sq : contiguous ndarray with length N - 1
        Reciprocal of the Wiener variance between each page and the next page.
    ratings : contiguous ndarray with length N
        The rating (gamma) for each page.
    gradient : contiguous ndarray with length N
        Output array for the first derivative of the log likelihood with
        respect to the rating for each page.
    """
    # WHR Appendix A.2 Terms of the Wiener prior:
    # d ln p / d r[t] = -(r[t] - r[t+1]) / sigma[t]^2
    cdef Py_ssize_t end = one_on_sigma_sq.shape[0]

    cdef double d
    cdef Py_ssize_t i
    for i in range(end):
        # -(r[t] - r[t+1]) = r[t+1] - r[t]
        # = log(gamma[t+1]) - log(gamma[t])
        # = log(gamma[t+1] / gamma[t])

        d = log(ratings[i + 1] / ratings[i])
        d *= one_on_sigma_sq[i]
        gradient[i] += d
        gradient[i + 1] -= d


def solve_lu_d(double[::1] c, double[::1] hd):
    """Compute the U-diagonal in the LU decomposition of a tri-diagonal matrix.

    Computes the array "d" following the recurrence:
    d[i] = hd[i] - c[i] / d[i-1]

    Parameters
    ----------
    c : contiguous ndarray with length N
        The "c" term from the recurrence.
    hd : contiguous ndarray with length N
        The input is used as the "hd" term from the recurrence.  Also used as
        the output array for the computed "d" terms.
    """
    cdef Py_ssize_t n = c.shape[0]

    cdef double d_prev = 1.0
    cdef double t
    for i in range(n):
        t = c[i]
        t /= d_prev
        t += hd[i]
        hd[i] = t
        d_prev = t


def solve_ul_d(double[::1] c, double[::1] hd):
    """Compute the L'-diagonal in the UL decomposition of a tri-diagonal matrix.

    Computes the array "d" following the recurrence:
    d[i] = hd[i] - c[i] / d[i+1]

    Parameters
    ----------
    c : contiguous ndarray with length N
        The "c" term from the recurrence.
    hd : contiguous ndarray with length N
        The input is used as the "hd" term from the recurrence.  Also used as
        the output array for the computed "d" terms.
    """
    cdef Py_ssize_t end = c.shape[0] - 1

    cdef double d_next = hd[end]
    cdef double t
    for i in range(end, -1, -1):
        t = c[i]
        t /= d_next
        t += hd[i]
        hd[i] = t
        d_next = t


def solve_y(double[::1] g, double[::1] a):
    """Compute the vector Y in LY = G, where L is from the LU decomposition.

    Computes the array "y" following the recurrence:
    y[i] = g[i] + a[i] y[i - 1]

    See WHR Appendix B.1.

    Parameters
    ----------
    g : contiguous ndarray with length N
        The input is used as the "g" term from the recurrence.
    a : contiguous ndarray with length N
        The "a" term from the recurrence.  Note this not just the "a" array
        from the LU decomposition: it should be padded and negated.
        Also used as the output array for the computed "y" terms.
    """
    cdef Py_ssize_t n = g.shape[0]

    cdef double y_prev = 0.0
    cdef double t
    for i in range(n):
        t = a[i]
        t *= y_prev
        t += g[i]
        a[i] = t
        y_prev = t


def solve_x(double[::1] b, double[::1] d, double[::1] y):
    """Compute the vector X in UX = Y, where U is from the LU decomposition.

    Computes the array "x" following the recurrence:
    x[i] = (y[i] - b[i] x[i+1]) / d[i]

    See WHR Appendix B.1.

    Parameters
    ----------
    b : contiguous ndarray with length N - 1
        The "b" term from the recurrence.
    d : contiguous ndarray with length N
        The input is used as the "d" term from the recurrence.
    y : contiguous ndarray with length N
        The "y" term from the recurrence.  Also used as
        the output array for the computed "x" terms.
    """
    cdef Py_ssize_t end = y.shape[0] - 1

    cdef double x_next = y[end] / d[end]
    y[end] = x_next

    cdef double t
    # Note: gcc is known to be bad at auto-vectorizing down-loops.
    # https://stackoverflow.com/questions/7919304#36772982
    for i in range(end - 1, -1, -1):
        t = x_next
        t *= b[i]
        t *= -1.0
        t += y[i]
        t /= d[i]
        y[i] = t
        x_next = t
