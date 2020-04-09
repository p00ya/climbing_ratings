"""Cython helpers for the climber module.

WHR uses some specialized matrix algorithms that are not easily vectorizable.
These arise in relation to the pages of a climber.

By compiling these algorithms via Cython, the high overhead of single-element
access to numpy arrays is avoided.
"""

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

import collections
cimport numpy as cnp
from libc.math cimport log

cnp.import_array()


class TriDiagonal(collections.namedtuple("TriDiagonal", ["d", "u", "l"])):
    """Stores a tri-diagonal matrix.

    We decompose the matrix into three arrays:
    - d (main diagonal)
    - u (upper sub-diagonal)
    - l (lower sub-diagonal)

    or in matrix notation:

         | d0 u0 ..  0 |
     M = | l0 d1 `.  : |
         |  : `. `. um |
         |  0 .. lm dn |
    """


class TriDiagonalLU(collections.namedtuple("TriDiagonalLU", ["d", "b", "a"])):
    """Stores the LU-decomposition of a tri-diagonal matrix.

    We decompose the LU matrices into 3 arrays:
    - d (main diagonal of U matrix)
    - b (upper sub-diagonal of U matrix)
    - a (lower sub-diagonal of L matrix)

             |  1  0  0 .. 0 | | d0 b0  0 ..  0 |
             | a0  1  0 .. 0 | |  0 d1 b1 ..  0 |
    M = LU = |  0 a1  1  . 0 | |  0  0 d2 `.  : |
             |  :  . `. `. : | |  :  .  . `. bm |
             |  0  0  . am 1 | |  0  0  0 .. dn |

    Similarly, for UL decompositions:
    - d (main diagonal of L' matrix)
    - b (lower sub-diagonal of L' matrix)
    - a (upper sub-diagonal of U' matrix)

               |  1 a0  0 ..  0 | | d0  0  0 ..  0 |
               |  0  1 a1 ..  0 | | b0 d1  0 ..  0 |
    M = U'L' = |  0  0  1 `.  0 | |  0 b1 d2 ..  0 |
               |  :  .  . `. am | |  :  . `. `.  : |
               |  0  0  0 ..  1 | |  0  0  0 bm dn |

    Note the indices are always 0-based (in WHR the indices are the 1-based row
    number).
    """


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
        The natural rating for each page.
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
        d = ratings[i + 1] - ratings[i]
        d *= one_on_sigma_sq[i]
        gradient[i] += d
        gradient[i + 1] -= d


def lu_decompose(object tri_diagonal):
    """Decompose a tri-diagonal matrix into LU form.

    Parameters
    ----------
    tri_diagonal : TriDiagonal
        Represents the matrix to decompose.
    """
    cdef double[::1] hd, hu, hl, d, b, a
    hd, hu, hl = tri_diagonal

    cdef Py_ssize_t n = hd.shape[0]
    d = cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0)
    a = cnp.PyArray_EMPTY(1, [n - 1], cnp.NPY_DOUBLE, 0)

    # WHR Appendix B: perform LU decomposition
    b = hu
    d[0] = hd[0]

    cdef Py_ssize_t i
    cdef double t
    for i in range(1, n):
        t = hl[i - 1] / d[i - 1]
        a[i - 1] = t
        d[i] = hd[i] - hu[i - 1] * t

    return TriDiagonalLU(d.base, b.base, a.base)


def ul_decompose(object tri_diagonal):
    """Decompose a tri-diagonal matrix into U'L' form.

    Parameters
    ----------
    tri_diagonal : TriDiagonal
        Represents the matrix to decompose.
    """
    cdef double[::1] hd, hu, hl, d, b, a
    hd, hu, hl = tri_diagonal

    cdef Py_ssize_t n = hd.shape[0]
    d = cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0)
    a = cnp.PyArray_EMPTY(1, [n - 1], cnp.NPY_DOUBLE, 0)

    # WHR Appendix B.2: Computing diagonal and sub-diagonal terms of H^-1
    b = hl
    d[n - 1] = hd[n - 1]

    cdef Py_ssize_t i
    cdef double t
    for i in range(n - 2, -1, -1):
        t = hu[i] / d[i + 1]
        a[i] = t
        d[i] = hd[i] - b[i] * t

    return TriDiagonalLU(d.base, b.base, a.base)


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
