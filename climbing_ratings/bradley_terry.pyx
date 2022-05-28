"""Bradley-Terry model"""

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

cimport numpy as cnp
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport fabs

from cython.parallel import prange


# Workaround for "expl" being missing from Cython's libc.math.
# See: https://github.com/cython/cython/issues/3570
cdef extern from "<math.h>" nogil:
    long double coshl(long double)
    long double expl(long double)
    long double fabsl(long double)


cnp.import_array()


def expand_to_slices(double[::1] values, list slices, Py_ssize_t n):
    """Expand normalized values to contiguous blocks.

    Parameters
    ----------
    values : ndarray
        The normalized values.
    slices : list of pairs
        The (start, end) pairs corresponding to a slice in the output.  The
        implied slices must be contiguous and in ascending order.
    n : Py_ssize_t
        Size of the output array.

    Returns
    -------
    ndarray
        A member of the returned array x[i] will equal values[j] if
        slices[j][0] <= i < slices[j][1].
    """
    cdef double[::1] expanded = cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0)

    cdef Py_ssize_t j = 0
    cdef Py_ssize_t i, end
    for i, (_, end) in enumerate(slices):
        while j < end:
            expanded[j] = values[i]
            j += 1

    return expanded.base


def expand_to_slices_sparse(double[::1] values, list slices, Py_ssize_t n):
    """Expand normalized values to non-overlapping blocks.

    Parameters
    ----------
    values : ndarray
        The normalized values.
    slices : list of pairs
        The (start, end) pairs corresponding to a slice in the output.  The
        slices should not overlap.
    n : Py_ssize_t
        Size of the output array.

    Returns
    -------
    ndarray
        A member of the returned array x[i] will equal values[j] if
        slices[j][0] <= i < slices[j][1], otherwise it will equal 0.0.
    """
    cdef double[::1] expanded = cnp.PyArray_ZEROS(1, [n], cnp.NPY_DOUBLE, 0)

    cdef Py_ssize_t j
    cdef Py_ssize_t i, start, end
    cdef double v
    for i, (start, end) in enumerate(slices):
        v = values[i]
        for j in range(start, end):
            expanded[j] = v

    return expanded.base


def get_bt_derivatives(
    list slices,
    double[::1] win,
    double[::1] player,
    double[::1] adversary,
):
    """Get the derivatives of the log-likelihood for each player.

    A player is an abstraction for an entity with a rating; it will correspond
    to a page or a route.

    Releases the GIL; must be called on the main thread.

    Parameters
    ----------
    slices : list of pairs
        (start, end) indices representing slices of the ascents for each player.
    win : contiguous ndarray
        1.0 if the ascent was a "win" for the player, -1.0 otherwise, for each
        ascent.
    player : contiguous ndarray
        Natural rating of the "player" for each ascent.
    adversary : contiguous ndarray
        Natural rating of the adversary for each ascent.

    Returns
    -------
    (d1 : ndarray, d2 : ndarray)
        A pair of ndarrays of the first and second derivative of the
        Bradley-Terry log-likelihood a "player" wins, with respect to the
        "natural rating" of that player.
    """
    cdef Py_ssize_t n = player.shape[0]
    cdef long double *d1_terms = <long double *> PyMem_Malloc(2 * n * sizeof(long double))
    cdef long double *d2_terms = &d1_terms[n]

    if not (d1_terms):
        raise MemoryError()

    _cget_bt_summation_terms(win, player, adversary, n, d1_terms, d2_terms)

    cdef Py_ssize_t num_slices = len(slices)
    cdef double[::1] d1 = cnp.PyArray_EMPTY(1, [num_slices], cnp.NPY_DOUBLE, 0)
    cdef double[::1] d2 = cnp.PyArray_EMPTY(1, [num_slices], cnp.NPY_DOUBLE, 0)

    cdef Py_ssize_t start, end
    cdef int i
    cdef tuple pair
    for i, pair in enumerate(slices):
        start, end = pair
        if start == end:
            d1[i] = 0.0
            d2[i] = 0.0
            continue

        # Instead of using WHR Appendix A.1's factorization (with its A, B, C,
        # D and gamma terms), we sum the actual derivatives for each ascent,
        # which is more numerically stable.
        d1[i] = _csum(d1_terms, start, end)
        d2[i] = _csum(d2_terms, start, end)

    PyMem_Free(d1_terms)

    return (d1.base, d2.base)


cdef void _cget_bt_summation_terms(
    double[::1] win,
    double[::1] player,
    double[::1] adversary,
    Py_ssize_t n,
    long double *d1_terms,
    long double *d2_terms,
):
    """Get the Bradley-Terry derivative terms for each ascent.

    A player is an abstraction for an entity with a rating; it will correspond
    to a page or a route.

    Extended precision numbers are used to compensate for the loss of precision
    due to exponentiation.

    Parameters
    ----------
    win : contiguous ndarray
        1.0 if the ascent was a "win" for the player, -1.0 otherwise, for each
        ascent.
    player : contiguous ndarray of longdouble
        Natural rating of the "player" for each ascent.
    adversary : contiguous ndarray of longdouble
        Natural rating of the adversary for each ascent.
    n : Py_ssize_t
        Number of ascents.
    d1_terms : long double *
        Output array of the "d ln P / dr" terms for each ascent.
    d2_terms : long double *
        Output array of the "d^2 ln P / dr^2" terms for each ascent.
    """
    cdef Py_ssize_t i
    cdef long double t
    for i in prange(n, nogil=True, schedule="static"):
        # From WHR 2.2 Bradley-Terry Model:
        #
        # P(player i beats player k)
        #   = gamma_i / (gamma_i + gamma_k)
        #   = 1 / (exp(r_k - r_i) + 1)
        #
        # Or in terms of this function's parameters:
        #
        # P(player beats adversary)
        #   = 1 / (exp(adversary - player) + 1)
        #
        # Then instead of using WHR Appendix A.1's factorization (with its A,
        # B, C, D and gamma terms), we evaluate the derivatives for each ascent.
        # Hopefully these forms promote numerical stability by netting out
        # the ratings difference before exponentiation.
        #
        # d ln P / dr =
        #   win:   1 / (exp(winner - loser) + 1)
        #   loss: -1 / (exp(winner - loser) + 1)
        # d^2 ln P / dr^2 = -1 / (2 (cosh(winner - loser) + 1))
        d1_terms[i] = (win[i] /
            (expl((<long double> player[i] - adversary[i]) * win[i]) + 1.0))
        d2_terms[i] = (-0.5 /
            (coshl((<long double> player[i] - adversary[i]) * win[i]) + 1.0))


def _get_bt_summation_terms(
    double[::1] win, double[::1] player, double[::1] adversary
):
    """Wraps _cget_bt_summation_terms() for testing"""
    cdef Py_ssize_t n = player.shape[0]
    cdef long double[::1] d1 = cnp.PyArray_EMPTY(1, [n], cnp.NPY_LONGDOUBLE, 0)
    cdef long double[::1] d2 = cnp.PyArray_EMPTY(1, [n], cnp.NPY_LONGDOUBLE, 0)
    _cget_bt_summation_terms(win, player, adversary, n, &d1[0], &d2[0])
    return (d1, d2)


cdef long double _csum(const long double *x, Py_ssize_t start, Py_ssize_t end):
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


def _sum(const long double[::1] x, Py_ssize_t start, Py_ssize_t end):
    """Wraps _csum() for testing"""
    return _csum(&x[0], start, end)
