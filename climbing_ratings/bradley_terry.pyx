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
from libc.math cimport fabs

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


cdef tuple cget_bt_summation_terms(double[::1] gamma, double[::1] aux_gamma,
                                   double[::1] adversary_gamma):
    """Get the Bradley-Terry summation terms for each player.

    A player is an abstraction for an entity with a rating; it will correspond
    to a page or a route.

    Parameters
    ----------
    gamma : contiguous ndarray
        Rating of the "player" for each ascent.
    aux_gamma : contiguous ndarray
        Auxiliary coefficient to the player's rating for each ascent.
    adversary_gamma : contiguous ndarray
        Rating of the adversary for each ascent.

    Returns
    -------
    (d1_terms : MemoryView, d2_terms : MemoryView)
        d1_terms contains the "gamma / (gamma + adversary_gamma)" terms for each
        player.

        d2_terms contains the
        "gamma adversary_gamma / (gamma + adversary_gamma)^2" terms for each
        player.
    """
    # WHR Appendix A.1 Terms of the Bradley-Terry model:
    #
    # P(G_j) = (A_ij gamma_i + B_ij) / (C_ij gamma_i + D_ij)
    #
    # WHR 2.2 Bradley-Terry Model:
    #
    # P(player i beats player k) =
    #    aux_i gamma_i / (aux_i gamma_i + gamma_k)
    #
    # So for an ascent on a climb with rating gamma_k:
    #
    # P(win) = (aux_i gamma_i + 0) / (aux_i gamma_i + gamma_k)
    #    so A = aux_i, B = 0, C = aux_i, D = gamma_k
    #
    # P(loss) = (0 gamma_i + gamma_k) / (aux_i gamma_i + gamma_k)
    #    so A = 0, B = gamma_k, C = aux_i, D = gamma_k
    cdef Py_ssize_t n = gamma.shape[0]
    cdef double[::1] d1_terms = cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0)
    cdef double[::1] d2_terms = cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0)

    cdef double t, u, phi
    for i in range(n):
        phi = aux_gamma[i] * gamma[i]
        t = phi + adversary_gamma[i]

        # C_ij gamma_i / (C_ij gamma_i + D_ij)
        u = phi / t
        d1_terms[i] = u

        # C_ij gamma_i D_ij / (C_ij gamma_i + D_ij)^2
        u *= adversary_gamma[i] / t

        d2_terms[i] = u

    return (d1_terms, d2_terms)


def get_bt_summation_terms(double[::1] gamma, double[::1] aux_gamma,
                           double[::1] adversary_gamma):
    """Wraps cget_bt_summation_terms() for testing"""
    return cget_bt_summation_terms(gamma, aux_gamma, adversary_gamma)


cdef double csum(const double[::1] x, Py_ssize_t start, Py_ssize_t end):
    """Compute the sum of x[start:end], with error compensation."""
    # Neumaier's improved Kahan–Babuška summation algorithm.  To be effective,
    # this must be compiled with clang's "-fno-associative-math" or equivalent.
    cdef double s = 0.0
    cdef Py_ssize_t i

    cdef double c = 0.0
    cdef double t
    for i in range(start, end):
        t = s + x[i]
        if fabs(s) >= fabs(x[i]):
            c += (s - t) + x[i]
        else:
            c += (x[i] - t) + s

        s = t
    return s + c


def sum(const double[::1] x, Py_ssize_t start, Py_ssize_t end):
    """Wraps csum() for testing"""
    return csum(x, start, end)


def get_bt_derivatives(list slices, double[::1] wins, double[::1] gamma,
                       double[::1] aux_gamma, double[::1] adversary_gamma):
    """Get the derivatives of the log-likelihood for each player.

    A player is an abstraction for an entity with a rating; it will correspond
    to a page or a route.

    Parameters
    ----------
    slices : list of pairs
        (start, end) indices representing slices of the ascents for each player.
    wins : contiguous ndarray
        Number of wins for each player.
    gamma : contiguous ndarray
        Rating of the "player" for each ascent.
    aux_gamma : contiguous ndarray
        Auxiliary coefficient to the player's rating for each ascent.
    adversary_gamma : contiguous ndarray
        Rating of the adversary for each ascent.

    Returns
    -------
    (d1 : ndarray, d2 : ndarray)
        A pair of ndarrays of the first and second derivative of the
        Bradley-Terry log-likelihood a "player" wins, with respect to the
        "natural rating" of that player.
    """
    cdef double[::1] d1_terms, d2_terms
    d1_terms, d2_terms = cget_bt_summation_terms(
        gamma, aux_gamma, adversary_gamma
    )

    cdef Py_ssize_t num_slices = len(slices)

    cdef double[::1] d1 = cnp.PyArray_EMPTY(1, [num_slices], cnp.NPY_DOUBLE, 0)
    cdef double[::1] d2 = cnp.PyArray_EMPTY(1, [num_slices], cnp.NPY_DOUBLE, 0)

    cdef Py_ssize_t start, end
    cdef int i
    cdef double d1_sum, d2_sum
    cdef tuple pair
    for i, pair in enumerate(slices):
        start, end = pair
        if start == end:
            d1[i] = 0.0
            d2[i] = 0.0
            continue

        # WHR Appendix A.1:
        # d ln P / d r = |W_i| - sum( C_ij gamma_i / (C_ij gamma_i + D_ij) )
        #
        # We move gamma_i into the sum for numerical stability: terms will be
        # closer to unity.
        d1[i] = wins[i] - csum(d1_terms, start, end)
        # WHR Appendix A.1:
        # d^2 ln P / dr ^2 = -sum( C_ij gamma_i D_ij / (C_ij gamma_i + D_ij)^2 )
        d2[i] = -csum(d2_terms, start, end)

    return (d1.base, d2.base)
