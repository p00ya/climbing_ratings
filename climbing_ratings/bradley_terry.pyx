"""Bradley-Terry model"""

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

import numpy as np
from libc.math cimport fabs


def get_bt_summation_terms(double[::1] gamma, double[::1] adversary_gamma):
    """Get the Bradley-Terry summation terms for each player.

    A player is an abstraction for an entity with a rating; it will correspond
    to a page or a route.

    Parameters
    ----------
    gamma : contiguous ndarray
        Rating of the "player" for each ascent.
    adversary_gamma : contiguous ndarray
        Rating of the adversary for each ascent.

    Returns
    -------
    (d1_terms : ndarray, d2_terms : ndarray)
        d1_terms contains the "1 / (gamma + adversary_gamma)" terms for each
        player.
        d2_terms contains the
        "adversary_gamma / (gamma + adversary_gamma)^2" terms for each player.
    """
    # WHR Appendix A.1 Terms of the Bradley-Terry model:
    #
    # P(G_j) = (A_ij gamma_i + B_ij) / (C_ij gamma_i + D_ij)
    #
    # WHR 2.2 Bradley-Terry Model:
    #
    # P(player i beats player k) =
    #    gamma_i / gamma_i + gamma_k
    #
    # So for an ascent on a climb with rating gamma_k:
    #
    # P(win) = (1 gamma_i + 0) / (1 gamma_i + gamma_k)
    #    so A = 1, B = 0, C = 1, D = gamma_k
    #
    # P(loss) = (0 gamma_i + gamma_k) / (1 gamma_i + gamma_k)
    #    so A = 0, B = gamma_k, C = 1, D = gamma_k
    cdef Py_ssize_t n = gamma.shape[0]
    d1_arr = np.empty([n], dtype=np.float)
    d2_arr = np.empty([n], dtype=np.float)

    cdef double[::1] d1_terms = d1_arr
    cdef double[::1] d2_terms = d2_arr

    cdef double t
    for i in range(n):
        t = gamma[i]
        t += adversary_gamma[i]
        t = 1.0 / t

        # 1 / (C_ij gamma_i + D_ij)
        d1_terms[i] = t

        # D_ij / (C_ij gamma_i + D_ij)^2
        t *= t
        t *= adversary_gamma[i]
        d2_terms[i] = t

    return (d1_arr, d2_arr)


cpdef double sum(const double[::1] x, Py_ssize_t start, Py_ssize_t end):
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


def get_bt_derivatives(list slices, double[::1] wins, double[::1] gamma,
                       double[::1] adversary_gamma):
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
    adversary_gamma : contiguous ndarray
        Rating of the adversary for each ascent.

    Returns
    -------
    (d1 : ndarray, d2 : ndarray)
        A pair of ndarrays of the first and second derivative of the
        Bradley-Terry log-likelihood a "player" wins, with respect to the
        "natural rating" of that player.
    """
    d1_terms_arr, d2_terms_arr = get_bt_summation_terms(gamma, adversary_gamma)
    cdef double[::1] d1_terms = d1_terms_arr
    cdef double[::1] d2_terms = d2_terms_arr

    cdef Py_ssize_t num_slices = len(slices)

    d1_arr = np.empty(num_slices)
    d2_arr = np.empty(num_slices)
    cdef double[::1] d1 = d1_arr
    cdef double[::1] d2 = d2_arr
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
        # d ln P / d r = |W_i| - gamma_i sum( C_ij / (C_ij gamma_i + D_ij) )
        d1_sum = sum(d1_terms, start, end)
        d1[i] = wins[i] - gamma[start] * d1_sum
        # WHR Appendix A.1:
        # d^2 ln P / d r^2 = -gamma * sum( C_ij D_ij / (C_ij + D_ij)^2 )
        d2_sum = sum(d2_terms, start, end)
        d2[i] = -d2_sum * gamma[start]

    return (d1_arr, d2_arr)
