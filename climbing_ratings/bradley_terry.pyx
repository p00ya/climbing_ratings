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

from .csum cimport csum
from .slices cimport Slices
cimport numpy as cnp
from cpython.mem cimport PyMem_Malloc, PyMem_Free


# Workaround for "expl" being missing from Cython's libc.math.
# See: https://github.com/cython/cython/issues/3570
cdef extern from "<math.h>" nogil:
    long double coshl(long double)
    long double expl(long double)


cnp.import_array()


def expand_to_slices(double[::1] values, Slices slices, double[::1] out):
    """Expand normalized values to contiguous blocks.

    A member of the output array x[i] will equal values[j] if
    slices[j][0] <= i < slices[j][1].

    Parameters
    ----------
    values : ndarray
        The normalized values.
    slices : list of pairs
        The (start, end) pairs corresponding to a slice in the output.  The
        implied slices must be contiguous and in ascending order.
    out : ndarray
        The output array.
    """
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t i, end
    for i in range(slices.end.shape[0]):
        while j < slices.end[i]:
            out[j] = values[i]
            j += 1

    return out.base


def expand_to_slices_sparse(double[::1] values, Slices slices, Py_ssize_t n):
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
    for i in range(slices.start.shape[0]):
        start = slices.start[i]
        end = slices.end[i]
        v = values[i]
        for j in range(start, end):
            expanded[j] = v

    return expanded.base


cdef class BradleyTerry():
    """Transient state for calculating Bradley-Terry derivatives.

    The buffers may be re-used across multiple iterations.  Re-use is just a
    performance optimization: results from previous iterations do not affect
    get_derivatives.
    """

    # Ratings by ascent.
    cdef double[::1] ratings

    # Derivative terms by ascent.
    cdef long double *d1_terms
    cdef long double *d2_terms

    # Derivatives by player.
    cdef double[::1] d1
    cdef double[::1] d2

    # Ascent slices for each player.
    cdef Slices slices

    def __init__(self, Slices slices, Py_ssize_t num_ascents, Py_ssize_t num_players):
        """
        Parameters
        ----------
        slices
            The ranges of ascents for each player.
        num_ascents
            Number of ascents.
        num_players
            Number of players.
        """
        if len(slices) != num_players:
            raise IndexError(f"len(slices) {len(slices)} != {num_players}")

        self.ratings = cnp.PyArray_EMPTY(1, [num_ascents], cnp.NPY_DOUBLE, 0)

        cdef long double *p
        p = <long double *> PyMem_Malloc(num_ascents * sizeof(long double))
        if not (p):
            raise MemoryError()
        self.d1_terms = p

        p = <long double *> PyMem_Malloc(num_ascents * sizeof(long double))
        if not (p):
            raise MemoryError()
        self.d2_terms = p

        self.d1 = cnp.PyArray_EMPTY(1, [num_players], cnp.NPY_DOUBLE, 0)
        self.d2 = cnp.PyArray_EMPTY(1, [num_players], cnp.NPY_DOUBLE, 0)

        self.slices = slices

    def __dealloc__(self):
        PyMem_Free(self.d1_terms)
        PyMem_Free(self.d2_terms)

    @property
    def ratings(self):
        """The ratings of the player for each ascent."""
        return self.ratings.base

    def get_derivatives(
        self,
        double[::1] win,
        double[::1] adversary,
    ):
        """Get the derivatives of the log-likelihood for each player.

        A player is an abstraction for an entity with a rating; it will
        correspond to a page or a route.

        The player ratings should be set using fill_from_slices prior to calling
        this method.

        Parameters
        ----------
        slices : Slices
            (start, end) indices representing slices of the ascents for each
            player.
        win : contiguous ndarray
            1.0 if the ascent was a "win" for the player, -1.0 otherwise, for
            each ascent.
        adversary : contiguous ndarray
            Natural rating of the adversary for each ascent.

        Returns
        -------
        (d1 : ndarray, d2 : ndarray)
            A pair of ndarrays of the first and second derivative of the
            Bradley-Terry log-likelihood a "player" wins, with respect to the
            "natural rating" of that player.
        """
        cdef Py_ssize_t num_ascents = self.ratings.shape[0]
        cdef Py_ssize_t num_players = self.d1.shape[0]

        if win.shape[0] != num_ascents:
            raise IndexError(f"len(win) {win.shape[0]} != {num_ascents}")

        if adversary.shape[0] != num_ascents:
            raise IndexError(f"len(adversary) {adversary.shape[0]} != {num_ascents}")

        _cget_bt_summation_terms(
            win,
            self.ratings,
            adversary,
            num_ascents,
            self.d1_terms,
            self.d2_terms,
        )

        cdef Py_ssize_t start, end
        cdef Py_ssize_t i
        for i in range(self.slices.start.shape[0]):
            start = self.slices.start[i]
            end = self.slices.end[i]
            if start == end:
                self.d1[i] = 0.0
                self.d2[i] = 0.0
                continue

            # Instead of using WHR Appendix A.1's factorization (with its A, B,
            # C, D and gamma terms), we sum the actual derivatives for each
            # ascent, which is more numerically stable.
            self.d1[i] = csum(self.d1_terms, start, end)
            self.d2[i] = csum(self.d2_terms, start, end)

        return (self.d1.base, self.d2.base)


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
    for i in range(n):
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

        t = <long double> player[i] - adversary[i]
        # t = winner - loser
        t *= win[i]

        d1_terms[i] = win[i] / (expl(t) + 1.0)
        d2_terms[i] = -0.5 / (coshl(t) + 1.0)


def _get_bt_summation_terms(
    double[::1] win, double[::1] player, double[::1] adversary
):
    """Wraps _cget_bt_summation_terms() for testing"""
    cdef Py_ssize_t n = player.shape[0]
    cdef long double[::1] d1 = cnp.PyArray_EMPTY(1, [n], cnp.NPY_LONGDOUBLE, 0)
    cdef long double[::1] d2 = cnp.PyArray_EMPTY(1, [n], cnp.NPY_LONGDOUBLE, 0)
    _cget_bt_summation_terms(win, player, adversary, n, &d1[0], &d2[0])
    return (d1, d2)
