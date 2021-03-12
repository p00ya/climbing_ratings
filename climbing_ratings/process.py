"""WHR Wiener process model"""

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
import numpy.typing as npt
from .normal_distribution import NormalDistribution
from .process_helpers import (
    add_wiener_gradient,
    lu_decompose,
    ul_decompose,
    solve_x,
    solve_y,
    TriDiagonal,
    TriDiagonalLU,
)
from numpy import ndarray
from typing import Sequence, Tuple


class Process:
    """Models ratings over time, as a Wiener process.

    A ratings process has an associated set of samples, each corresponding to a
    particular time period (a page).  Each page corresponds to zero or more
    ascents.

    Instances store state that is invariant over estimation.
    """

    # Private attributes
    # -------------------
    # _initial_prior
    #     The prior distribution for the first page of the climber.
    # _one_on_sigma_sq
    #     The Wiener variance between each page and the next page.  The length
    #     is 1 fewer than the number of pages.
    # _wiener_d2
    #     The second derivative terms (diagonal of the Hessian matrix) from the
    #     Wiener prior, for each page.
    __slots__ = ("_initial_prior", "_one_on_sigma_sq", "_wiener_d2")

    def __init__(
        self,
        wiener_variance: float,
        initial_prior: NormalDistribution,
        gaps: ndarray,
    ):
        """Initialize a Process.

        Parameters
        ----------
        wiener_variance
            The variance of the Wiener process per unit time.
        initial_prior
            Prior distribution for the first page of the climber.
        gaps
            gaps[i] is the time interval between the page i and page i + 1.
            The length of gaps should be 1 fewer than the number of pages.
        """
        self._initial_prior = initial_prior
        s = gaps * wiener_variance

        np.reciprocal(s, s)
        self._one_on_sigma_sq = s
        # WHR Appendix A.2 Terms of the Wiener prior:
        # d^2 ln p / d r[t]^2 = -1 / sigma^2
        hd = np.empty(s.shape[0] + 1)
        hd[0] = 0.0
        np.negative(s, hd[1:])
        hd[:-1] -= s
        self._wiener_d2 = hd

    def __get_derivatives(
        self, ratings: ndarray, bt_d1: ndarray, bt_d2: ndarray
    ) -> Tuple[ndarray, TriDiagonal]:
        """Return the Hessian and gradient at the given ratings point.

        Evaluates the Hessian matrix and gradient vector for the conditional
        log-likelihood.  The matrix is tri-diagonal; so given the notation
        defined by the TriDiagonal class it is defined by:

          d[t] = d^2 ln P / d (r[t])^2
          u[t] = d^2 ln P / (d r[t] d r[t+1])
          l[t] = d^2 ln P / (d r[t] d r[t+1])

        It incorporates terms from both the Bradley-Terry model of individual
        ascent success, and the Wiener prior for the change of climber's
        rating between periods.

        Parameters
        ----------
        ratings
            The natural rating for each of the climber's pages.
        bt_d1
            First derivative from the Bradley-Terry model.
        bt_d2
            Second derivative from the Bradley-Terry model.

        Returns
        -------
            The gradient and hessian of the conditional log-likelihood.
        """

        # Wiener terms.
        gradient = np.zeros_like(ratings)
        one_on_sigma_sq = self._one_on_sigma_sq
        add_wiener_gradient(one_on_sigma_sq, ratings, gradient)
        hd = np.copy(self._wiener_d2)

        # Bradley-Terry terms.
        gradient += bt_d1
        hd += bt_d2

        # First page Gaussian terms.
        initial_d = self._initial_prior.get_derivatives(ratings[0])
        gradient[0] += initial_d[0]
        hd[0] += initial_d[1]

        hessian = TriDiagonal(hd, one_on_sigma_sq, one_on_sigma_sq)
        return (gradient, hessian)

    def get_ratings_adjustment(
        self, ratings: ndarray, bt_d1: ndarray, bt_d2: ndarray
    ) -> ndarray:
        """Apply Newton's method to revise ratings estimates.

        Parameters
        ----------
        ratings
            Last estimate of natural ratings for each of this climber's pages.
        bt_d1
            Bradley-Terry derivatives for each of this climber's pages.
        bt_d2
            Bradley-Terry second-derivatives for each of this climber's pages.

        Returns
        -------
        ratings
            Deltas to subtract from the current ratings.
        """
        gradient, hessian = self.__get_derivatives(ratings, bt_d1, bt_d2)
        lu = lu_decompose(hessian)
        return _invert_lu_dot_g(lu, gradient)

    def get_covariance(
        self,
        ratings: ndarray,
        bt_d1: ndarray,
        bt_d2: ndarray,
        var: ndarray,
        cov: ndarray,
    ) -> None:
        """Return the covariance matrix for the ratings.

        Parameters
        ----------
        ratings
            The natural rating for each of the climber's pages.
        bt_d1
            First derivative from the Bradley-Terry model.
        bt_d2
            Second derivative from the Bradley-Terry model.
        var
            The output array for the variance for each of the natural ratings.
        cov
            The output array for the covariance between the natural ratings of
            each page the next page.
        """
        _, hessian = self.__get_derivatives(ratings, bt_d1, bt_d2)
        lu = lu_decompose(hessian)
        ul = ul_decompose(hessian)
        _invert_lu(lu, ul, var, cov)


def _invert_lu_dot_g(lu: TriDiagonalLU, g: ndarray) -> ndarray:
    """Compute M^-1 G.

    Where M = LU, and LU X = G, this is equivalent to solving X.

    See WHR Appendix B.1.

    Parameters
    ----------
    lu
        The tri-diagonal LU decomposition for a square matrix of shape (N, N).
    g
        Contiguous array with length N.

    Returns
    -------
        An array with length X, satisfying LU X = G.
    """
    d, b, a = lu

    # Create a copy of "a", negated and padded with a leading 0.
    a1 = np.empty_like(d)
    a1[0] = 0.0
    a1[1:] = np.negative(a)

    y = a1  # output parameter
    solve_y(g, a1)

    x = y  # output parameter
    solve_x(b, d, y)
    return x


def _invert_lu(
    lu: TriDiagonalLU, ul: TriDiagonalLU, d_arr: ndarray, l_arr: ndarray
) -> None:
    """Compute -M^-1.

    For the square matrix M = LU = U'L', solve the diagonal and lower
    sub-diagonal of the negative inverse of M.

    Parameters
    ----------
    lu
        The tri-diagonal LU decomposition for the square matrix M
    ul
        The tri-diagonal UL decomposition for the square matrix M.
    d_arr
        The output array for the diagonal of the negative inverse of M.  Its
        length must be the same as the order of M.
    l_arr
        The output array for the lower sub-diagonal of the negative inverse
        of M.  Its length must be one less than the order of M.
    """
    # WHR Appendix B.2: Computing Diagonal and Sub-diagonal Terms of H^-1
    d = d_arr[:-1]
    d_ul = ul.d[1:]

    # d[i] d'[i+1]
    np.copyto(d_arr, lu.d)
    np.multiply(d, d_ul, d)

    # b[i] b'[i]
    b = np.multiply(lu.b, ul.b, l_arr)

    # b[i] b'[i] - d[i] d'[i+1]
    np.subtract(b, d, d)

    # diagonal[i] = d'[i+1] / (b[i] b'[i] - d[i] d'[i+1])
    np.divide(d_ul, d, d)
    # base case; diagonal[n] = -1 / d[n]
    d_arr[-1] = -1.0 / d_arr[-1]

    # subdiagonal[i] = -a[i] diagonal[i+1]
    np.multiply(lu.a, d_arr[1:], b)
    np.negative(b, b)
