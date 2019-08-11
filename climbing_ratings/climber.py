"""WHR model of a climber"""

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

import collections
import numpy as np
from climbing_ratings.gamma_distribution import GammaDistribution
import climbing_ratings.loop_helpers as loop_helpers


class TriDiagonal(collections.namedtuple('TriDiagonal', ['d', 'u', 'l'])):
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


class TriDiagonalLU(collections.namedtuple('TriDiagonalLU', ['d', 'b', 'a'])):
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
    """


def lu_decomposition(tri_diagonal):
    """Decompose a tri-diagonal matrix into LU form.

    Parameters
    ----------
    tri_diagonal : TriDiagonal
        Represents the matrix to decompose.
    """
    # WHR Appendix B: perform LU decomposition
    #
    # d[0] = hd[0]
    # b[i] = hu[i]
    #
    # Iterative algorithm:
    #   d[i] = hd[i] - hu[i-1] a[i-1]
    #   a[i] = hl[i] / d[i]

    hd, hu, hl = tri_diagonal
    b = hu

    # We want to vectorize the calculation of d and a as much as possible,
    # instead of using WHR's iterative algorithm directly.
    #
    # Substitute a[i] into the expression for d[i+1] to get a recurrence
    # relation for d:
    #
    #   d[i] = hd[i] - hu[i-1] a[i-1]
    #        = hd[i] - hu[i-1] * hl[i-1] / d[i-1]
    #
    # Let c[i] = hu[i-1] * hl[i-1].
    # c[0] = 0, which is meaningless but convenient for the helper.
    #
    #   d[i] = hd[i] - c[i] / d[i-1]
    c = np.empty_like(hd)
    c[0] = 0.
    np.multiply(hu, hl, c[1:])
    np.negative(c, c)
    d = hd.copy()
    loop_helpers.lu_decomposition_helper(c, d)

    # a[i] = hl[i] / d[i]
    a = np.divide(hl, d[:-1])
    return TriDiagonalLU(d, b, a)


def invert_h_dot_g(lu, g):
    """Compute M^-1 G.

    Where M = LU, and LU X = G, this is equivalent to solving X.

    See WHR Appendix B.1.

    Parameters
    ----------
    lu : TriDiagonal
        The tri-diagonal LU decomposition for a square matrix of shape (N, N)
    g : contiguous ndarray with length N
    """
    d, b, a = lu

    # Create a copy of "a", negated and padded with a leading 0.
    a1 = np.empty_like(d)
    a1[0] = 0.
    a1[1:] = np.negative(a)

    y = a1  # output parameter
    loop_helpers.ly_helper(g, a1)

    x = y  # output parameter
    loop_helpers.ux_helper(b, d, y)
    return x


class Climber:
    """Models a climber.

    A climber has an associated set of "ratings", each corresponding to a
    particular period (a page).  Each page corresponds to one or more ascents.

    Instances store state that is invariant over estimation.

    Class attributes
    ----------------
    wiener_variance : float
        The variance of the climber's Wiener process per unit time.
    gamma_distribution : GammaDistribution
        Prior distribution for the climbers' initial rating.
    """
    wiener_variance = 1.
    gamma_distribution = GammaDistribution(1.)

    def __init__(self, gaps):
        """Initializes a Climber.

        Parameters
        ----------
        gaps : ndarray
            gaps[i] is the time interval between the page i and page i + 1.
            Must be consistent with the time scale for Climber.wiener_variance.
            The length of gaps should be 1 less than the number of pages.
        """
        self.gaps = gaps
        s = np.full_like(gaps, Climber.wiener_variance)
        s *= gaps
        np.reciprocal(s, s)
        self.one_on_sigma_sq = s

    def add_wiener_hessian(self, hessian):
        """Add terms from the Wiener prior to the Hessian and gradient."""
        # WHR Appendix A.2 Terms of the Wiener prior:
        # d^2 ln p / d r[t]^2 = -1 / sigma^2
        # d^2 ln p / d r[t] d r[t+1] = 1 / sigma^2
        hd, hu, hl = hessian
        hd[:-1] -= self.one_on_sigma_sq
        hd[1:] -= self.one_on_sigma_sq
        hu += self.one_on_sigma_sq
        hl += self.one_on_sigma_sq

    def add_wiener_gradient(self, ratings, gradient):
        """Add terms from the Wiener prior to the gradient."""
        # WHR Appendix A.2 Terms of the Wiener prior:
        # d ln p / d r[t] = - (r[t] - r[t+1]) / sigma[t]^2
        r = np.log(ratings)
        d = r[:-1]  # output parameter
        np.subtract(r[1:], r[:-1], d)
        d *= self.one_on_sigma_sq
        gradient[:-1] += d
        gradient[1:] += d

    def get_derivatives(self, ratings, bt_d1, bt_d2):
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
        ratings : ndarray
            The rating (gamma) for each of the climber's pages.
        bt_d1 : ndarray
            First derivative from the Bradley-Terry model.
        bt_d2 : ndarray
            Second derivative from the Bradley-Terry model.

        Returns
        -------
        (gradient : ndarray, hessian : TriDiagonal)
            The gradient and hessian of the conditional log-likelihood.
        """

        # Wiener terms.
        n = ratings.shape[0]
        hd = np.zeros([n])
        hu = np.zeros([n - 1])
        hl = np.zeros([n - 1])
        hessian = TriDiagonal(hd, hu, hl)
        gradient = np.zeros([n])

        self.add_wiener_hessian(hessian)
        self.add_wiener_gradient(ratings, gradient)

        # Bradley-Terry terms.
        gradient += bt_d1
        hd += bt_d2

        # Gamma terms.
        gamma_d1, gamma_d2 = Climber.gamma_distribution.get_derivatives(
            ratings[0])
        gradient[0] += gamma_d1
        hessian.d[0] += gamma_d2

        return (gradient, hessian)

    def get_ratings_adjustment(self, ratings, bt_d1, bt_d2):
        """Apply Newton's method to revise ratings estimates.

        Parameters
        ----------
        ratings : ndarray
            Current estimate of ratings for each of this climber's pages.
        bt_d1 : ndarray
            Bradley-Terry derivatives for each of this climber's pages.
        bt_d2 : ndarray
            Bradley-Terry second-derivatives for each of this climber's pages.

        Returns
        -------
        ratings : ndarray
            Deltas to subtract from the current ratings.
        """
        gradient, hessian = self.get_derivatives(ratings, bt_d1, bt_d2)
        lu = lu_decomposition(hessian)
        return invert_h_dot_g(lu, gradient)
