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


def get_bt_derivatives(slices, wins, gamma, adversary_gamma):
    """Get the derivatives of the log-likelihood for each player.

    A player is an abstraction for an entity with a rating; in will correspond
    to a page or a route.

    Parameters
    ----------
    slices : list of pairs
        (start, end) indices representing slices of the ascents for each player.
    wins : ndarray
        Number of wins for each player.
    gamma : ndarray
        Rating of the "player" for each ascent.
    adversary_gamma : ndarray
        Rating of the adversary for each ascent.

    Returns
    -------
    (d1 : ndarray, d2 : ndarray)
    A pair of ndarrays of the first and second derivative of the
    Bradley-Terry log-likelihood a "player" wins, with respect to the
    "natural rating" of that player.
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

    # 1 / (C_ij gamma_i + D_ij)
    one_on_gamma_plus_d = gamma + adversary_gamma
    np.reciprocal(one_on_gamma_plus_d, one_on_gamma_plus_d)

    # C_ij D_ij / (C_ij + D_ij)^2
    d2_terms = np.square(one_on_gamma_plus_d)
    d2_terms *= adversary_gamma

    d1_terms = np.cumsum(one_on_gamma_plus_d, out=one_on_gamma_plus_d)
    np.cumsum(d2_terms, out=d2_terms)

    d1 = np.empty(len(slices))
    d2 = np.empty(len(slices))
    for i, (start, end) in enumerate(slices):
        # WHR Appendix A.1:
        # d ln P / d r = |W_i| - gamma_i sum( C_ij / (C_ij gamma_i + D_ij) )
        d1_sum = d1_terms[end - 1]
        if start > 0:
            d1_sum -= d1_terms[start - 1]
        d1[i] = wins[i] - gamma[start] * d1_sum
        # WHR Appendix A.1:
        # d^2 ln P / d r^2 = -gamma * sum( C_ij D_ij / (C_ij + D_ij)^2 )
        d2_sum = d2_terms[end - 1]
        if start > 0:
            d2_sum -= d2_terms[start - 1]
        d2[i] = -d2_sum * gamma[start]

    return (d1, d2)
