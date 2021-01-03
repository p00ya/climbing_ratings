"""Normal distribution"""

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


class NormalDistribution:
    """Models variables drawn from a Gaussian distribution."""

    __slots__ = ("_mu", "_sigma_sq")

    def __init__(self, mu, sigma_sq):
        """
        Parameters
        ----------
        mu : float or array_like
            Mean of the normal distribution.
        sigma_sq : float or array_like
            Variance of the normal distribution.  Must be positive.
        """
        self._mu = mu
        self._sigma_sq = sigma_sq

    @property
    def sigma_sq(self):
        """The variance parameter."""
        return self._sigma_sq

    def get_derivatives(self, x):
        """Return the first and second derivative of the log-likelihood.

        This method is vectorized: it will pair the distribution parameters
        from the initialization to each value of "x", and return an array of
        the same length.

        Parameters
        ----------
        x : ndarray
            Samples from the distribution.

        Returns
        -------
        (d1 : ndarray, d2 : ndarray)
            The first and second derivatives of the log-PDF, evaluated at x.
        """
        y = self._mu - x
        y /= self._sigma_sq

        return (y, -1.0 / self._sigma_sq)
