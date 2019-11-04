"""Gamma distribution"""

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


class GammaDistribution:
    """Models variables drawn from a gamma distribution.

    Note the "gamma" here is not the same as WHR's "gamma" rating.  It refers
    to the well-known Gamma probability distribution.

    Class Attributes
    ----------
    shape : float
        Shape parameter of the gamma distribution.
    """

    shape = 2.0

    def __init__(self, mode):
        """
        Parameters
        ----------
        mode : array_like
            Mode of the distribution(s).
        """
        # Theta is the conventional "scale" parameter for the distribution.
        self._theta = mode / (GammaDistribution.shape - 1.0)

    def get_derivatives(self, x):
        """Return the first and second derivative of the log-likelihood.

        Note these are the derivatives of "ln P" with respect to "ln x", not
        with respect to x.

        This method is vectorized: it will pair the distribution parameters
        from the initialization to each value of "x", and return an array of
        the same length.

        Parameters
        ----------
        x : ndarray
            Samples from the distribution.  Should have the same length as the
            this object's "mode" parameter.

        Returns
        -------
        (d1 : ndarray, d2 : ndarray)
            The first and second derivatives of the log-PDF wrt log(x),
            evaluated at x.
        """
        d2 = -x
        d2 /= self._theta
        d1 = d2 + (GammaDistribution.shape - 1)
        return (d1, d2)
