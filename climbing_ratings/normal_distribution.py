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


import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Sequence, Tuple, Union


_Array = NDArray[np.float_]


_FloatLike = Union[float, _Array]


class NormalDistribution:
    """Models variables drawn from a Gaussian distribution."""

    __slots__ = ("_mu", "_sigma_sq")

    def __init__(self, mu: _FloatLike, sigma_sq: _FloatLike):
        """
        Parameters
        ----------
        mu
            Mean of the normal distribution.
        sigma_sq
            Variance of the normal distribution.  Must be positive.
        """
        self._mu: _Array = np.asarray(mu)
        self._sigma_sq: _Array = np.asarray(sigma_sq)

    @property
    def sigma_sq(self) -> Union[float, ArrayLike]:
        """The variance parameter."""
        return self._sigma_sq

    def get_derivatives(self, x: _Array) -> Tuple[_Array, _Array]:
        """Return the first and second derivative of the log-likelihood.

        This method is vectorized: it will pair the distribution parameters
        from the initialization to each value of "x", and return an array of
        the same length.

        Parameters
        ----------
        x
            Samples from the distribution.

        Returns
        -------
            The first and second derivatives of the log-PDF, evaluated at x.
        """
        y: _Array = self._mu - x
        y /= self._sigma_sq

        return (y, -1.0 / self._sigma_sq)
