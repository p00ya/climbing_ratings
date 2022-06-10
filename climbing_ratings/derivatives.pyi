# type stubs for derivatives.pyx

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
from collections.abc import Sized
from numpy.typing import NDArray
from typing import List, Tuple

_Array = NDArray[np.float_]

class TriDiagonal:
    def __init__(self, d: _Array, u: _Array, l: _Array) -> None: ...
    def as_tuple(self) -> Tuple[_Array, _Array, _Array]: ...

class TriDiagonalLU:
    def __init__(self, d: _Array, b: _Array, a: _Array) -> None: ...
    def as_tuple(self) -> Tuple[_Array, _Array, _Array]: ...

class NormalDistribution:
    def __init__(self, mu: float, sigma_sq: float) -> None: ...
    def gradient(self, x: float) -> float: ...
    def d2(self) -> float: ...

class MultiNormalDistribution:
    def __init__(self, mu: _Array, sigma_sq: float) -> None: ...
    def add_gradient(self, x: _Array, out: _Array) -> None: ...
    def d2(self) -> float: ...

class WienerProcess:
    def __init__(
        self, gaps: _Array, slices: Slices, wiener_variance: float
    ) -> None: ...
    def _add_gradient(self, ratings: _Array, gradient: _Array) -> None: ...
    def as_tuple(self) -> Tuple[_Array, _Array]: ...

class Slices(Sized):
    def __init__(self, slices: List[Tuple[int, int]]) -> None: ...
    def __len__(self) -> int: ...

class PageInvariants:
    def __init__(
        self,
        initial: NormalDistribution,
        wiener: WienerProcess,
        slices: Slices,
    ) -> None: ...

class PageModel:
    def __init__(self, model: PageInvariants, ratings: _Array) -> None: ...
    def _update_derivatives(self, bt_d1: _Array, bt_d2: _Array) -> None: ...
    def update_ratings(
        self,
        bt_d1: _Array,
        bt_d2: _Array,
    ) -> _Array: ...
    def update_covariance(
        self,
        bt_d1: _Array,
        bt_d2: _Array,
    ) -> Tuple[_Array, _Array]: ...
    def __copy__(self) -> PageModel: ...
    @property
    def ratings(self) -> _Array: ...
    @property
    def var(self) -> _Array: ...
    @property
    def cov(self) -> _Array: ...
    @property
    def _gradient(self) -> _Array: ...
    @property
    def _hessian(self) -> Tuple[_Array, _Array, _Array]: ...

def lu_decompose(h: TriDiagonal, lu: TriDiagonalLU, start: int, end: int) -> None: ...
def invert_lu_dot_g(
    lu: TriDiagonalLU,
    g: _Array,
    out: _Array,
    start: int,
    end: int,
) -> None: ...
def ul_decompose(h: TriDiagonal, ul: TriDiagonalLU, start: int, end: int) -> None: ...
def solve_ul_d(c: _Array, hd: _Array, start: int, end: int) -> None: ...
def solve_y(g: _Array, a: _Array, y: _Array, start: int, end: int) -> None: ...
def solve_x(
    b: _Array,
    d: _Array,
    y: _Array,
    start: int,
    end: int,
) -> None: ...
def invert_lu(
    lu: TriDiagonalLU,
    ul: TriDiagonalLU,
    out_d: _Array,
    out_l: _Array,
    start: int,
    end: int,
) -> None: ...
