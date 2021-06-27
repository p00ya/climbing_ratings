# type stubs for Python functions in process_helpers.pyx

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
from numpy.typing import NDArray
from typing import NamedTuple

_Array = NDArray[np.float_]

class TriDiagonal(NamedTuple):
    d: _Array
    u: _Array
    l: _Array

class TriDiagonalLU(NamedTuple):
    d: _Array
    b: _Array
    a: _Array

def add_wiener_gradient(
    one_on_sigma_sq: _Array,
    ratings: _Array,
    gradient: _Array,
) -> None: ...
def lu_decompose(tri_diagonal: TriDiagonal) -> TriDiagonalLU: ...
def ul_decompose(tri_diagonal: TriDiagonal) -> TriDiagonalLU: ...
def solve_ul_d(c: _Array, hd: _Array) -> None: ...
def solve_y(g: _Array, a: _Array) -> None: ...
def solve_x(b: _Array, d: _Array, y: _Array) -> None: ...
