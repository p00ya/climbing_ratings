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
from typing import NamedTuple

class TriDiagonal(NamedTuple):
    d: np.ndarray
    u: np.ndarray
    l: np.ndarray

class TriDiagonalLU(NamedTuple):
    d: np.ndarray
    b: np.ndarray
    a: np.ndarray

def add_wiener_gradient(
    one_on_sigma_sq: np.ndarray,
    ratings: np.ndarray,
    gradient: np.ndarray,
) -> None: ...
def lu_decompose(tri_diagonal: TriDiagonal) -> TriDiagonalLU: ...
def ul_decompose(tri_diagonal: TriDiagonal) -> TriDiagonalLU: ...
def solve_ul_d(c: np.ndarray, hd: np.ndarray) -> None: ...
def solve_y(g: np.ndarray, a: np.ndarray) -> None: ...
def solve_x(b: np.ndarray, d: np.ndarray, y: np.ndarray) -> None: ...
