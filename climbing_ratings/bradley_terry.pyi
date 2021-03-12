# type stubs for Python functions in bradley_terry.pyx

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
from ctypes import c_longdouble
from typing import List, Tuple

_Slice = Tuple[int, int]

def expand_to_slices(
    values: np.ndarray,
    slices: List[_Slice],
    n: int,
) -> np.ndarray: ...
def expand_to_slices_sparse(
    values: np.ndarray,
    slices: List[_Slice],
    n: int,
) -> np.ndarray: ...
def get_bt_derivatives(
    slices: List[_Slice],
    win: np.ndarray,
    player: np.ndarray,
    aux: np.ndarray,
    adversary: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]: ...
def _get_bt_summation_terms(
    win: np.ndarray,
    player: np.ndarray,
    aux: np.ndarray,
    adversary: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]: ...
def _sum(
    x: np.ndarray,
    start: int,
    end: int,
) -> c_longdouble: ...
