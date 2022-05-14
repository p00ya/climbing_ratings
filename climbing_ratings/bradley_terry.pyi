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
from numpy.typing import NDArray
from typing import List, Tuple

_Array = NDArray[np.float_]
_Slice = Tuple[int, int]

def expand_to_slices(
    values: _Array,
    slices: List[_Slice],
    n: int,
) -> _Array: ...
def expand_to_slices_sparse(
    values: _Array,
    slices: List[_Slice],
    n: int,
) -> _Array: ...
def get_bt_derivatives(
    slices: List[_Slice],
    win: _Array,
    player: _Array,
    adversary: _Array,
) -> Tuple[_Array, _Array]: ...
def _get_bt_summation_terms(
    win: _Array,
    player: _Array,
    adversary: _Array,
) -> Tuple[_Array, _Array]: ...
def _sum(
    x: NDArray[np.longdouble],
    start: int,
    end: int,
) -> c_longdouble: ...
