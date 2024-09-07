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
from numpy.typing import NDArray
from typing import List, Tuple
from .slices import Slices

_Array = NDArray[np.float64]

def expand_to_slices(
    values: _Array,
    slices: Slices,
    out: _Array,
) -> _Array: ...
def expand_to_slices_sparse(
    values: _Array,
    slices: Slices,
    n: int,
) -> _Array: ...

class BradleyTerry:
    def __init__(self, slices: Slices, num_ascents: int, num_players: int) -> None: ...
    @property
    def ratings(self) -> _Array: ...
    def get_derivatives(
        self,
        win: _Array,
        adversary: _Array,
    ) -> Tuple[_Array, _Array]: ...

def get_bt_derivatives(
    win: _Array,
    player: _Array,
    adversary: _Array,
) -> Tuple[_Array, _Array]: ...
def _get_bt_summation_terms(
    win: _Array,
    player: _Array,
    adversary: _Array,
) -> Tuple[_Array, _Array]: ...
