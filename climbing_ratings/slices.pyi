# type stubs for slices.pyx

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

from collections.abc import Generator, Iterable
from typing import List, Tuple

class Slices(Iterable[Tuple[int, int]]):
    def __init__(self, slices: List[Tuple[int, int]]) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Tuple[int, int]: ...
    def __iter__(self) -> Generator[Tuple[int, int], None, None]: ...
