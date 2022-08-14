"""Slices"""

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


cimport numpy as cnp

cnp.import_array()


cdef class Slices:
    """The start and end indices of entities in a contiguous array."""

    # start and end attributes declared in pxd.

    def __init__(self, list slices):
        """Initialize slices.

        Parameters
        ----------
        slices : List[Tuple[Int, Int]]
            Start and end indices for each entity.
        """
        cdef Py_ssize_t num_slices = len(slices)
        self.start = cnp.PyArray_EMPTY(1, [num_slices], cnp.NPY_INTP, 0)
        self.end = cnp.PyArray_EMPTY(1, [num_slices], cnp.NPY_INTP, 0)
        cdef Py_ssize_t i, start, end
        cdef tuple pair
        for i, pair in enumerate(slices):
            start, end = pair
            self.start[i] = start
            self.end[i] = end

    def __len__(self):
        return self.start.shape[0]

    def __getitem__(self, int i):
        if i >= self.start.shape[0]:
            raise IndexError(f"{i} >= length {self.start.shape[0]}")

        return (self.start[i], self.end[i])

    def __iter__(self):
        """Return a generator over (start, end) pairs."""
        cdef Py_ssize_t i
        for i in range(self.start.shape[0]):
            yield (self.start[i], self.end[i])
