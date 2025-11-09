"""Package setup for climbing_ratings"""

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

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


# setuptools.Extension kwargs for building Cython extensions.
cython_ext = {
    "extra_compile_args": [
        "-Ofast",
        "-ffast-math",
        "-fno-math-errno",
        "-Wno-deprecated-declarations",
    ],
    "define_macros": [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    "include_dirs": [numpy.get_include()],
}

csum = Extension(
    "climbing_ratings.csum",
    ["climbing_ratings/csum.pyx"],
    # Disable some maths optimizations that defeat precision-preserving
    # ordering.
    extra_compile_args=(cython_ext["extra_compile_args"] + ["-fno-associative-math"]),
    define_macros=cython_ext["define_macros"],
)

bradley_terry = Extension(
    "climbing_ratings.bradley_terry",
    ["climbing_ratings/bradley_terry.pyx"],
    **cython_ext,
)

derivatives = Extension(
    "climbing_ratings.derivatives",
    ["climbing_ratings/derivatives.pyx"],
    **cython_ext,
)

slices = Extension(
    "climbing_ratings.slices",
    ["climbing_ratings/slices.pyx"],
    **cython_ext,
)

if __name__ == "__main__":
    # See also pyproject.toml.
    setup(
        ext_modules=cythonize(
            [csum, bradley_terry, derivatives, slices],
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "cdivision": True,
                "embedsignature": True,
                "initializedcheck": False,
                "nonecheck": False,
                "wraparound": False,
            },
        ),
    )
