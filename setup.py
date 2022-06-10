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

bradley_terry = Extension(
    "climbing_ratings.bradley_terry",
    ["climbing_ratings/bradley_terry.pyx"],
    # Disable some maths optimizations that defeat precision-preserving
    # ordering.
    extra_compile_args=cython_ext["extra_compile_args"]
    + ["-fno-associative-math", "-fno-reciprocal-math"],
    define_macros=cython_ext["define_macros"],
    include_dirs=cython_ext["include_dirs"],
)

derivatives = Extension(
    "climbing_ratings.derivatives",
    ["climbing_ratings/derivatives.pyx"],
    **cython_ext,
)

long_description = """
climbing_ratings is a library for estimating ratings for the sport of rock
climbing.  The ratings can be used to predict route difficulty and climber
performance on a particular route.

The algorithms are based on the Whole-History Rating system."""

if __name__ == "__main__":
    setup(
        name="climbing_ratings",
        author="Dean Scarff",
        author_email=("dos" + chr(64) + "scarff.id.au"),
        description="Estimate climber and route ratings from ascents data",
        long_description=long_description,
        version="4.2.1",
        url="https://github.com/p00ya/climbing_ratings",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: POSIX",
            "Programming Language :: Cython",
            "Programming Language :: Python :: 3",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        license="Apache License 2.0",
        packages=["climbing_ratings", "climbing_ratings.tests"],
        include_package_data=False,
        package_data={"climbing_ratings": ["*.pyi", "*.pyx"]},
        platforms=["POSIX"],
        python_requires=">=3.7",
        install_requires=["numpy>=1.21"],
        setup_requires=["Cython", "numpy>=1.21"],
        test_suite="climbing_ratings.tests.test_suite",
        tests_require=["numpy", "pytest"],
        ext_modules=cythonize(
            [bradley_terry, derivatives],
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
