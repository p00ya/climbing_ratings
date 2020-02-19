# Unit tests for 00-data_prep_functions.R

# Copyright 2020 Dean Scarff
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


context("Tests for 00-data_prep_functions.R")

source("../00-data_prep_functions.R")

describe("NormalizeAscentType", {
  it("converts to lower-case", {
    expect_equal(
      NormalizeAscentType(c("TopRope", "Clean")),
      c("toprope", "clean")
    )
  })
  it("removes spaces", {
    expect_equal(
      NormalizeAscentType(c("top rope", "top rope solo")),
      c("toprope", "topropesolo")
    )
  })
  it("rewrites hangdog to dog", {
    expect_equal(
      NormalizeAscentType("Hang Dog"),
      "dog"
    )
  })
})

describe("TransformGrade", {
  it("applies scaling", {
    expect_equal(
      TransformGrade(0:2, 1, 0),
      exp(0:2)
    )
  })
  it("applies ref translation", {
    expect_equal(
      TransformGrade(0:2, 1, 1),
      exp(-1:1)
    )
  })
})
