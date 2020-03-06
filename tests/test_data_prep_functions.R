# Unit tests for 00-data_prep_functions.R

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

describe("AsIntegerOrNA", {
  it("applies idx", {
    expect_equal(AsIntegerOrNA(list(1L, 10L), 2), 10L)
  })
  it("returns integers unmodified", {
    expect_equal(AsIntegerOrNA(1L), 1L)
  })
  it("replaces NULL values with NA", {
    expect_equal(AsIntegerOrNA(NULL), NA)
  })
})

describe("AsCharacterOrNA", {
  it("applies idx", {
    expect_equal(AsCharacterOrNA(list("a", "b"), 2), "b")
  })
  it("returns characters unmodified", {
    expect_equal(AsCharacterOrNA("a"), "a")
  })
  it("replaces NULL values with NA", {
    expect_equal(AsCharacterOrNA(NULL), NA)
  })
})

test_that("FlattenInt converts lists to atomic integer vectors", {
  expect_equal(FlattenInt(list(list(1L, 2L), NULL, list(3L))), c(1L, NA, 3L))
})

test_that("FlattenChr converts lists to atomic character vectors", {
  expect_equal(
    FlattenChr(list(list("a", "b"), NULL, list("c"))),
    c("a", NA, "c")
  )
})
