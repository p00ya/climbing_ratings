# Unit tests for json.R

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


context("Tests for json_utils.cpp")


describe("FlattenInt", {
  it("converts lists to atomic integer vectors", {
    expect_equal(
      FlattenInt(list(list(1L, 2L), NULL, list(3L))),
      c(1L, NA, 3L)
    )
  })
  it("applies idx", {
    expect_equal(
      FlattenInt(list(list(1L, 10L)), 2L),
      10L
    )
  })
  it("returns integers unmodified", {
    expect_equal(FlattenInt(list(1L), 1L), 1L)
  })
  it("replaces NULL values with NA", {
    expect_equal(FlattenInt(list(NULL), 1L), as.integer(NA))
  })
  it("errors for non-positive idx", {
    expect_error(FlattenInt(list(list(1L)), 0L))
  })
  it("warns for out-of-bounds idx", {
    expect_warning(FlattenInt(list(list(1L)), 2L))
  })
})

describe("FlattenChr", {
  it("converts lists to atomic character vectors", {
    expect_equal(
      FlattenChr(list(list("a", "b"), NULL, list("c"))),
      c("a", NA, "c")
    )
  })
  it("applies idx", {
    expect_equal(
      FlattenChr(list(list("a", "b")), 2L),
      "b"
    )
  })
  it("returns characters unmodified", {
    expect_equal(FlattenChr(list("a"), 1L), "a")
  })
  it("replaces NULL values with NA", {
    expect_equal(FlattenChr(list(NULL), 1L), as.character(NA))
  })
  it("errors for non-positive idx", {
    expect_error(FlattenChr(list(list("a")), 0L))
  })
  it("warns for out-of-bounds idx", {
    expect_warning(FlattenChr(list(list("a")), 2L))
  })
})
