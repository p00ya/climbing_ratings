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

describe("ExpandPitches", {
  it("copies no-pitch ascent", {
    df <- data.frame(
      ascentId = "1",
      route = "2",
      tick = "onsight",
      climber = "3",
      timestamp = 5L,
      grade = 7L,
      style = "Sport",
      pitch = I(list(NULL))
    )
    expect_equal(
      ExpandPitches(df),
      data.frame(
        ascentId = "1",
        route = "2",
        tick = "onsight",
        climber = "3",
        timestamp = 5L,
        grade = 7L,
        style = "Sport"
      )
    )
  })

  it("drops pitch with no tick", {
    df <- data.frame(
      ascentId = "1",
      route = "2",
      tick = "onsight",
      climber = "3",
      timestamp = 5L,
      grade = 7L,
      style = "Sport",
      pitch = I(list(list(list("1", NULL))))
    )
    expect_equal(
      ExpandPitches(df),
      data.frame(
        ascentId = character(),
        route = character(),
        tick = character(),
        climber = character(),
        timestamp = integer(),
        grade = integer(),
        style = character()
      )
    )
  })

  it("drops pitch with no number", {
    df <- data.frame(
      ascentId = "1",
      route = "2",
      tick = "onsight",
      climber = "3",
      timestamp = 5L,
      grade = 7L,
      style = "Sport",
      pitch = I(list(list(list(NULL, NULL))))
    )
    expect_equal(
      ExpandPitches(df),
      data.frame(
        ascentId = character(),
        route = character(),
        tick = character(),
        climber = character(),
        timestamp = integer(),
        grade = integer(),
        style = character()
      )
    )
  })

  it("translates single pitch and tick", {
    df <- data.frame(
      ascentId = "1",
      route = "2",
      tick = "onsight",
      climber = "3",
      timestamp = 5L,
      grade = 7L,
      style = "Sport",
      pitch = I(list(list(list("1", list("onsight")))))
    )
    expect_equal(
      ExpandPitches(df),
      data.frame(
        ascentId = "1P1",
        route = "2P1",
        tick = "onsight",
        climber = "3",
        timestamp = 5L,
        grade = 7L,
        style = "Sport"
      )
    )
  })

  it("expands one ascent multiple pitches", {
    df <- data.frame(
      ascentId = "1",
      route = "2",
      tick = "onsight",
      climber = "3",
      timestamp = 5L,
      grade = 7L,
      style = "Sport",
      pitch = I(list(list(
        list("1", list("onsight")),
        list("2", list("dog"))
      )))
    )
    expect_equal(
      ExpandPitches(df),
      data.frame(
        ascentId = c("1P1", "1P2"),
        route = c("2P1", "2P2"),
        tick = c("onsight", "dog"),
        climber = c("3", "3"),
        timestamp = c(5L, 5L),
        grade = c(7L, 7L),
        style = c("Sport", "Sport")
      )
    )
  })

  it("expands multipitch and no-pitch ascents", {
    df <- data.frame(
      ascentId = c("1", "21"),
      route = c("2", "22"),
      tick = c("onsight", "redpoint"),
      climber = c("3", "23"),
      timestamp = c(5L, 25L),
      grade = c(7L, 27L),
      style = c("Sport", "Sport"),
      pitch = I(list(
        list(list("1", list("onsight")), list("2", list("dog"))),
        NULL
      ))
    )
    expect_equal(
      ExpandPitches(df),
      data.frame(
        ascentId = c("1P1", "1P2", "21"),
        route = c("2P1", "2P2", "22"),
        tick = c("onsight", "dog", "redpoint"),
        climber = c("3", "3", "23"),
        timestamp = c(5L, 5L, 25L),
        grade = c(7L, 7L, 27L),
        style = c("Sport", "Sport", "Sport")
      )
    )
  })
})
