# Unit tests for data_prep.R

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


context("Tests for data_prep.R")


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
      0:2
    )
  })
  it("applies ref translation", {
    expect_equal(
      TransformGrade(0:2, 1, 1),
      -1:1
    )
  })
})

describe("CleanAscents", {
  it("adds clean column", {
    df_raw <- data.frame(
      ascentId = c("A", "B"),
      route = factor(c("R", "R")),
      climber = factor(c("C", "C")),
      tick = factor(c("clean", "dog")),
      grade = c(22L, 22L),
      timestamp = c(0L, 0L),
      style = c(1L, 1L)
    )

    expect_equal(
      CleanAscents(df_raw),
      data.frame(
        ascentId = c("A", "B"),
        route = factor(c("R", "R")),
        climber = factor(c("C", "C")),
        tick = factor(c("clean", "dog")),
        grade = c(22L, 22L),
        timestamp = c(0L, 0L),
        style = c(1L, 1L),
        clean = c(TRUE, FALSE)
      )
    )
  })

  it("drops routes with a single ascent", {
    df_raw <- data.frame(
      ascentId = "A",
      route = factor("R"),
      climber = factor("C"),
      tick = factor("dog"),
      grade = 22L,
      timestamp = 0L,
      style = 1L
    )

    expect_equal(nrow(CleanAscents(df_raw)), 0L)
  })

  it("drops climbers who only log ticks", {
    df_raw <- data.frame(
      ascentId = c("A", "B"),
      route = factor(c("R", "R")),
      climber = factor(c("C", "C")),
      tick = factor(c("clean", "clean")),
      grade = c(22L, 22L),
      timestamp = c(0L, 0L),
      style = c(1L, 1L)
    )

    expect_equal(nrow(CleanAscents(df_raw)), 0L)
  })

  it("removes rows with NA", {
    df_raw <- data.frame(
      ascentId = c("clean", "dog", "notick", "nograde", "notime", "nostyle"),
      route = factor(rep("R", 6)),
      climber = factor(rep("C", 6)),
      tick = factor(c("clean", "dog", NA, "dog", "dog", "dog")),
      grade = c(22L, 22L, 22L, NA, 22L, 22L),
      timestamp = c(0L, 0L, 0L, 0L, NA, 0L),
      style = c(1L, 1L, 1L, 1L, 1L, NA)
    )

    expect_equal(CleanAscents(df_raw)$ascentId, c("clean", "dog"))
  })

  it("removes rows outside time bracket", {
    df_raw <- data.frame(
      ascentId = c("clean", "dog", "tooearly", "toolate"),
      route = factor(rep("R", 4)),
      climber = factor(rep("C", 4)),
      tick = factor(rep("dog", 4)),
      grade = rep(22L, 4),
      timestamp = c(0L, 0L, -2L, 2L),
      style = rep(1L, 4)
    )

    expect_equal(CleanAscents(df_raw, -1, 2)$ascentId, c("clean", "dog"))
  })

  it("adds base style", {
    df_raw <- data.frame(
      ascentId = c("A", "B"),
      route = factor(c("R", "R")),
      climber = factor(c("C", "C")),
      tick = factor(c("clean", "dog")),
      grade = c(22L, 22L),
      timestamp = c(0L, 0L)
    )

    expect_equal(CleanAscents(df_raw)$style, c(1L, 1L))
  })
})
