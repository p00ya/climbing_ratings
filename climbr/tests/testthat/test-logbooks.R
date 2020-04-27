# Unit tests for logbooks.R

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


context("Tests for logbooks.R")


describe(".GetPitchCount", {
  it("with no pitch info", {
    expect_identical(.GetPitchCount(""), NA_integer_)
    expect_identical(.GetPitchCount("just a comment"), NA_integer_)
    expect_identical(.GetPitchCount("multiline\ncomment"), NA_integer_)
    expect_identical(.GetPitchCount("1 :"), NA_integer_)
  })
  it("with comments after pitch info", {
    expect_identical(
      .GetPitchCount("comment\n1: lead by me\ncomment"), NA_integer_
    )
  })
  it("with pitch info", {
    expect_identical(.GetPitchCount("1:"), 1L)
    expect_identical(.GetPitchCount("1:[18]"), 1L)
    expect_identical(.GetPitchCount("1:[ABC]"), 1L)
    expect_identical(.GetPitchCount("1: lead by me"), 1L)
    expect_identical(.GetPitchCount("1:[18] lead by me"), 1L)
    expect_identical(.GetPitchCount("comment\n1:"), 1L)
  })
  it("with multiple pitches", {
    expect_identical(.GetPitchCount("1:\n2:"), 2L)
    expect_identical(.GetPitchCount("1: lead by me\n2: lead by me"), 2L)
    expect_identical(.GetPitchCount("comment\n1:\n2:"), 2L)
  })
  it("with missing pitches", {
    # The pitch numbers may be sparse, e.g. if users didn't log linked pitches.
    expect_identical(.GetPitchCount("2:"), 1L)
    expect_identical(.GetPitchCount("1:\n3:"), 3L)
  })
})

describe(".ParseLogbook", {
  it("extracts raw ascents", {
    # This is a subset of the fields in the actual logbook exports.
    df <- data.frame(
      Ascent.ID = "4294967296",
      Ascent.Type = "Onsight",
      Route.ID = "8589934592",
      Route.Grade = "18",
      Comment = "",
      Ascent.Date = "2019-07-21T00:00:00Z",
      Log.Date = "2019-07-22T01:23:45Z",
      stringsAsFactors = FALSE
    )
    raw <- data.frame(
      ascentId = "4294967296",
      route = "8589934592",
      climber = "me",
      tick = "onsight",
      grade = 18L,
      timestamp = 1563667200L,
      style = 0L,
      pitches = NA_integer_,
      stringsAsFactors = FALSE
    )
    expect_equal(.ParseLogbook(df, "me"), raw)
  })
  it("drops bad grades", {
    df <- data.frame(
      Ascent.ID = "4294967296",
      Ascent.Type = "Onsight",
      Route.ID = "8589934592",
      Route.Grade = "V8",
      Comment = "",
      Ascent.Date = "2019-07-21T00:00:00Z",
      Log.Date = "2019-07-22T01:23:45Z",
      stringsAsFactors = FALSE
    )
    raw <- data.frame(
      ascentId = character(),
      route = character(),
      climber = character(),
      tick = character(),
      grade = integer(),
      timestamp = integer(),
      style = integer(),
      pitches = integer(),
      stringsAsFactors = FALSE
    )
    expect_equal(.ParseLogbook(df, "me"), raw)
  })
  it("with pitches", {
    df <- data.frame(
      Ascent.ID = c("4294967296", "4294967297"),
      Ascent.Type = c("Redpoint", "Redpoint"),
      Route.ID = c("8589934592", "8589934592"),
      Route.Grade = c("18", "18"),
      Comment = c("", "1: lead by me\n2: lead by you"),
      Ascent.Date = c("2019-07-21T00:00:00Z", "2019-07-21T00:00:00Z"),
      Log.Date = c("2020-01-01T01:23:45Z", "2020-01-01T01:23:45Z"),
      stringsAsFactors = FALSE
    )
    raw <- data.frame(
      ascentId = c("4294967296", "4294967297"),
      route = c("8589934592", "8589934592"),
      climber = c("me", "me"),
      tick = c("redpoint", "redpoint"),
      grade = c(18L, 18L),
      timestamp = c(1563667200L, 1563667200L),
      style = c(0L, 0L),
      pitches = c(NA, 2L),
      stringsAsFactors = FALSE
    )
    expect_equal(.ParseLogbook(df, "me"), raw)
  })
  it("orders by log date", {
    df <- data.frame(
      Ascent.ID = c("4294967296", "4294967297"),
      Ascent.Type = c("Onsight", "Onsight"),
      Route.ID = c("8589934592", "8589934593"),
      Route.Grade = c("18", "19"),
      Comment = c("", ""),
      Ascent.Date = c("2019-07-21T00:00:00Z", "2019-07-21T00:00:00Z"),
      # Row 1 was logged after row 2.
      Log.Date = c("2020-01-01T01:23:45Z", "2019-07-22T01:23:45Z"),
      stringsAsFactors = FALSE
    )
    raw <- data.frame(
      ascentId = c("4294967297", "4294967296"),
      route = c("8589934593", "8589934592"),
      climber = c("me", "me"),
      tick = c("onsight", "onsight"),
      grade = c(19L, 18L),
      timestamp = c(1563667200L, 1563667200L),
      style = c(0L, 0L),
      pitches = c(NA_integer_, NA_integer_),
      stringsAsFactors = FALSE
    )
    expect_equal(.ParseLogbook(df, "me"), raw)
    # Reverse the input row-order; output order should be the same.
    df <- df[order(nrow(df):1), ]
    expect_equal(.ParseLogbook(df, "me"), raw)
  })
})
