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


describe(".ParseLogbook", {
  it("extracts raw ascents", {
    # This is a subset of the fields in the actual logbook exports.
    df <- data.frame(
      Ascent.ID = "4294967296",
      Ascent.Type = "Onsight",
      Route.ID = "8589934592",
      Route.Grade = "18",
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
      stringsAsFactors = FALSE
    )
    expect_equal(.ParseLogbook(df, "me"), raw)
  })
})
