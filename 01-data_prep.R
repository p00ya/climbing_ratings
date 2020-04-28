# Normalize a table of ascents into ascent, page and route tables.

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


library(climbr)

period_length <- 604800L # seconds in 1 week
if (!exists("now")) {
  now <- Sys.time()
}

# Read in the ascents table.
df_raw <- read.csv(
  file.path(data_dir, "raw_ascents.csv"),
  comment.char = "#",
  colClasses = c(
    ascentId = "character",
    route = "factor",
    climber = "factor",
    tick = "factor",
    grade = "integer",
    timestamp = "integer"
  ),
  stringsAsFactors = FALSE
)

df_clean <- CleanAscents(df_raw, max_time = now)
message(SummarizeAscents(df_clean))
dfs <- NormalizeTables(df_clean, period_length)
dfs$routes <- dplyr::mutate(dfs$routes, rating = TransformGrade(grade))
WriteNormalizedTables(dfs, data_dir)
