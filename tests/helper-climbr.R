# testthat helper for loading the climbr package.

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

devtools::load_all("../climbr")
library(climbr)

#' Asserts that two CSV files have identical contents.
#'
#' @param filename character basename of both CSV files.
#' @param src_dir character directory containing the golden file.
#' @param data_dir character directory containg the test file.
ExpectCsvsEqual <- function(filename, src_dir, data_dir) {
  actual_file <- file.path(data_dir, filename)
  expected_file <- file.path(src_dir, filename)
  expect_equal(read.csv(!!actual_file), read.csv(!!expected_file))
}
