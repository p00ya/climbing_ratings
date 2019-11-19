# Goldens test for 01-data_prep.R

# Copyright 2019 Dean Scarff
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


context("Tests for 01-data_prep.R")

test_that("data_prep outputs match goldens", {
  # Check outputs.
  ExpectCsvsEqual <- function(filename) {
    actual_file <- file.path(data_dir, filename)
    expected_file <- file.path(src_dir, filename)
    actual <- read.csv(actual_file)
    expected <- read.csv(expected_file)
    expect_equal(actual, expected)
  }

  ExpectCsvsEqual("ascents.csv")
  ExpectCsvsEqual("pages.csv")
  ExpectCsvsEqual("routes.csv")
  suppressWarnings(unlink(data_dir, recursive = TRUE))
})
