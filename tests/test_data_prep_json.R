# Goldens test for 01-data_prep_json.R

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


context("Tests for 01-data_prep_json.R")

suppressMessages(source("../00-data_prep_functions.R"))

src_dir <- "testdata/json"
data_dir <- tempfile("dir")

setup({
  # Create temporary directory and copy testdata data there.
  dir.create(data_dir)
  file.copy(
    file.path(src_dir, "ascents-001.json"),
    file.path(data_dir)
  )

  source("../01-data_prep_json.R", local = TRUE)
})

teardown({
  suppressWarnings(unlink(data_dir, recursive = TRUE))
})

# Asserts that two CSV files both named "filename" in the src_dir and data_dir
# have identical contents.
ExpectCsvsEqual <- function(filename) {
  actual_file <- file.path(data_dir, filename)
  expected_file <- file.path(src_dir, filename)
  actual <- read.csv(actual_file)
  expected <- read.csv(expected_file)
  expect_equal(actual, expected)
}

test_that("data_prep_json outputs match goldens", {
  ExpectCsvsEqual("ascents.csv")
  ExpectCsvsEqual("pages.csv")
  ExpectCsvsEqual("routes.csv")
})
