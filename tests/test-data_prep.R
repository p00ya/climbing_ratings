# Goldens test for 01-data_prep.R

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


context("Tests for 01-data_prep.R")

src_dir <- "testdata"
data_dir <- tempfile("dir")
now <- 1582243200L # 2020-02-21

setup({
  # Create temporary directory and copy testdata data there.
  dir.create(data_dir)
  file.copy(
    file.path(src_dir, "raw_ascents.csv"),
    file.path(data_dir)
  )

  source("../01-data_prep.R", local = TRUE)
})

teardown({
  suppressWarnings(unlink(data_dir, recursive = TRUE))
})

describe("data_prep_json", {
  it("ascents.csv matches golden", {
    ExpectCsvsEqual("ascents.csv", src_dir, data_dir)
  })
  it("pages.csv matches golden", {
    ExpectCsvsEqual("pages.csv", src_dir, data_dir)
  })
  it("routes.csv matches golden", {
    ExpectCsvsEqual("routes.csv", src_dir, data_dir)
  })
})
