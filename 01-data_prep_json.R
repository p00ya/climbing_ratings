# Convert theCrag API responses to ascent, page and route tables.

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

# Make sure to source("00-data_prep_functions.R") before sourcing this script.

period_length <- 604800 # seconds in 1 week

df_raw <- ReadAllJsonAscents(data_dir)
cat(
  prettyNum(nrow(df_raw), big.mark=","), "raw ascents by",
  prettyNum(nlevels(df_raw$climber), big.mark=","), "climbers, over",
  prettyNum(nlevels(df_raw$route), big.mark=","), "routes\n"
)

dfs <- NormalizeTables(CleanAscents(df_raw), period_length)
dfs$routes <- mutate(dfs$routes, grade = TransformGrade(ewbank))
WriteNormalizedTables(dfs, data_dir)
