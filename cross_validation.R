# Model cross-validation.

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

# Set these global parameters before running this script:
# data_dir <- "data"
# period_length <- 604800 # seconds in 1 week


library(caret)
library(climbr)

df_raw <- ReadAllJsonAscents(data_dir)
dfs <- NormalizeTables(CleanAscents(df_raw), period_length)

set.seed(1337L)

# Evaluate model with 10-fold cross-validation.
train_result <- train(
  dfs$ascents,
  factor(dfs$ascents$clean, levels = c(0, 1), labels = c("ATTEMPT", "CLEAN")),
  method = MakeWhrModel(dfs, max_iterations = 256L),
  tuneGrid = expand.grid(
    w = 1:3 / 52,
    sigma_c = (1:4)^2,
    sigma_r = (1:4)^2,
    b = c(0, 0.02),
    g0 = 259
  ),
  metric = "logLoss",
  trControl = trainControl(
    method = "adaptive_cv", repeats = 1, verboseIter = TRUE,
    summaryFunction = multiClassSummary,
    classProbs = TRUE,
    adaptive = list(min = 2, alpha = 0.05, method = "gls", complete = FALSE)
  )
)

print(train_result)
