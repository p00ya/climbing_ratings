# Model cross-validation.

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

# Set these global parameters before running this script:
# json_dir <- "json"
# period_length <- 604800 # seconds in 1 week

library(caret)

source("00-data_prep_functions.R")
source("00-post_estimation_functions.R")

# Creates a model object for fitting climbing ratings.
#
# The return value can be passed as the "model" parameter to caret::train.
#
# This function creates a closure that makes the complete normalized tables
# available to the prediction function.  This is a workaround for caret's
# interface assuming subsets of the observations can be fit independently.
# This doesn't work well with the estimation script because resampling the
# ascents means that missing parameters (routes and ascents not in the sample)
# won't be estimated.
MakeWhrModel <- function(dfs_full) {
  # Columns expected to be in an ascents (explanatory variables) table.
  ascents_columns <- c("route", "page")

  # Returns the probability each of the given ascents is clean, according
  # to the estimates in "ratings".
  ProbClean <- function(ratings, ascents) {
    stopifnot(all(c("routes", "pages") %in% names(ratings)))
    stopifnot(all(ascents_columns %in% colnames(ascents)))
    dfs <- dfs_full
    dfs$ascents <- ascents
    dfs <- MergeWithRatings(dfs, ratings)
    PredictBradleyTerry(dfs)
  }

  # Returns a list of page and route rating data frames.
  Fit <- function(x, y, wts = NULL, param, ...) {
    stopifnot(all(ascents_columns %in% colnames(x)))

    # Keep the full route and page tables, so that the estimated ratings
    # tables are always aligned.  It's possible some routes and pages may
    # have no ascents.
    dfs <- dfs_full
    dfs$routes <- mutate(
      dfs$routes,
      grade = TransformGrade(ewbank, as.numeric(param["b"]))
    )
    dfs$ascents <- x

    # Write normalized tables to a temporary directory, run the estimation,
    # then return the results.
    data_dir <- tempfile(pattern = "climbing_ratings-")
    tryCatch({
      dir.create(data_dir)
      WriteNormalizedTables(dfs, data_dir)
      status <- system2(
        "python3",
        c(
          "./02-run_estimation.py",
          "--wiener-variance", param["w"],
          "--gamma-shape", param["k"],
          data_dir
        )
      )
      if (status != 0) stop("Error executing estimation script")

      ReadRatings(data_dir)
    }, finally = {
      suppressWarnings(unlink(data_dir, recursive = TRUE))
    })
  }

  list(
    label = "Climbing Ratings (Whole History Rating)",
    library = NULL,
    loop = NULL,
    # caret doesn't allow empty parameters; create a dummy parameter for
    # model validation.
    parameters = data.frame(
      parameter = c("w", "k", "b"),
      class = rep("numeric", 3),
      label = c(
        "Wiener variance",
        "Gamma shape",
        "Grade scaling"
      )
    ),
    grid = NULL,
    type = "Classification",
    fit = Fit,
    prob = function(modelFit, newdata, preProc = NULL, submodels = NULL) {
      clean <- ProbClean(modelFit, newdata)
      data.frame(ATTEMPT = 1 - clean, CLEAN = clean)
    },
    predict = function(modelFit, newdata, preProc = NULL, submodels = NULL) {
      clean <- ProbClean(modelFit, newdata)
      ifelse(clean >= 0.5, "CLEAN", "ATTEMPT")
    },
    sort = identity
  )
}

df_raw <- ReadAllJsonAscents(json_dir)
dfs <- NormalizeTables(CleanAscents(df_raw), period_length)

set.seed(1337)

# Evaluate model with 10-fold repeated cross-validation.
train_result <- train(
  dfs$ascents,
  factor(dfs$ascents$clean, levels = c(0, 1), labels = c("ATTEMPT", "CLEAN")),
  method = MakeWhrModel(dfs),
  tuneGrid = expand.grid(
    w = 1:3 / 52,
    k = seq(1.25, 2.5, by = 0.25),
    b = c(0, 0.15, 0.22)
  ),
  trControl = trainControl(
    method = "repeatedcv", repeats = 3, verboseIter = TRUE
  )
)

print(train_result)
