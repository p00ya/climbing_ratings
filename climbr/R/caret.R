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


#' Runs the Python estimation script.
#'
#' @param w2_c climber Wiener variance.
#' @param w2_s climber-style Wiener variance.
#' @param sigma2_c climber variance.
#' @param sigma2_r route variance.
#' @param sigma2_s climber-style variance.
#' @param max_iterations integer maximum number of WHR iterations.
#' @param data_dir directory to read and write CSV files.
#' @param script arguments to the Python interpreter to run the Python
#' estimation script.
#' @return exit status of the estimation script.
#' @keywords internal
.RunEstimationScript <- function(w2_c, w2_s, sigma2_c, sigma2_r, sigma2_s,
                                 max_iterations, data_dir,
                                 script = c("-m", "climbing_ratings")) {
  system2(
    "python3",
    c(
      script,
      "--wiener-variance", w2_c,
      "--style-wiener-variance", w2_s,
      "--climber-prior-variance", sigma2_c,
      "--route-prior-variance", sigma2_r,
      "--style-prior-variance", sigma2_s,
      "--max-iterations", max_iterations,
      data_dir
    )
  )
}

#' Creates a caret model object for fitting climbing ratings.
#'
#' This function creates a closure that makes the complete normalized tables
#' available to the prediction function.  This is a workaround for caret's
#' interface assuming subsets of the observations can be fit independently.
#' Caret's interface doesn't work well with the estimation script because
#' resampling the ascents means that missing parameters (routes and ascents not
#' in the sample) won't be estimated.
#'
#' @param dfs_full list of data frames (ascents, pages, routes).
#' @param max_iterations integer maximum number of WHR iterations.
#' @param threshold the minimum predicted probability of a clean ascent to
#' classify as CLEAN.
#' @param run_script function to run the estimation script; only exposed for
#' testing purposes.
#' @return model that can be passed to [caret::train()].
MakeWhrModel <- function(dfs_full, max_iterations = 64L, threshold = 0.5,
                         run_script = .RunEstimationScript) {
  # Columns expected to be in an ascents (explanatory variables) table.
  ascents_columns <- c("route", "page", "style_page")

  # Returns the probability each of the given ascents is clean, according
  # to the estimates in "ratings".
  #
  # @param ratings list of data frames (routes, pages, style_pages).
  # @param ascents ascents data frame.
  # @return numeric vector of the probability of clean ascents.
  .ProbClean <- function(ratings, ascents) {
    stopifnot(all(c("routes", "pages", "style_pages") %in% names(ratings)))
    stopifnot(all(ascents_columns %in% colnames(ascents)))
    dfs <- dfs_full
    dfs$ascents <- ascents
    dfs <- MergeWithRatings(dfs, ratings)
    PredictBradleyTerry(dfs)
  }

  # Implements the fit component of a caret model.
  #
  # @return a list of page and route rating data frames.
  .Fit <- function(x, y, wts = NULL, param, ...) {
    stopifnot(all(ascents_columns %in% colnames(x)))

    # Keep the full route and page tables, so that the estimated ratings
    # tables are always aligned.  It's possible some routes and pages may
    # have no ascents.
    dfs <- dfs_full
    dfs$routes <- dplyr::mutate(
      dfs$routes,
      rating = TransformGrade(.data$grade, param$b, param$g0)
    )
    dfs$ascents <- x

    # Write normalized tables to a temporary directory, run the estimation,
    # then return the results.
    data_dir <- tempfile(pattern = "climbing_ratings-")
    tryCatch(
      {
        dir.create(data_dir)
        WriteNormalizedTables(dfs, data_dir)
        status <- run_script(
          w2_c = param$w2_c,
          w2_s = param$w2_s,
          sigma2_c = param$sigma2_c,
          sigma2_r = param$sigma2_r,
          sigma2_s = param$sigma2_s,
          max_iterations,
          data_dir
        )
        if (status != 0) stop("Error executing estimation script")
        ReadRatings(data_dir)
      },
      finally = {
        suppressWarnings(unlink(data_dir, recursive = TRUE))
      }
    )
  }

  # See the "Using your own model in train" chapter of the caret manual.
  list(
    label = "Climbing Ratings (Whole History Rating)",
    library = NULL,
    loop = NULL,
    parameters = data.frame(
      parameter = c(
        "w2_c", "w2_s", "sigma2_c", "sigma2_r", "sigma2_s", "b", "g0"
      ),
      class = rep("numeric", 7),
      label = c(
        "Climber Wiener variance",
        "Climber-style Wiener variance",
        "Climber ratings variance",
        "Route ratings variance",
        "Climber-style ratings variance",
        "Grade scaling",
        "Reference grade"
      ),
      stringsAsFactors = FALSE
    ),
    grid = NULL,
    type = "Classification",
    fit = .Fit,
    prob = function(modelFit, newdata, preProc = NULL, submodels = NULL) {
      clean <- .ProbClean(modelFit, newdata)
      data.frame(ATTEMPT = 1 - clean, CLEAN = clean)
    },
    predict = function(modelFit, newdata, preProc = NULL, submodels = NULL) {
      clean <- .ProbClean(modelFit, newdata)
      ifelse(clean >= threshold, "CLEAN", "ATTEMPT")
    },
    sort = identity
  )
}
