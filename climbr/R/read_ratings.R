# Functions for reading estimated ratings from the Python script.

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


#' Read ratings results from "dir".
#'
#' @param dir a directory containing two files:
#' - "page_ratings.csv" containing climber, rating, var and cov columns
#' - "route_ratings.csv" containing route, rating and var columns.
ReadRatings <- function(dir) {
  df_page_ratings <- utils::read.csv(file.path(dir, "page_ratings.csv"),
    colClasses = c("integer", "numeric", "numeric", "numeric")
  )

  df_style_page_ratings <- utils::read.csv(
    file.path(dir, "style_page_ratings.csv"),
    colClasses = c("integer", "numeric", "numeric", "numeric")
  )

  df_route_ratings <- utils::read.csv(file.path(dir, "route_ratings.csv"),
    colClasses = c("factor", "numeric", "numeric"),
    stringsAsFactors = FALSE
  )

  list(
    routes = df_route_ratings,
    pages = df_page_ratings,
    style_pages = df_style_page_ratings
  )
}

#' Updates the data frames in "dfs" with ratings results.
#'
#' @param dfs a list of data frames.
#' @param ratings a data frame.
MergeWithRatings <- function(dfs, ratings) {
  dfs$pages$r <- ratings$pages$rating
  dfs$pages$var <- ratings$pages$var
  dfs$pages$cov <- ratings$pages$cov

  dfs$style_pages$r <- ratings$style_pages$rating
  dfs$style_pages$var <- ratings$style_pages$var
  dfs$style_pages$cov <- ratings$style_pages$cov

  dfs$routes$r <- ratings$routes$rating
  dfs$routes$var <- ratings$routes$var

  dfs$routes$gamma <- exp(dfs$routes$r)
  dfs$pages$gamma <- exp(dfs$pages$r)
  dfs$style_pages$gamma <- exp(dfs$style_pages$r)
  dfs$ascents$predicted <- PredictBradleyTerry(dfs)
  dfs
}

#' Returns the probability an ascent is clean according to the Bradley-Terry
#' model.
#'
#' @param dfs a list of data frames.
PredictBradleyTerry <- function(dfs) {
  # Ascents may be missing styles (style_page = 0L), so shift the indices
  # and create a "default" style-page with natural rating 0.
  style_r <- c(0, dfs$style_pages$r)[dfs$ascents$style_page + 1L]

  # Combine base page and style-page ratings.
  climber_gamma <- exp(dfs$pages$r[dfs$ascents$page] + style_r)

  route_gamma <- dfs$routes$gamma[dfs$ascents$route]
  climber_gamma / (climber_gamma + route_gamma)
}


#' Returns the contingency table for a binary classifier
#'
#' @param df_ascents a data frame.
#' @param threshold the maximum probability of a clean ascent to predict as an
#' ATTEMPT.
GetContingencyTable <- function(df_ascents, threshold = 0.5) {
  table(
    factor(df_ascents$clean, labels = c("ATTEMPT", "CLEAN")),
    factor(df_ascents$predicted > threshold, labels = c("pATTEMPT", "pCLEAN"))
  )
}
