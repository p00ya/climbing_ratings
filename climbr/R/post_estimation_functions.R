# Functions for post-processing ratings estimates.

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


#' Returns the probability an ascent is clean according to the Bradley-Terry
#' model.
#'
#' @param dfs a list of data frames.
PredictBradleyTerry <- function(dfs) {
  page_gamma <- dfs$pages$gamma[dfs$ascents$page]
  route_gamma <- dfs$routes$gamma[dfs$ascents$route]
  page_gamma / (page_gamma + route_gamma)
}

#' Read ratings results from "dir".
#'
#' @param dir a directory containing two files:
#' - "page_ratings.csv" containing climber, rating, var and cov columns
#' - "route_ratings.csv" containing route, rating and var columns.
ReadRatings <- function(dir) {
  df_page_ratings <- utils::read.csv(file.path(dir, "page_ratings.csv"),
    colClasses = c("integer", "numeric", "numeric", "numeric")
  )

  df_route_ratings <- utils::read.csv(file.path(dir, "route_ratings.csv"),
    colClasses = c("factor", "numeric", "numeric")
  )

  list(routes = df_route_ratings, pages = df_page_ratings)
}

#' Updates the data frames in "dfs" with ratings results.
#'
#' @param dfs a list of data frames.
#' @param ratings a data frame.
MergeWithRatings <- function(dfs, ratings) {
  dfs$pages$r <- ratings$pages$rating
  dfs$pages$var <- ratings$pages$var
  dfs$pages$cov <- ratings$pages$cov

  dfs$routes$r <- ratings$routes$rating
  dfs$routes$var <- ratings$routes$var

  dfs$routes$gamma <- exp(dfs$routes$r)
  dfs$pages$gamma <- exp(dfs$pages$r)
  dfs$ascents$predicted <- PredictBradleyTerry(dfs)
  dfs
}

#' Plots the timeseries of rating estimates for a set of climbers.
#'
#' @param df_pages data frame of pages.
#' @param friends a named vector where the names are the climber levels and
#' the values are corresponding labels to apply in the plot.
#' @param level the width of the intervals.
#' @param wsq the Wiener variance per second (should be consistent with the
#' estimation --wiener-variance flag, but potentially with different units).
PlotProgression <- function(df_pages, friends, level = 0.5,
                            wsq = 1 / (86400 * 7 * 52)) {
  df_friends <- df_pages %>%
    dplyr::filter(.data$climber %in% names(friends)) %>%
    transform(
      date = as.POSIXct(.data$timestamp, origin = "1970-01-01"),
      climber = dplyr::recode(.data$climber, !!!friends)
    ) %>%
    dplyr::select(.data$date, .data$climber, .data$r, .data$var, .data$cov)

  ggplot2::ggplot(
    df_friends,
    ggplot2::aes(.data$date, .data$r, color = .data$climber)
  ) +
    ggplot2::geom_point() +
    stat_wiener_smooth(
      ggplot2::aes(var = .data$var, cov = .data$cov, fill = .data$climber),
      wsq = wsq, # consistent with
      level = level,
      n = 1000L
    )
}

#' Plots (1 - alpha) confidence intervals for a set of routes.
#'
#' @param df_routes a data frame.
#' @param selected a named vector where the names are the route levels and
#' the values are corresponding labels to apply in the plot.
#' @param alpha the significance level.
PlotSelectedRoutes <- function(df_routes, selected, alpha = 0.05) {
  df <- df_routes %>%
    dplyr::filter(.data$route %in% names(selected)) %>%
    transform(
      error = qnorm(alpha / 2, sd = sqrt(.data$var)),
      route = dplyr::recode(.data$route, !!!selected)
    )
  ggplot2::ggplot(df, ggplot2::aes(.data$r, .data$route)) +
    ggplot2::geom_point() +
    ggplot2::geom_errorbarh(ggplot2::aes(
      xmin = .data$r - .data$error,
      xmax = .data$r + .data$error
    ))
}

#' Generates outlier labels for the route plot.
#'
#' @param df_routes a data frame.
#' @param alpha the significance level.
GetOutliers <- function(df_routes, alpha = 0.05) {
  m <- loess(r ~ grade, df_routes, weights = 1 / df_routes$var)
  e <- residuals(m)
  e <- e / sd(e)
  q <- qnorm(1 - alpha / 2)
  ifelse(
    abs(e) > q & sqrt(df_routes$var) < q,
    as.character(df_routes$route),
    NA
  )
}

#' Plots conventional grades vs the estimated "natural rating" of routes.
#' Outliers are labeled.
#'
#' @param df_routes a data frame.
PlotRouteRating <- function(df_routes) {
  ggplot2::ggplot(
    df_routes %>% dplyr::mutate(outlier = GetOutliers(df_routes)),
    ggplot2::aes(.data$grade, .data$r)
  ) +
    ggplot2::geom_point(alpha = 0.1, size = 0.5, color = "red") +
    ggplot2::geom_smooth() +
    ggplot2::geom_text(ggplot2::aes(label = .data$outlier),
      na.rm = TRUE, size = 2, check_overlap = TRUE,
      hjust = 0, nudge_x = 0.1, vjust = "outward"
    )
}

#' Plots the ratings distribution for pages and routes.
#'
#' @param df_pages a data frame.
#' @param df_routes a data frame.
PlotRatingsDensity <- function(df_pages, df_routes) {
  page_ratings <- df_pages %>%
    dplyr::mutate(type = .data$climber) %>%
    dplyr::select(.data$r, .data$type)
  route_ratings <- df_routes %>%
    dplyr::mutate(type = .data$route) %>%
    dplyr::select(.data$r, .data$type)
  ratings <- dplyr::bind_rows(page_ratings, route_ratings) %>%
    dplyr::mutate(type = as.factor(.data$type))
  ggplot2::ggplot(ratings, ggplot2::aes(.data$r)) +
    ggplot2::facet_grid(rows = ggplot2::vars(.data$type)) +
    ggplot2::geom_density()
}

#' Plots precision vs recall given predicted probabilities of clean ascents.
#'
#' @param df_ascents a data frame.
PlotPrecisionRecall <- function(df_ascents) {
  sscurves <- precrec::evalmod(
    scores = df_ascents$predicted,
    labels = df_ascents$clean
  )
  ggplot2::autoplot(sscurves, "PRC", show_legend = FALSE)
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
