# Functions for plotting ratings results.

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


#' Plots the timeseries of rating estimates for a set of climbers.
#'
#' @param df_pages data frame of pages.
#' @param friends a named vector where the names are the climber levels and
#' the values are corresponding labels to apply in the plot.  The order of the
#' labels in the plot's legend will match the order in the friends vector.
#' @param level the width of the intervals.
#' @param wsq the Wiener variance per second (should be consistent with the
#' estimation `--wiener-variance` flag, but potentially with different units).
PlotProgression <- function(df_pages, friends, level = 0.5,
                            wsq = 1 / (86400 * 7 * 52)) {
  df_friends <- df_pages %>%
    dplyr::filter(.data$climber %in% names(friends)) %>%
    dplyr::mutate(
      date = as.POSIXct(.data$timestamp, origin = "1970-01-01"),
      climber = dplyr::recode_factor(.data$climber, !!!friends),
      sigma = sqrt(.data$var) * qnorm((1 - level) / 2, lower.tail = FALSE)
    ) %>%
    dplyr::select(
      .data$date, .data$climber, .data$r, .data$var, .data$cov, .data$sigma
    )

  ggplot2::ggplot(
    df_friends,
    ggplot2::aes(
      .data$date,
      .data$r,
      colour = .data$climber,
      fill = .data$climber,
    )
  ) +
    ggplot2::geom_linerange(
      alpha = 0.5,
      ggplot2::aes(ymin = .data$r - .data$sigma, ymax = .data$r + .data$sigma)
    ) +
    stat_wiener_smooth(
      ggplot2::aes(var = .data$var, cov = .data$cov),
      wsq = wsq,
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
    dplyr::mutate(
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

#' Plots the rating PDFs for each route.
#'
#' This will only produce a nice graph for a small number of routes.
#'
#' @param df_routes a data frame with route, r and var columns.
#' @param alpha significance level for bounds of the graph.
PlotRoutesProbabilityDensity <- function(df_routes, alpha = 0.001) {
  df <- df_routes %>% dplyr::transmute(
    route = .data$route,
    mean = .data$r,
    sd = sqrt(.data$var)
  )

  # Adds a geom for the route's rating PDF to the given plot.
  AddPDFGeom <- function(plot, row) {
    plot + ggplot2::geom_function(
      fun = stats::dnorm,
      ggplot2::aes(colour = row$route),
      args = list(mean = row$mean, sd = row$sd)
    )
  }

  minx <- min(stats::qnorm(alpha / 2, df$mean, df$sd))
  maxx <- max(stats::qnorm(1 - alpha / 2, df$mean, df$sd))

  Reduce(
    AddPDFGeom,
    split(df, seq(nrow(df_routes))),
    init = ggplot2::ggplot() +
      ggplot2::xlim(minx, maxx) +
      ggplot2::labs(x = "rating", y = "probability density", col = "route")
  )
}

#' Plots the ratings distribution for pages and routes.
#'
#' @param df_pages a data frame.
#' @param df_routes a data frame.
PlotRatingsDensity <- function(df_pages, df_routes) {
  page_ratings <- df_pages %>%
    dplyr::mutate(type = "climber") %>%
    dplyr::select(.data$r, .data$type)
  route_ratings <- df_routes %>%
    dplyr::mutate(type = "route") %>%
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
