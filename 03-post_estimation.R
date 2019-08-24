# Exploration of estimated ratings.

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

# Make sure to source("01-data_prep.R") before sourcing this script.

library(dplyr)
library(ggplot2)
library(grid)

# Returns the probability an ascent is clean according to the Bradley-Terry
# model.
PredictBradleyTerry <- function(dfs) {
  page_gamma <- dfs$pages$gamma[dfs$ascents$page]
  route_gamma <- dfs$routes$gamma[dfs$ascents$route]
  page_gamma / (page_gamma + route_gamma)
}

# Updates the data frames in "dfs" with ratings results from "dir".
MergeWithRatings <- function(dfs, dir) {
  df_page_ratings <- read.csv(file.path(dir, "page_ratings.csv"),
    colClasses = c("integer", "numeric", "numeric")
  )

  df_route_ratings <- read.csv(file.path(dir, "route_ratings.csv"),
    colClasses = c("factor", "numeric", "numeric")
  )

  dfs$pages$gamma <- df_page_ratings$gamma
  dfs$pages$var <- df_page_ratings$var
  dfs$routes$gamma <- df_route_ratings$gamma

  dfs$ascents$predicted <- PredictBradleyTerry(dfs)
  dfs$routes$r <- log(dfs$routes$gamma)
  dfs$pages$r <- log(dfs$pages$gamma)
  dfs
}

GetTimestampForPeriod <- function(t) {
  as.POSIXct(t * period_length + min(df_raw$timestamp), origin = "1970-01-01")
}

# Plots the timeseries of rating estimates for a set of climbers.  The "friends"
# parameter should be a named vector where the names are the climber levels and
# the values are corresponding labels to apply in the plot.
PlotProgression <- function(df_pages, friends) {
  df_friends <- df_pages %>%
    filter(climber %in% names(friends)) %>%
    mutate(
      date = GetTimestampForPeriod(t),
      climber = recode(climber, !!!friends),
      r = log(gamma),
      r_upper = r + qnorm(0.25) * sqrt(var),
      r_lower = r - qnorm(0.25) * sqrt(var)
    )
  ggplot(df_friends, aes(date, r, color = climber)) +
    geom_smooth(aes(ymin = r_lower, ymax = r_upper, fill = climber),
      stat = "identity"
    )
}

# Generates outlier labels for the route plot.
GetOutliers <- function(df_routes) {
  m <- loess(r ~ ewbank, df_routes)
  e <- residuals(m)
  e <- e / sd(e)
  ifelse(abs(e) > 1.65, as.character(df_routes$route), NA)
}

# Assume 01-data_prep.R has already been sourced and 02-run_estimation.py has
# already been run.
dfs <- MergeWithRatings(dfs, data_dir)

total_accuracy <-
  NROW(filter(
    dfs$ascents,
    (predicted > 0.5 & clean == 1) | (predicted < 0.5 & clean == 0)
  )) /
    NROW(dfs$ascents)

# Plots the "predicted" probability of clean ascents vs the actual proportion
# of clean ascents.  Ideally the fit follows the y=x line.
accuracy_plot <- ggplot(dfs$ascents, aes(predicted, clean)) + geom_smooth() +
  geom_abline(slope = 1)

# Indicates the relative frequency of different prediction values.  A weak
# model would have a mode near the "average" number of clean ascents; i.e.
# it isn't adding much value over summary statistics.
prediction_density_plot <- ggplot(dfs$ascents, aes(predicted)) + geom_density()

# Plots the residuals vs the conventional grade of routes.  Ideally the fit
# follows the y=0 line.
residuals_ewbank_plot <- ggplot(
  dfs$ascents %>% inner_join(dfs$routes, by = "route"),
  aes(ewbank, predicted - clean)
) + geom_smooth()

# Plots the residuals vs the estimated natural route rating.  Ideally the fit
# follows the y=0 line.
residuals_route_rating_plot <- ggplot(
  dfs$ascents %>% inner_join(dfs$routes, by = "route"),
  aes(r, predicted - clean)
) + geom_smooth()

# Plots conventional grades vs the estimated "natural rating" of routes.
# Outliers are labeled.
route_rating_plot <- ggplot(
  dfs$routes %>% mutate(outlier = GetOutliers(dfs$routes)),
  aes(ewbank, r)
) + geom_point(alpha = 0.1, size = 0.5, color = "red") +
  geom_smooth() +
  geom_text(aes(label = outlier),
    na.rm = TRUE, size = 2, check_overlap = TRUE,
    hjust = 0, nudge_x = 0.1, vjust = "outward"
  )

png(filename = file.path(data_dir, "Rplot%03d.png"), width = 1024, res = 120)
cat(sprintf("Total accuracy was %0.2f%%\n", total_accuracy * 100.0))
suppressMessages(grid.draw(rbind(
  ggplotGrob(accuracy_plot),
  ggplotGrob(prediction_density_plot),
  size = "last"
)))
suppressMessages(print(residuals_ewbank_plot))
suppressMessages(print(residuals_route_rating_plot))
suppressMessages(print(route_rating_plot))
dev.off()
