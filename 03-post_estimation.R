# Exploration of estimated ratings.

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


library(climbr)
library(dplyr)
library(ggplot2)
library(grid)

# Assume 01-data_prep.R has already been sourced and 02-run_estimation.py has
# already been run.
dfs <- MergeWithRatings(dfs, ReadRatings(data_dir))

total_accuracy <-
  NROW(filter(
    dfs$ascents,
    (predicted > 0.5 & clean == 1) | (predicted < 0.5 & clean == 0)
  )) /
    NROW(dfs$ascents)

log_loss <- -mean(log(ifelse(
  dfs$ascents$clean,
  dfs$ascents$predicted,
  1 - dfs$ascents$predicted
)))

# Plots the "predicted" probability of clean ascents vs the actual proportion
# of clean ascents.  Ideally the fit follows the y=x line.
accuracy_plot <- ggplot(dfs$ascents, aes(predicted, clean)) +
  geom_smooth() +
  geom_abline(slope = 1)

# Indicates the relative frequency of different prediction values.  A weak
# model would have a mode near the "average" number of clean ascents; i.e.
# it isn't adding much value over summary statistics.
prediction_density_plot <- ggplot(dfs$ascents, aes(predicted)) +
  geom_density()

# Plots the residuals vs the conventional grade of routes.  Ideally the fit
# follows the y=0 line.
residuals_grade_plot <- ggplot(
  dfs$ascents %>% inner_join(dfs$routes, by = "route"),
  aes(grade, clean - predicted)
) +
  geom_smooth()

# Plots the residuals vs the estimated natural route rating.  Ideally the fit
# follows the y=0 line.
residuals_route_rating_plot <- ggplot(
  dfs$ascents %>% inner_join(dfs$routes, by = "route"),
  aes(r, clean - predicted)
) +
  geom_smooth()

# Plots conventional grades vs the estimated "natural rating" of routes.
# Outliers are labeled.
route_rating_plot <- PlotRouteRating(dfs$routes)

png(filename = file.path(data_dir, "Rplot%03d.png"), width = 1024, height = 768, res = 120)
cat(sprintf("Total accuracy was %0.2f%%\n", total_accuracy * 100.0))
cat(sprintf("Log loss was %0.2f\n", log_loss))

suppressMessages(grid.draw(rbind(
  ggplotGrob(accuracy_plot),
  ggplotGrob(prediction_density_plot),
  size = "last"
)))
suppressMessages(print(residuals_grade_plot))
suppressMessages(print(residuals_route_rating_plot))
suppressMessages(print(route_rating_plot))
dev.off()
