# Functions for pre-processing ascents data.

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


#' Converts ascent labels to all lower-case codes with no spaces.
#'
#' @param ascent_type a label like "Hang dog"
NormalizeAscentType <- function(ascent_type) {
  ascent_type %>%
    stringr::str_to_lower() %>%
    stringr::str_remove_all(stringr::fixed(" ")) %>%
    dplyr::recode(hangdog = "dog")
}

#' Classifies tick types into a logical indicating whether the ascent was
#' clean.
#'
#' Note that we have pessimistic interpretations of some tick types that are
#' ambiguous in practice, e.g. a plain "tick" is not counted as clean.
#' Furthermore, no tick shift is applied - a clean top rope ascent is equivalent
#' to an onsight.
#'
#' @param ticktype vector of ticktype codes like "dog".
#' @return a logical vector, which may contain NA values.
IsTickClean <- function(ticktype) {
  # See https://www.thecrag.com/en/article/ticktypes
  dplyr::case_when(
    ticktype %in% c(
      "onsight", "flash", "redpoint", "groundupredpoint",
      "pinkpoint", "clean", "onsightsolo", "topropeonsight", "topropeflash",
      "topropeclean", "secondclean", "leadsolo", "firstfreeascent"
    ) ~ TRUE,
    ticktype %in% c(
      "dog", "attempt", "retreat", "working",
      "allfreewithrest", "toproperest", "ghost", "secondrest"
    ) ~ FALSE
  )
}

#' Tidies a raw ascents table.
#'
#' Removes unclassifiable ascents, removes routes with less than 2 ascents, adds
#' a "clean" column, and re-orders the route levels so the first route has the
#' most ascents at the most common grade.  If the input was missing a style
#' column, the base style will be added to all ascents.
#'
#' Also prints a summary of the resulting data.
#'
#' @param df_raw data frame with the columns "ascentId", "route", "climber",
#' "tick", "grade", "timestamp".  It can optionally have the column "style".
#' @param min_time a POSIXct; ascents from before this time are removed.
#' @param max_time a POSIXct; ascents from after this time are removed.
CleanAscents <- function(df_raw, min_time = 0L, max_time = NA) {
  df <- df_raw
  if (!("style" %in% colnames(df))) {
    df$style <- 1L
  }

  df <- df %>%
    dplyr::mutate(clean = IsTickClean(.data$tick)) %>%
    tidyr::drop_na(.data$clean, .data$grade, .data$timestamp, .data$style) %>%
    dplyr::filter(min_time <= .data$timestamp) %>%
    dplyr::filter(is.na(max_time) | .data$timestamp < max_time)

  # Summarise routes by their grade and number of ascents:
  routes <- df %>%
    dplyr::group_by(.data$route) %>%
    dplyr::summarise(n = dplyr::n(), grade = floor(median(.data$grade)))

  # Make the route with the most ascents for the most common grade the first
  # route.  This means it will be used as the reference route (natural rating
  # prior with mode 0).  Having the most common grade and lots of ascents means
  # it is (hopefully) a good reference point.
  route_grades <- routes %>% dplyr::count(.data$grade)
  routes <- routes %>%
    dplyr::inner_join(
      route_grades,
      by = "grade", suffix = c(".ascents", ".grade")
    ) %>%
    dplyr::arrange(
      dplyr::desc(.data$n.grade),
      dplyr::desc(.data$n.ascents)
    ) %>%
    dplyr::select(-.data$n.grade, -.data$grade)

  # Drop ascents where the route has a single ascent.
  df <- df %>%
    dplyr::inner_join(routes, by = "route") %>%
    dplyr::filter(.data$n.ascents > 1) %>%
    dplyr::select(-.data$n.ascents)

  # Find how often climbers log clean ascents:
  climbers <- df %>%
    dplyr::group_by(.data$climber) %>%
    dplyr::summarise(clean_p = mean(.data$clean))

  # Drop ascents where the climber hasn't logged any non-clean ascents.
  df <- df %>%
    dplyr::inner_join(
      dplyr::filter(climbers, .data$clean_p < 1),
      by = "climber"
    ) %>%
    dplyr::select(-.data$clean_p)

  # Apply previous filters to the routes too.
  routes <- routes %>% dplyr::inner_join(df, by = "route")

  # Recompute factors from the preprocessed data.
  df <- df %>% dplyr::mutate(
    climber = droplevels(.data$climber)
  )
  if (nrow(df) > 1L) {
    df$route <- df$route %>%
      droplevels() %>%
      relevel(ref = as.character(routes[[1, 1]]))
  }

  df
}

#' Summarizes ascents data.
#'
#' @param df a data frame containing ascents (as produced by [CleanAscents()]).
#' @return a character summary of the ascents.
SummarizeAscents <- function(df) {
  paste(
    prettyNum(nrow(df), big.mark = ","), "ascents by",
    prettyNum(nlevels(df$climber), big.mark = ","), "climbers, over",
    prettyNum(nlevels(df$route), big.mark = ","), "routes;",
    sprintf("%0.2f%%", mean(df$clean) * 100.0), "clean ascents\n"
  )
}

#' Normalizes ascents to ascent, page and route tables.
#'
#' @param df a data frame containing filtered ascents (as produced by
#' [CleanAscents()]).
#' @param period_length the number of seconds per page.
NormalizeTables <- function(df, period_length) {
  df_routes <- df %>%
    dplyr::group_by(.data$route) %>%
    dplyr::summarise(grade = floor(median(.data$grade)))

  df_ascents <- df %>%
    dplyr::mutate(
      timestamp = (.data$timestamp %/% period_length) * period_length,
      clean = as.numeric(.data$clean)
    ) %>%
    dplyr::arrange(.data$climber, .data$timestamp, .data$style)

  df_style_pages <- df_ascents %>%
    dplyr::group_by(.data$climber, .data$style, .data$timestamp) %>%
    dplyr::summarise() %>%
    dplyr::ungroup() %>%
    dplyr::filter(as.integer(.data$style) != 1L) %>%
    dplyr::arrange(.data$climber, .data$style, .data$timestamp) %>%
    dplyr::mutate(style_page = dplyr::row_number()) %>%
    dplyr::select(.data$style_page, .data$climber, .data$style, .data$timestamp)

  df_climber_styles <- df_style_pages %>%
    dplyr::group_by(.data$climber, .data$style) %>%
    dplyr::summarise() %>%
    dplyr::ungroup() %>%
    dplyr::mutate(climber_style = dplyr::row_number())

  df_style_pages <- df_style_pages %>%
    dplyr::inner_join(
      df_climber_styles,
      by = c("climber", "style")
    )

  df_pages <- df_ascents %>%
    dplyr::group_by(.data$climber, .data$timestamp) %>%
    dplyr::summarise() %>%
    dplyr::ungroup() %>%
    dplyr::mutate(page = dplyr::row_number()) %>%
    dplyr::select(.data$page, .data$climber, .data$timestamp)

  df_ascents <- df_ascents %>%
    dplyr::inner_join(
      df_pages,
      by = c("climber", "timestamp")
    ) %>%
    dplyr::left_join(
      df_style_pages,
      by = c("climber", "style", "timestamp")
    ) %>%
    dplyr::mutate(
      style_page = ifelse(is.na(.data$style_page), 0L, .data$style_page)
    ) %>%
    dplyr::select(
      .data$route, .data$climber, .data$clean, .data$page, .data$style_page
    )

  list(
    ascents = df_ascents, pages = df_pages, style_pages = df_style_pages,
    routes = df_routes
  )
}

#' Performs a transformation of a grade to a natural rating.
#'
#' This transformation assumes a linear relationship between the grade and the
#' "natural" rating.
#'
#' @param grade numeric vector of grades to transform.
#' @param scale a scale of 1 implies that if a climber has a 0.5 probability of
#' ascending a route at grade X cleanly, then they have a 1 / (1 + e)
#' (approx. 0.27) probability of ascending a route at grade X + 1 cleanly.
#' @param ref the grade to normalize to a natural rating of 0.
TransformGrade <- function(grade, scale = 0, ref = grade[[1]]) {
  scale * (grade - ref)
}

#' Writes normalized tables to CSV files with standard names.
#'
#' @param dfs a list of data frames, with the tags "ascents", "routes" and
#' "pages"; like what NormalizeTables returns.
#' @param dir the directory to write the CSV files to.
WriteNormalizedTables <- function(dfs, dir) {
  utils::write.csv(
    dfs$ascents %>%
      dplyr::transmute(
        route = as.integer(.data$route) - 1L,
        .data$clean,
        page = .data$page - 1L,
        style_page = .data$style_page - 1L
      ),
    file.path(dir, "ascents.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    dfs$routes %>% dplyr::select(.data$route, .data$rating),
    file.path(dir, "routes.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    dfs$pages %>%
      dplyr::transmute(
        climber = as.integer(.data$climber) - 1L,
        .data$timestamp,
      ),
    file.path(dir, "pages.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    dfs$style_pages %>%
      dplyr::transmute(
        climber_style = as.integer(.data$climber_style) - 1L,
        .data$timestamp
      ),
    file.path(dir, "style_pages.csv"),
    row.names = FALSE
  )
}

#' Writes normalized tables in full to an RDS file.
#'
#' Writes "dfs" to the file "dfs.rds", adding a "version" character vector
#' containing the current version of the "climbr" package.
#'
#' @param dfs a list of data frames, with the tags "ascents", "routes" and
#' "pages"; like what NormalizeTables returns.
#' @param dir the directory to write the RDS file to.
SaveNormalizedTablesAsRDS <- function(dfs, dir) {
  dfs$version <- utils::packageVersion("climbr")
  saveRDS(dfs, file.path(dir, "dfs.rds"))
}
