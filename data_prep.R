# Normalize a table of ascents into ascent, page and route tables.

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

library(dplyr)

period_length <- 604800  # seconds in 1 week

# Classifies tick types like "dog" into either 1 (clean), 0 (not clean) or NA.
# Note that we have pessimistic interpretations of some tick types that are
# ambiguous in practice, e.g. a plain "tick" is not counted as clean.
# Note top-rope ascents can be considered clean.
IsTickClean <- function(ticktype) {
  # See https://www.thecrag.com/en/article/ticktypes
  case_when(
    ticktype %in% c("onsight", "flash", "redpoint", "groundupredpoint",
        "pinkpoint", "clean", "onsightsolo", "topropeonsight", "topropeflash",
        "topropeclean", "secondclean", "leadsolo", "firstfreeascent") ~ 1,
    ticktype %in% c("tick", "second", "dog", "attempt", "retreat", "working",
        "allfreewithrest", "toproperest", "ghost", "secondrest") ~ 0
  )
}

# Given a data frame containing "raw" ascents, normalize these to ascent,
# page and route tables.
#
# df_raw is expected to have the columns "ascentId", "route", "climber",
# "tick", "grade", "timestamp".
NormalizeTables <- function(df_raw) {
  df <- df_raw %>%
    mutate(clean = IsTickClean(tick)) %>%
    filter(!is.na(clean)) %>%
    filter(!is.na(grade)) %>%
    mutate(route = droplevels(route), climber = droplevels(climber))

  # Summarize the tick counts.
  print(paste(nrow(df), "ascents by",
      nlevels(df$climber), "climbers over",
      nlevels(df$route), "routes."))

  # Find the most popular routes:
  top_routes <- df %>% count(route, sort=TRUE)

  # Make the route with the most ascents the "first" route.  This means it
  # will be used as the reference route (rating of 1).  Having lots of ascents
  # means it is (hopefully) a good reference point for comparing climbers.
  df$route <- relevel(df$route, ref=as.character(top_routes[[1,1]]))

  df_routes <- df %>%
    group_by(route) %>%
    summarise(ewbank=floor(median(grade)))

  df_ascents <- df %>%
    mutate(t = (timestamp - min(df$timestamp)) %/% period_length) %>%
    arrange(climber, t)

  df_pages <- df_ascents %>% group_by(climber, t) %>% summarise()
  # Set first_page[c] to be the index in df_pages of the first page for climber c.
  first_page <- (df_pages %>% group_by(climber) %>% summarise(n=n()) %>%
      mutate(idx=head(cumsum(c(1, n)), -1)))$idx

  # time relative to the first ascent from the same climber
  df_pages$rel_t <- df_pages$t - df_pages$t[first_page[df_pages$climber]]
  # time relative to the next page for the same climber (meaningless for last
  # page of each climber).
  df_pages$gap <- c(diff(df_pages$rel_t), 0)

  df_pages <- df_pages %>%
    select(climber, t, gap) %>%
    ungroup() %>%
    mutate(page=row_number())

  df_ascents <- df_ascents %>%
    inner_join(df_pages, by=c("climber", "t")) %>%
    select("route", "climber", "clean", "page")

  df_ascents <- as_tibble(df_ascents)

  return(list(ascents=df_ascents, pages=df_pages, routes=df_routes))
}

# Write normalized tables to CSV files.
#
# The files are written with standard names to the directory specified by
# "dir".
#
# "dfs" should be a list of data frames, with the tags "ascents", "routes" and
# "pages"; like what NormalizeTables returns.
WriteNormalizedTables <- function(dfs, dir) {
  write.csv(dfs$ascents %>%
      mutate(route=as.integer(route)-1, page=page-1) %>%
      select(-climber),
      file.path(dir, "ascents.csv"), row.names=FALSE)
  write.csv(dfs$routes %>%
      mutate(grade=pmax(1, 1 + ewbank - ewbank[[1]])) %>%
      select(-ewbank),
      file.path("data", "routes.csv"), row.names=FALSE)
  write.csv(dfs$pages %>%
      select(climber, gap) %>%
      mutate(climber=as.integer(climber) - 1),
      file.path("data", "pages.csv"), row.names=FALSE)
}

# Read in the ascents table.
df_raw <- read.csv("data/raw_ascents.csv",
  comment.char = "#",
  colClasses=c("integer", "factor", "factor", "factor", "integer", "integer"))

dfs <- NormalizeTables(df_raw)
WriteNormalizedTables(dfs, "data")
