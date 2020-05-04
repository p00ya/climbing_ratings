# Functions for reading ascents from theCrag's JSON API.

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
    # tolower() is about 10x faster than stringr::str_to_lower().
    tolower() %>%
    stringr::str_remove_all(stringr::fixed(" ")) %>%
    dplyr::recode(hangdog = "dog")
}

#' Coerces `x[[idx]]` to an integer, replacing NULL with NA.
#'
#' @param x a list/vector of lists
#' @param idx the element of the outer list
.AsIntegerOrNA <- function(x, idx = 1) {
  ifelse(is.null(x), NA, as.integer(x[[idx]]))
}

#' Coerces `x[[idx]]` to a character, replacing NULL with NA.
#' @param x a list/vector of lists
#' @param idx the element of the outer list
.AsCharacterOrNA <- function(x, idx = 1) {
  ifelse(is.null(x), NA, as.character(x[[idx]]))
}

#' Converts the list-of-singleton-integer-list structures typical of jsonlite to
#' an atomic vector of integers.  NULLs are converted to NA.
#'
#' @param lst a list of lists
#' @param idx the element of the outer list
.FlattenInt <- function(lst, idx = 1) {
  purrr::map_int(lst, .AsIntegerOrNA, idx = idx)
}

#' Converts the list-of-singleton-character-list structures typical of jsonlite
#' to an atomic vector of characters.  NULLs are converted to NA.
#'
#' @param lst a list of lists
#' @param idx the element of the outer list
.FlattenChr <- function(lst, idx = 1) {
  purrr::map_chr(lst, .AsCharacterOrNA, idx = idx)
}


#' Parses ascent data from JSON responses as returned by theCrag's ascent facet
#' API.
#'
#' @param json a parsed JSON object as returned by using jsonlite to
#' parse the files written by "FetchJsonAscentsFromApi".
#' @return a "raw ascents" data frame (see [ReadAllJsonAscents()]).
ParseJsonAscents <- function(json) {
  names(json) <- "data"
  names(json$data) <- c("numberAscents", "page", "perPage", "ascents")

  # Due to the input JSON's use of heterogeneous arrays, what comes out of
  # jsonlite is horribly structured: ascents is a list, each ascent is a list,
  # and the fields have no names.
  df_json <- as.data.frame(
    do.call(rbind, json$data$ascents),
    stringsAsFactors = FALSE
  )
  # These columns must be consistent with the "flatten" parameter in the URL.
  colnames(df_json) <-
    c(
      "id", "routeID", "accountID", "tick", "gradeID", "gradeScore",
      "cprStyle", "date", "pitch"
    )
  df_json %>%
    dplyr::transmute(
      ascentId = .FlattenChr(.data$id),
      route = .FlattenChr(.data$routeID),
      tick = NormalizeAscentType(.FlattenChr(.data$tick)),
      climber = .FlattenChr(.data$accountID),
      timestamp = suppressWarnings(as.integer(as.POSIXct(
        .FlattenChr(.data$date),
        format = "%FT%H:%M:%SZ",
        optional = TRUE
      ))),
      grade = .FlattenInt(.data$gradeScore),
      style = relevel(as.factor(.FlattenChr(.data$cprStyle)), "Sport"),
      .data$pitch
    ) %>%
    purrr::pmap_dfr(function(ascentId, route, tick, climber, timestamp, grade,
                             style, pitch, ...) {
      if (is.null(pitch)) {
        # Return a list rather than a data.frame here, because
        # data.frame is horribly slow.
        return(
          list(
            ascentId = ascentId,
            tick = tick,
            route = route,
            grade = grade,
            style = style,
            climber = climber,
            timestamp = timestamp
          )
        )
      }
      # Split an ascent of a multipitch route into multiple ascents
      # corresponding to each pitch.  Suffix the ascent and route IDs, e.g.
      # with a "P2" suffix for pitch 2.
      purrr::map_dfr(pitch, function(p) {
        list(
          pitch.number = .AsIntegerOrNA(p, 1),
          pitch.tick = .AsCharacterOrNA(p, 2)
        )
      }) %>%
        tidyr::drop_na() %>%
        dplyr::transmute(
          ascentId = paste0(ascentId, "P", .data$pitch.number),
          tick = NormalizeAscentType(unlist(.data$pitch.tick)),
          route = paste0(route, "P", .data$pitch.number),
          grade = grade,
          style = style,
          climber = climber,
          timestamp = timestamp
        )
    })
}

#' Reads all ascent JSON in directory "dir".
#'
#' @param dir character directory containing JSON responses.  JSON filenames
#' are assumed to have the pattern "ascents-*.json".
#' @return a "raw ascents" data frame with columns "ascentId", "route",
#' "climber", "tick", "grade" and "timestamp".
ReadAllJsonAscents <- function(dir) {
  Sys.glob(file.path(dir, "ascents-*.json")) %>%
    purrr::map(function(file) ParseJsonAscents(jsonlite::read_json(file))) %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(
      route = factor(.data$route),
      climber = factor(.data$climber),
      tick = factor(.data$tick)
    )
}

#' Downloads ascent data from theCrag's paginated ascent facet API.
#'
#' @param area character node ID on theCrag.
#' @param api_key character API key.
#' @param dir character directory to save JSON responses.
#' @param start integer first page to read.
#' @param per_page integer number of ascents per page.
#' @param host character hostname for theCrag's server.
FetchJsonAscentsFromApi <- function(area, api_key, dir, start = 1L,
                                    per_page = 5000L,
                                    host = "sandpit.thecrag.com") {
  flatten_param <- paste(
    "data[numberAscents", "page", "perPage", "ascents[id", "route[id]",
    "account[id]", "tick[label]", "gradeID", "gradeScore", "cprStyle", "date",
    "pitch[number", "tick[label]]]]",
    sep = ","
  )
  base_url <- paste0(
    "https://", host, "/api",
    "/facet/ascents/at/", area,
    "/in-setting/natural",
    "?key=", api_key,
    "&flatten=", flatten_param,
    "&sortby=when",
    "&perPage=", per_page
  )

  for (page in start:1024) {
    url <- paste0(base_url, "&page=", page)
    filename <- file.path(dir, sprintf("ascents-%03d.json", page))

    if (utils::download.file(url, filename, method = "libcurl")) {
      # Stop making requests if there was an error.
      warning(paste("Error downloading URL: ", url))
      break
    }

    j <- jsonlite::read_json(filename)
    names(j) <- "data"
    names(j$data) <- c("numberAscents", "page", "perPage", "ascents")
    # Note the perPage parameter of the request may not be respected; calculate
    # the actual pagination from the response.  The response may code these
    # fields as strings.
    j_num_ascents <- as.integer(j$data$numberAscents)
    j_page <- as.integer(j$data$page)
    j_per_page <- as.integer(j$data$perPage)
    # Stop on the last page.
    if (j_page * j_per_page >= j_num_ascents) {
      break
    }

    # Don't hammer theCrag's servers.
    Sys.sleep(1)
    page <- page + 1
  }
}
