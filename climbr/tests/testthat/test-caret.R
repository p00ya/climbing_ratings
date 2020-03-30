# Unit tests for caret.R

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


context("Tests for caret.R")


ascents <- data.frame(
  route = factor(c("R1", "R1", "R2", "R2")),
  climber = factor(c("C1", "C2", "C1", "C2")),
  page = c(1L, 2L, 1L, 2L),
  clean = c(1L, 1L, 1L, 0L)
)

pages <- data.frame(
  climber = factor(c("C1", "C2")),
  t = c(0, 0),
  timestamp = c(0L, 0L)
)

routes <- data.frame(
  route = factor(c("R1", "R2")),
  grade = c(15, 20)
)

dfs <- list(ascents = ascents, pages = pages, routes = routes)

page_ratings <- data.frame(
  climber = c(1L, 2L),
  rating = c(0, 1),
  var = c(1, 1),
  cov = c(0, 0)
)

route_ratings <- data.frame(
  route = c("R1", "R2"),
  rating = c(0, 1),
  var = c(1, 1)
)

ratings <- list(routes = route_ratings, pages = page_ratings)

describe("MakeWhrModel()$prob", {
  model <- MakeWhrModel(dfs)
  p <- 1 / (exp(1) + 1)
  it("with all ascents", {
    expect_equal(
      model$prob(ratings, dfs$ascents),
      data.frame(
        ATTEMPT = c(0.5, p, 1 - p, 0.5),
        CLEAN = c(0.5, 1 - p, p, 0.5)
      )
    )
  })
  it("with a subset of ascents", {
    expect_equal(
      model$prob(ratings, dfs$ascents[1, ]),
      data.frame(ATTEMPT = 0.5, CLEAN = 0.5)
    )
  })
})

describe("MakeWhrModel()$predict", {
  model <- MakeWhrModel(dfs)
  it("with all ascents", {
    expect_equal(
      model$predict(ratings, dfs$ascents),
      c("CLEAN", "CLEAN", "ATTEMPT", "CLEAN")
    )
  })
  it("with a subset of ascents", {
    expect_equal(
      model$predict(ratings, dfs$ascents[1, ]),
      "CLEAN"
    )
  })
  model80 <- MakeWhrModel(dfs, threshold = 0.8)
  it("with threshold = 0.8", {
    expect_equal(
      model80$predict(ratings, dfs$ascents),
      c("ATTEMPT", "ATTEMPT", "ATTEMPT", "ATTEMPT")
    )
  })
})

describe("MakeWhrModel()$fit", {
  .fake_run_script_params <- NULL
  # Fake substitute for .RunEstimationScript, which captures the arguments and
  # writes output files.
  .FakeRunScript <- function(w, sigma_c, sigma_r, max_iterations, data_dir) {
    .fake_run_script_params <<- list(
      w = w, sigma_c = sigma_c, sigma_r = sigma_r,
      max_iterations = max_iterations
    )
    write.csv(
      page_ratings,
      file.path(data_dir, "page_ratings.csv"),
      row.names = FALSE
    )
    write.csv(
      route_ratings,
      file.path(data_dir, "route_ratings.csv"),
      row.names = FALSE
    )
    return(0L)
  }

  model <- MakeWhrModel(dfs, run_script = .FakeRunScript)
  train_param <- list(
    g0 = 0, b = 1, w = 10, sigma_c = 20, sigma_r = 30
  )
  fit <- model$fit(ascents, ascents$clean, NULL, train_param)
  it("return value", {
    expect_equal(fit, ratings)
  })
  it("estimation script parameters", {
    expect_equal(
      .fake_run_script_params,
      list(w = 10, sigma_c = 20, sigma_r = 30, max_iterations = 64L)
    )
  })
})
