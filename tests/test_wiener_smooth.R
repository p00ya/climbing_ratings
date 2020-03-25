# Unit tests for 00-wiener_smooth.R

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


context("Tests for 00-wiener_smooth.R")

source("../00-wiener_smooth.R")

describe("WienerSmooth", {
  it("applies formula", {
    df <- data.frame(t = 1:3, u = c(10, 20, 10), var = 4, cov = 0.5)
    mod <- WienerSmooth(u ~ 2 * t, df, wsq = 1)
    expect_equal(mod$df$x, 2 * df$t)
    expect_equal(mod$df$y, df$u)
  })
  it("defaults formula", {
    df <- data.frame(x = 1:3, y = c(10, 20, 10), var = 4, cov = 0.5)
    mod <- WienerSmooth(data = df, wsq = 1)
    expect_equal(mod$df$x, df$x)
    expect_equal(mod$df$y, df$y)
  })
})

describe("AlignForInterpolation", {
  it("with a == b", {
    expect_equal(
      AlignForInterpolation(c(10, 20), c(10, 20)),
      c(2, 3)
    )
  })
  it("with interpolated values", {
    expect_equal(
      AlignForInterpolation(c(0, 3, 6, 8), c(0, 4, 8)),
      c(2, 2, 3, 4)
    )
  })
  it("with non-increasing values", {
    expect_error(AlignForInterpolation(c(20, 10), c(10, 20)))
    expect_error(AlignForInterpolation(c(10, 20), c(20, 10)))
  })
})

describe("predict.WienerSmooth", {
  df <- data.frame(x = 1:3, y = c(10, 20, 10), var = 4, cov = 0.5)
  mod <- WienerSmooth(data = df, wsq = 1)
  p <- predict.WienerSmooth(mod, data.frame(x = 2:6 / 2), level = 0.682)
  it("interpolates mean", {
    expect_equal(p$fit$fit, c(10, 15, 20, 15, 10))
  })
  it("interpolates intervals", {
    expect_equal(p$fit$lwr, c(12, 16.6, 22, 16.6, 12), tolerance = 1e-2)
    expect_equal(p$fit$upr, c(8, 13.4, 18, 13.4, 8), tolerance = 1e-2)
  })
})

describe("stat_wiener_smooth", {
  df <- data.frame(x = 1:3, y = c(10, 20, 10), var = 4, cov = 0.5)
  p <- ggplot2::ggplot(df, ggplot2::aes(x, y)) +
    stat_wiener_smooth(
      ggplot2::aes(var = var, cov = cov),
      n = 5, level = 0.682
    )
  df_smooth <- ggplot2::layer_data(p)
  it("computes x, y, ymin, ymax", {
    expect_equal(df_smooth$x, 2:6 / 2)
    expect_equal(df_smooth$y, c(10, 15, 20, 15, 10))
    expect_equal(df_smooth$ymin, c(12, 16.6, 22, 16.6, 12), tolerance = 1e-2)
    expect_equal(df_smooth$ymax, c(8, 13.4, 18, 13.4, 8), tolerance = 1e-2)
  })
})
