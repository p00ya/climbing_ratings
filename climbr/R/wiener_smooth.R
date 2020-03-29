# Custom ggplot2 stat for plotting rating intervals over time.

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


#' Returns an object of class WienerSmooth.  This class supports the `predict()`
#' generic function.
#'
#' @param formula the formula.  If NULL, default as if it was "y ~ x".
#' @param data a data frame with the variables `x`, 'y', (or equivalents
#' defined by the RHS and LFS of "formula" respectively), "var", and "cov";
#' rows should be in increasing order of "x".
#' @param wsq the Wiener variance.
#' @param ... ignored.
WienerSmooth <- function(formula = NULL, data, wsq, ...) {
  if (!is.null(formula)) {
    data <- data.frame(
      x = eval(rlang::f_rhs(formula), data),
      y = eval(rlang::f_lhs(formula), data),
      data[, c("var", "cov")]
    )
  }
  stopifnot(!is.unsorted(data$x, strictly = TRUE))
  df <- data[, c("x", "y", "var", "cov")]
  structure(list(df = df, wsq = wsq), class = "WienerSmooth")
}

#' For each a[i] find the smallest j such that a[i] < b[j], or length(b) + 1 if
#' no such element exists.  Both a and b should be increasing.
#'
#' @param a a numeric vector in increasing order.
#' @param b a numeric vector in increasing order.
.AlignForInterpolation <- function(a, b) {
  stopifnot(!is.unsorted(a, strictly = TRUE))
  stopifnot(!is.unsorted(b, strictly = TRUE))
  ii <- integer(length(a))
  j <- 1L
  n <- length(b)
  # O(n) but awkward to vectorize.
  for (i in seq_along(a)) {
    while (j <= n && b[j] <= a[i]) {
      j <- j + 1L
    }
    ii[i] <- j
  }
  ii
}

#' Implements the `stats::predict()` generic function for WienerSmooth.
#'
#' Interpolates intervals for a Wiener process where the mean, variance and
#' covariance are known only for a set of samples.
#'
#' @param object a model with class WienerSmooth.
#' @param newdata a list(x), where x is a strictly increasing set of x-values;
#' the x-values should be a subset of the range of the x-values in the model.
#' @param level the confidence level.
#' @param ... ignored.
#' @return list(fit = list(fit, lwr, upr), se.fit), like [stats::predict.lm].
predict.WienerSmooth <- function(object, newdata = NULL, level = 0.9, ...) {
  z <- qnorm((1 - level) / 2)
  n <- nrow(object$df)

  # Find closest i1 and i2 such that t[i1] <= t < t[i2].
  t <- utils::head(newdata$x, -1L) # remove the last sample
  i2 <- .AlignForInterpolation(t, object$df$x)
  i1 <- i2 - 1L
  t1 <- object$df$x[i1]
  t2 <- object$df$x[i2]
  v1 <- object$df$var[i1]
  v2 <- object$df$var[i2]
  y1 <- object$df$y[i1]
  y2 <- object$df$y[i2]
  cov <- object$df$cov[i1]

  dt1 <- t - t1 # gap from previous sample
  dt2 <- t2 - t # gap to next sample
  dt <- t2 - t1 # total gap between samples

  # Interpolate variance.
  # See WHR Appendix C.
  sigmasq <- (
    dt1 * dt2 * object$wsq / dt +
      (dt2^2 * v1 +
        2 * dt2 * dt1 * cov +
        dt1^2 * v2) /
        dt^2
  )
  # Linear interpolation of mean.
  mu <- (y1 * dt2 + y2 * dt1) / dt

  # Add back the last sample (doesn't require interpolation).
  sigma <- sqrt(c(sigmasq, object$df$var[n]))
  mu <- c(mu, object$df$y[n])

  q <- z * sigma
  list(fit = list(fit = mu, lwr = mu - q, upr = mu + q), se.fit = sigma)
}

#' Custom ggplot2 Stat for plotting interpolated confidence bands around
#' Wiener processes with discrete samples.  It computes the same variables
#' as `ggplot2::stat_smooth()`.
.StatWienerSmooth <- ggplot2::ggproto("StatWienerSmooth", ggplot2::Stat,
  required_aes = c("x", "y", "cov", "var"),
  compute_group = function(data, scales,
                           formula = NULL, n = 80, level = 0.9, wsq = 1) {
    # Include all data points, plus interpolated samples.
    rng <- range(data$x, na.rm = TRUE)
    interp <- seq(rng[1], rng[2], length.out = n)
    xseq <- sort(unique(c(interp, data$x)))

    mod <- WienerSmooth(formula, data = data, wsq)
    pred <- predict(mod, newdata = data.frame(x = xseq), level = level)
    fit <- as.data.frame(pred$fit)
    names(fit) <- c("y", "ymin", "ymax")
    data.frame(x = xseq, fit, se = pred$se.fit)
  }
)

#' Returns a ggplot2 layer that by default renders a confidence band for
#' a WHR rating over time.
#'
#' @inheritParams ggplot2::stat_smooth
#' @param wsq the Wiener variance.
stat_wiener_smooth <- function(
                               mapping = NULL, data = NULL,
                               geom = "smooth", position = "identity",
                               ...,
                               formula = NULL,
                               se = TRUE,
                               n = 80,
                               level = 0.9,
                               wsq = 1,
                               show.legend = NA,
                               inherit.aes = TRUE) {
  # See vignette("extending-ggplot2", "ggplot2").
  ggplot2::layer(
    stat = .StatWienerSmooth, data = data, mapping = mapping, geom = geom,
    position = position, show.legend = show.legend, inherit.aes = inherit.aes,
    params = list(
      formula = formula,
      se = se,
      n = n,
      level = level,
      wsq = wsq
    )
  )
}
