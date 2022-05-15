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
#' @param formula the formula.  If NULL, default as if it was `y ~ x`.
#' @param data a data frame with the variables `x`, `y`, (or equivalents
#' defined by the RHS and LFS of "formula" respectively), `var`, and `cov`;
#' rows should be in increasing order of `x`.
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
  data <- data[, c("x", "y", "var", "cov")]
  structure(list(data = data, wsq = wsq), class = "WienerSmooth")
}

#' For each `a[i]` find the smallest j such that `a[i] < b[j]`, or
#' `length(b) + 1` if no such element exists.  Both a and b should be
#' increasing.
#'
#' @param a a numeric vector in increasing order.
#' @param b a numeric vector in increasing order.
#' @keywords internal
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

#' Extrapolates statistics for a Wiener process from a single sample.
#'
#' @param data a data frame with the variables `x`, `y` and `var`, with a
#' single row to extrapolate from.
#' @param x a strictly increasing set of x-values.
#' @param wsq the Wiener variance.
#' @return `list(mu, sigmasq)` with the expected value and variance
#' respectively, for each sample in newdata.
#' @keywords internal
.ExtrapolateWiener <- function(data, x, wsq) {
  stopifnot(nrow(data) == 1L)

  y <- rep(data$y, length(x))
  sigmasq <- abs(x - data$x) * wsq + data$var

  list(y = y, var = sigmasq)
}

#' Interpolates statistics for a Wiener process where the mean, variance and
#' covariance are known only for a set of samples.
#'
#' @param data a data frame with the variables `x`, `y`, `var`, and `cov`;
#' rows should be in increasing order of `x`.
#' @param x a strictly increasing set of x-values.
#' @param wsq the Wiener variance.
#' @return `list(mu, sigmasq)` with the expected value and variance
#' respectively, for each sample in newdata.
#' @keywords internal
.InterpolateWiener <- function(data, x, wsq) {
  n <- nrow(data)

  # Find closest i1 and i2 such that t[i1] <= t < t[i2].
  t <- utils::head(x, -1L) # remove the last sample
  i2 <- .AlignForInterpolation(t, data$x)
  i1 <- i2 - 1L
  t1 <- data$x[i1]
  t2 <- data$x[i2]
  v1 <- data$var[i1]
  v2 <- data$var[i2]
  y1 <- data$y[i1]
  y2 <- data$y[i2]
  cov <- data$cov[i1]

  dt1 <- t - t1 # gap from previous sample
  dt2 <- t2 - t # gap to next sample
  dt <- t2 - t1 # total gap between samples

  # Interpolate variance.
  # See WHR Appendix C.
  sigmasq <- (
    dt1 * dt2 * wsq / dt +
      (dt2^2 * v1 +
        2 * dt2 * dt1 * cov +
        dt1^2 * v2) /
        dt^2
  )
  # Linear interpolation of mean.
  y <- (y1 * dt2 + y2 * dt1) / dt

  # Add back the last sample (doesn't require interpolation).
  sigmasq <- c(sigmasq, data$var[n])
  y <- c(y, data$y[n])

  list(y = y, var = sigmasq)
}

#' Implements the [stats::predict()] generic function for WienerSmooth.
#'
#' Interpolates intervals for a Wiener process where the mean, variance and
#' covariance are known only for a set of samples.
#'
#' @param object a model with class WienerSmooth.
#' @param newdata a list(x), where x is a strictly increasing set of x-values;
#' the x-values should be a subset of the range of the x-values in the model.
#' @param level the confidence level.
#' @param ... ignored.
#' @return `list(fit = list(fit, lwr, upr), se.fit)`, like
#' [stats::predict.lm()].
predict.WienerSmooth <- function(object, newdata = NULL, level = 0.9, ...) {
  stopifnot(nrow(newdata$x) > 1L)
  stopifnot(nrow(object$data) >= 1L)

  z <- qnorm((1 - level) / 2, lower.tail = FALSE)
  xlim <- range(object$data$x)

  # Interpolate.
  interpolated <- .InterpolateWiener(
    object$data,
    dplyr::filter(newdata, xlim[1L] <= .data$x & .data$x <= xlim[2L])$x,
    object$wsq
  )

  # Extrapolate to the left of the data.
  pre <- .ExtrapolateWiener(
    object$data[1L, ],
    dplyr::filter(newdata, .data$x < xlim[1L])$x,
    object$wsq
  )

  # Extrapolate to the right of the data.
  post <- .ExtrapolateWiener(
    utils::tail(object$data, n = 1L),
    dplyr::filter(newdata, .data$x > xlim[2L])$x,
    object$wsq
  )

  y <- c(pre$y, interpolated$y, post$y)
  sigma <- sqrt(c(pre$var, interpolated$var, post$var))

  q <- z * sigma
  list(fit = list(fit = y, lwr = y - q, upr = y + q), se.fit = sigma)
}

#' Custom ggplot2 Stat for plotting interpolated confidence bands around
#' Wiener processes with discrete samples.  It computes the same variables
#' as `ggplot2::stat_smooth()`.
#'
#' @keywords internal
.StatWienerSmooth <- ggplot2::ggproto("StatWienerSmooth", ggplot2::Stat,
  required_aes = c("x", "y", "cov", "var"),
  compute_group = function(data, scales,
                           formula = NULL, n = 80, level = 0.9, wsq = 1) {
    mod <- WienerSmooth(formula, data = data, wsq)

    # Include original data points and uniformly-spaced samples.
    xlim <- scales$x$dimension()
    xseq <- sort(unique(c(seq(xlim[1L], xlim[2L], length.out = n), data$x)))

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
#' @param geom the ggplot2 geom.
stat_wiener_smooth <- function(mapping = NULL, data = NULL,
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
