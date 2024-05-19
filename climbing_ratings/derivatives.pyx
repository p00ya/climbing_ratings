"""Derivative calculations.

This module calculates the derivatives of the log-likelihood for the Wiener
and Gaussian priors, the logic for combining multiple priors into the Hessian
and gradient vector, and for inverting the Hessian.

WHR uses some specialized matrix algorithms that are not easily vectorizable by
numpy.
"""

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


from .slices cimport Slices
cimport numpy as cnp

cnp.import_array()


cdef class TriDiagonal:
    """A tri-diagonal matrix.

    We decompose the matrix into three arrays:
    - d (main diagonal)
    - u (upper sub-diagonal)
    - l (lower sub-diagonal)

    or in matrix notation:

         | d0 u0 ..  0 |
     M = | l0 d1 `.  : |
         |  : `. `. um |
         |  0 .. lm dn |
    """
    cdef double[::1] d
    cdef double[::1] u
    cdef double[::1] l

    def __init__(self, double[::1] d, double[::1] u, double[::1] l):
        self.d = d
        self.u = u
        self.l = l  # no-cython-lint: E741 allow variable name "l"

    def as_tuple(self):
        """Return a tuple (d, u, l) of the diagonal and sub-diagonals."""
        return (self.d.base, self.u.base, self.l.base)


cdef class TriDiagonalLU:
    """An LU-decomposition of a tri-diagonal matrix.

    We decompose the LU matrices into 3 arrays:
    - d (main diagonal of U matrix)
    - b (upper sub-diagonal of U matrix)
    - a (lower sub-diagonal of L matrix)

             |  1  0  0 .. 0 | | d0 b0  0 ..  0 |
             | a0  1  0 .. 0 | |  0 d1 b1 ..  0 |
    M = LU = |  0 a1  1  . 0 | |  0  0 d2 `.  : |
             |  :  . `. `. : | |  :  .  . `. bm |
             |  0  0  . am 1 | |  0  0  0 .. dn |

    Similarly, for UL decompositions:
    - d (main diagonal of L' matrix)
    - b (lower sub-diagonal of L' matrix)
    - a (upper sub-diagonal of U' matrix)

               |  1 a0  0 ..  0 | | d0  0  0 ..  0 |
               |  0  1 a1 ..  0 | | b0 d1  0 ..  0 |
    M = U'L' = |  0  0  1 `.  0 | |  0 b1 d2 ..  0 |
               |  :  .  . `. am | |  :  . `. `.  : |
               |  0  0  0 ..  1 | |  0  0  0 bm dn |

    Note the indices are always 0-based (in WHR the indices are the 1-based row
    number).
    """
    cdef double[::1] d
    cdef double[::1] b
    cdef double[::1] a

    def __init__(self, double[::1] d, double[::1] b, double[::1] a):
        self.d = d
        self.b = b
        self.a = a

    def as_tuple(self):
        """Return a tuple (d, b, a) of the diagonal and sub-diagonals."""
        return (self.d.base, self.b.base, self.a.base)


cdef inline double gaussian_gradient(
    double mu,
    double sigma_sq,
    double x
) noexcept nogil:
    """Inline implementation of NormalDistribution.gradient."""
    return (mu - x) / sigma_sq


cdef inline double gaussian_d2(double sigma_sq) noexcept nogil:
    """Inline implementation of NormalDistribution.d2."""
    return -1.0 / sigma_sq


cdef class NormalDistribution:
    """A Gaussian distribution."""
    cdef double mu
    cdef double sigma_sq

    def __init__(self, double mu, double sigma_sq):
        """
        Parameters
        ----------
        mu
            Mean of the distribution.
        sigma_sq
            Variance of the distribution.
        """
        self.mu = mu
        self.sigma_sq = sigma_sq

    def gradient(self, double x):
        """First-derivative of the log-PDF of the Gaussian distribution.

        Parameters
        ----------
        x
            Quantile of the Gaussian distribution to evaluate."""
        return gaussian_gradient(self.mu, self.sigma_sq, x)

    def d2(self):
        """Second-derivative of the log-PDF of the Gaussian distribution."""
        return gaussian_d2(self.sigma_sq)


cdef class MultiNormalDistribution:
    """A multivariate Gaussian distribution with constant variance."""
    cdef double[::1] mu
    cdef double sigma_sq

    def __init__(self, double[::1] mu, double sigma_sq):
        """
        Parameters
        ----------
        mu
            Means of the individual Gaussian distributions.
        sigma_sq
            Variance of the distribution.
        """
        self.mu = mu
        self.sigma_sq = sigma_sq

    def add_gradient(self, double[::1] x, double[::1] out):
        """First-derivative of the log-PDF of the Gaussian distribution.

        Adds the gradient vector to the output vector.

        Parameters
        ----------
        x
            Quantile of the Gaussian distribution to evaluate.  Must have the
            same length as the mean and output vectors.
        out
            Output vector.
        """
        cdef Py_ssize_t i
        for i in range(x.shape[0]):
            out[i] += gaussian_gradient(self.mu[i], self.sigma_sq, x[i])

    def d2(self):
        """Second-derivative of the log-PDF of the Gaussian distribution."""
        return gaussian_d2(self.sigma_sq)


cdef class WienerProcess:
    """Statistics for Wiener processes.

    This class represents multiple Wiener processes (e.g. for multiple climbers)
    using slices over 1D arrays.

    one_on_sigma_sq
        Reciprocal of the Wiener variance between each page and the next page.
        The last value for each climber is zero.
    d2
        The second-derivative of the log-likelihood for each page.
    """
    cdef double[::1] one_on_sigma_sq
    cdef double[::1] d2

    def __init__(self, double[::1] gaps, Slices slices, double wiener_variance):
        """
        Parameters
        ----------

        gaps
            gaps[i] is the time interval between the page i and page i + 1.
            The length of gaps should equal the number of pages, and the last
            gap for each process will be ignored.
        slices
            slices.start represents the first page for each process, and
            slices.end represents one past the last page for each process.
        wiener_variance
            The variance of the Wiener process per unit time.
        """
        cdef Py_ssize_t n = gaps.shape[0]
        self.one_on_sigma_sq = cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0)
        cdef Py_ssize_t i, j
        cdef double d

        # Shorthand for the attribute.
        cdef double[::1] t = self.one_on_sigma_sq

        for j in range(len(slices)):
            for i in range(slices.start[j], slices.end[j] - 1):
                d = gaps[i] * wiener_variance
                t[i] = 1.0 / d
            t[slices.end[j] - 1] = 0.0

        # WHR Appendix A.2 Terms of the Wiener prior:
        # d^2 ln p / d r[t]^2 = -1 / sigma^2
        # This term is added to the diagonal at both t and t-1.
        self.d2 = cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0)
        self.d2[0] = -t[0]
        for i in range(1, n):
            self.d2[i] = -t[i - 1] - t[i]

    cdef void add_gradient(
        self,
        double[::1] ratings,
        double[::1] gradient,
    ) noexcept nogil:
        """Add terms from the Wiener prior to the gradient.

        Parameters
        ----------
        ratings : contiguous ndarray with length N
            The natural rating for each page.
        gradient : contiguous ndarray with length N
            Output array for the first derivative of the log likelihood with
            respect to the rating for each page.
        """
        # WHR Appendix A.2 Terms of the Wiener prior:
        # d ln p / d r[t] = -(r[t] - r[t+1]) / sigma[t]^2
        cdef Py_ssize_t n = ratings.shape[0]

        cdef double d
        cdef Py_ssize_t i
        for i in range(n - 1):
            d = ratings[i + 1] - ratings[i]
            d *= self.one_on_sigma_sq[i]
            gradient[i] += d
            gradient[i + 1] -= d

    def _add_gradient(self, ratings, gradient):
        """Wrapper around add_gradient for testing."""
        self.add_gradient(ratings, gradient)

    def as_tuple(self):
        """Return the internal state as a tuple of numpy arrays.

        Returns
        -------
        (one_on_sigma_sq : ndarray, d2 : ndarray)
        one_on_sigma_sq
            The reciprocal of the Wiener variance for each page.
        d2
            The second derivative of the log-likelihood function with respect to
            the ratings, for each page.
        """
        return (self.one_on_sigma_sq.base, self.d2.base)


cdef class PageInvariants:
    """Page model parameters that do not depend on the ratings."""
    cdef NormalDistribution initial
    cdef WienerProcess wiener
    cdef Slices slices

    def __init__(
        self,
        NormalDistribution initial,
        WienerProcess wiener,
        Slices slices,
    ):
        self.initial = initial
        self.wiener = wiener
        self.slices = slices


cdef inline Py_ssize_t num_pages(PageInvariants invariants) noexcept nogil:
    return invariants.wiener.one_on_sigma_sq.shape[0]


cdef inline Py_ssize_t num_slices(PageInvariants invariants) noexcept nogil:
    return invariants.slices.start.shape[0]


cdef class PageModel:
    """Reusable state for derivative calculations.

    The derivatives calculated by update_derivatives incorporate terms from both
    the Bradley-Terry model of individual ascent success, the Wiener prior for
    the change of climber's rating between periods, and a Gaussian prior for the
    first page of each climber.

    One PageModel object can represent multiple climbers (processes), with each
    process having one or more pages.  The values for all pages are concatenated
    into flat arrays, such that all pages for a process are contiguous, and
    the start and end indices of a process are found in invariants.slices.

    Attributes
    ----------
    model
        The model contains invariants for the process and the priors for the
        first page of each climber.

    h
        The Hessian matrix of the log-likelihood with respect to all ratings.
        The matrix is tri-diagonal, with the main diagonal defined by:

          h.d[t] = d^2 ln P / d (r[t])^2
          h.u[t] = d^2 ln P / (d r[t] d r[t+1])
          h.l[t] = d^2 ln P / (d r[t] d r[t+1])

    gradient
        The gradient vector of the log-likelihood, i.e.

          gradient[t] = d ln P / (d r[t]).

    lu
        The LU decomposition of the Hessian matrix.  This is used in the WHR
        algorithm to efficiently multiply by the Hessian's inverse.

    ul
        The UL decomposition of the Hessian matrix.  This is used in the WHR
        algorithm to efficiently compute the covariance matrix (the negative
        inverse of the Hessian matrix).

    adjustment
        The adjustment term for one iteration of the Newton-Raphson algorithm.
        This is equivalent to H^-1 G where H is the Hessian and G is the
        gradient vector.

    var
        The estimated variance for each of the natural ratings.

    cov
        The he covariance between the natural ratings of each page and the next
        page.  The last page for each process is not meaningful.
    """
    cdef PageInvariants invariants
    cdef double[::1] g
    cdef TriDiagonal h
    cdef TriDiagonalLU lu
    cdef TriDiagonalLU ul
    cdef double[::1] adjustment
    cdef double[::1] ratings
    cdef double[::1] var
    cdef double[::1] cov

    def __init__(self, PageInvariants invariants, double[::1] ratings):
        cdef Py_ssize_t n = num_pages(invariants)
        self.invariants = invariants

        self.h = TriDiagonal(
            cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0),
            invariants.wiener.one_on_sigma_sq,
            invariants.wiener.one_on_sigma_sq,
        )
        self.g = cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0)
        self.lu = TriDiagonalLU(
            cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0),
            invariants.wiener.one_on_sigma_sq,
            cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0),
        )
        self.ul = TriDiagonalLU(
            cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0),
            cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0),
            cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0),
        )
        self.adjustment = cnp.PyArray_EMPTY(1, [n], cnp.NPY_DOUBLE, 0)
        self.ratings = cnp.PyArray_FROMANY(
            ratings.base,
            cnp.NPY_DOUBLE,
            0,
            0,
            cnp.NPY_ARRAY_ENSURECOPY,
        )
        self.cov = cnp.PyArray_ZEROS(1, [n], cnp.NPY_DOUBLE, 0)
        self.var = cnp.PyArray_ZEROS(1, [n], cnp.NPY_DOUBLE, 0)

    def __copy__(self):
        """Return a copy with independent mutable state."""
        return PageModel(self.invariants, self.ratings)

    cdef void update_derivatives(self, double[::1] bt_d1, double[::1] bt_d2):
        """Set the Hessian and gradient with the given ratings point.

        Parameters
        ----------
        ratings
            The natural rating for each of the climber's pages.
        bt_d1
            First derivative from the Bradley-Terry model.
        bt_d2
            Second derivative from the Bradley-Terry model.
        """
        cdef Py_ssize_t n = self.ratings.shape[0]
        cdef Py_ssize_t i = 0

        # Bradley-Terry terms.
        for i in range(n):
            self.g[i] = bt_d1[i]
        for i in range(n):
            self.h.d[i] = bt_d2[i]

        # Wiener terms.
        self.invariants.wiener.add_gradient(self.ratings, self.g)
        for i in range(n):
            self.h.d[i] += self.invariants.wiener.d2[i]
        # The sub-diagonals of the Hessian aren't dependent on the ratings and
        # are initialized in __init__.

        # Gaussian terms for the first page of each climber.
        cdef Py_ssize_t[::1] starts = self.invariants.slices.start
        cdef Py_ssize_t start
        for i in range(starts.shape[0]):
            start = starts[i]
            # First page Gaussian terms.
            self.g[start] += gaussian_gradient(
                self.invariants.initial.mu,
                self.invariants.initial.sigma_sq,
                self.ratings[start],
            )

        cdef double initial_d2 = gaussian_d2(self.invariants.initial.sigma_sq)
        for i in range(starts.shape[0]):
            self.h.d[starts[i]] += initial_d2

    def _update_derivatives(self, double[::1] bt_d1, double[::1] bt_d2):
        """Wrapper around update_derivatives for testing."""
        self.update_derivatives(bt_d1, bt_d2)

    def update_ratings(
        self,
        double[::1] bt_d1,
        double[::1] bt_d2,
    ):
        """Update each rating term using Newton's method.

        Parameters
        ----------
        bt_d1
            First derivative from the Bradley-Terry model.
        bt_d2
            Second derivative from the Bradley-Terry model.

        Returns
        -------
        ndarray
            The adjustments for each of the ratings to be subtracted from the
            current ratings.
        """
        self.update_derivatives(bt_d1, bt_d2)
        cdef Py_ssize_t i, start, end
        for i in range(num_slices(self.invariants)):
            start = self.invariants.slices.start[i]
            end = self.invariants.slices.end[i]
            lu_decompose(self.h, self.lu, start, end)
            invert_lu_dot_g(self.lu, self.g, self.adjustment, start, end)

        for i in range(self.ratings.shape[0]):
            self.ratings[i] -= self.adjustment[i]

    def update_covariance(
        self,
        double[::1] bt_d1,
        double[::1] bt_d2,
    ):
        """Return the covariance matrix for the ratings.

        Parameters
        ----------
        ratings
            The natural rating for each of the climber's pages.
        bt_d1
            First derivative from the Bradley-Terry model.
        bt_d2
            Second derivative from the Bradley-Terry model.

        Returns
        -------
        (var : ndarray, cov : ndarray)
        var
            The output array for the variance for each of the natural ratings.
        cov
            The output array for the covariance between the natural ratings of
            each page and the next page.  The last page for each climber is
            not meaningful.
        """
        self.update_derivatives(bt_d1, bt_d2)
        cdef Py_ssize_t i, start, end
        for i in range(num_slices(self.invariants)):
            start = self.invariants.slices.start[i]
            end = self.invariants.slices.end[i]
            lu_decompose(self.h, self.lu, start, end)
            ul_decompose(self.h, self.ul, start, end)
            invert_lu(self.lu, self.ul, self.var, self.cov, start, end)
            self.cov[end - 1] = 0.0

        return (self.var.base, self.cov.base)

    @property
    def ratings(self):
        return self.ratings.base

    @property
    def var(self):
        return self.var.base

    @property
    def cov(self):
        return self.cov.base

    @property
    def _gradient(self):
        return self.g.base

    @property
    def _hessian(self):
        return self.h.as_tuple()

cpdef void lu_decompose(
    TriDiagonal h,
    TriDiagonalLU lu,
    Py_ssize_t start,
    Py_ssize_t end
):
    """Decompose a tri-diagonal matrix into LU form.

    Parameters
    ----------
    h : TriDiagonal
        Represents the matrix to decompose.
    lu : TriDiagonalLU
        Represents the LU decomposition.
    start
        The first index of the process slice.
    end
        One plus the last index of the process slice.
    """
    lu.b = h.u
    lu.d[start] = h.d[start]

    cdef Py_ssize_t i
    cdef double t
    for i in range(start + 1, end):
        t = h.l[i - 1] / lu.d[i - 1]
        lu.a[i - 1] = t
        lu.d[i] = h.d[i] - h.u[i - 1] * t


cpdef ul_decompose(
    TriDiagonal h,
    TriDiagonalLU ul,
    Py_ssize_t start,
    Py_ssize_t end
):
    """Decompose a tri-diagonal matrix into U'L' form.

    Parameters
    ----------
    h : TriDiagonal
        Represents the matrix to decompose.
    ul : TriDiagonalLU
        The output to store the U'L' decomposition.
    start
        The first index of the process slice.
    end
        One plus the last index of the process slice.
    """
    # WHR Appendix B.2: Computing diagonal and sub-diagonal terms of H^-1
    ul.b = h.l
    ul.d[end - 1] = h.d[end - 1]

    cdef Py_ssize_t i
    cdef double t
    for i in range(end - 2, start - 1, -1):
        t = h.u[i] / ul.d[i + 1]
        ul.a[i] = t
        ul.d[i] = h.d[i] - ul.b[i] * t


cpdef solve_ul_d(const double[::1] c, double[::1] hd, Py_ssize_t start, Py_ssize_t end):
    """Compute the L'-diagonal in the UL decomposition of a tri-diagonal matrix.

    Computes the array "d" following the recurrence:
    d[i] = hd[i] - c[i] / d[i+1]

    Parameters
    ----------
    c : contiguous ndarray with length N
        The "c" term from the recurrence.
    hd : contiguous ndarray with length N
        The input is used as the "hd" term from the recurrence.  Also used as
        the output array for the computed "d" terms.
    start
        The first index of the process slice.
    end
        One plus the last index of the process slice.
    """
    cdef double d_next = hd[end - 1]
    cdef double t
    for i in range(end - 1, start - 1, -1):
        t = c[i]
        t /= d_next
        t += hd[i]
        hd[i] = t
        d_next = t


cpdef void invert_lu_dot_g(
    TriDiagonalLU lu,
    const double[::1] g,
    double[::1] out,
    Py_ssize_t start,
    Py_ssize_t end,
):
    """Compute M^-1 G.

    Where M = LU, and LU X = G, this is equivalent to solving X.

    See WHR Appendix B.1.

    Parameters
    ----------
    lu
        The tri-diagonal LU decomposition for a square matrix of shape (N, N).
    g
        Contiguous array.
    out
        Contiguous array to store the solution X.
    start
        The first index of the process slice.
    end
        One plus the last index of the process slice.
    """
    # out = Y from LY = G
    solve_y(g, lu.a, out, start, end)
    # out = X from UX = Y
    solve_x(lu.b, lu.d, out, start, end)


cpdef solve_y(
    const double[::1] g,
    const double[::1] a,
    double[::1] y,
    Py_ssize_t start,
    Py_ssize_t end
):
    """Compute the vector Y in LY = G, where L is from the LU decomposition.

    Computes the array "y" following the recurrence:
    y[i] = g[i] - a[i - 1] y[i - 1]

    See WHR Appendix B.1 (but note that our indices for y and g are one less
    than the notation in WHR, and our indices for a are two less).

    Parameters
    ----------
    g
        The input is used as the "g" term from the recurrence.
    a
        The "a" term from the recurrence; only a[start:end-1] is read.
    y
        The output array for the computed "y" terms.
    start
        The first index of the slice.
    end
        One plus the last index of the slice.
    """
    cdef Py_ssize_t i
    y[start] = g[start]
    for i in range(start + 1, end):
        y[i] = g[i] - (a[i - 1] * y[i - 1])


cpdef solve_x(
    const double[::1] b,
    const double[::1] d,
    double[::1] y,
    Py_ssize_t start,
    Py_ssize_t end,
):
    """Compute the vector X in UX = Y, where U is from the LU decomposition.

    Computes the array "x" following the recurrence:
    x[i] = (y[i] - b[i] x[i+1]) / d[i]

    See WHR Appendix B.1.

    Parameters
    ----------
    b
        The "b" term from the recurrence, for the slice b[start:end-1].
    d
        The input is used as the "d" term from the recurrence for the slice
        d[start:end].
    y
        The "y" term from the recurrence for the slice y[start:end].  Also used
        as the output array for the computed "x" terms.
    start
        The first index of the slice.
    end
        One plus the last index of the slice.
    """
    cdef Py_ssize_t i
    y[end - 1] = y[end - 1] / d[end - 1]
    # Note: gcc is known to be bad at auto-vectorizing down-loops.
    # https://stackoverflow.com/questions/7919304#36772982
    for i in range(end - 2, start - 1, -1):
        y[i] = (y[i] - (b[i] * y[i+1])) / d[i]


cpdef invert_lu(
    TriDiagonalLU lu,
    TriDiagonalLU ul,
    double[::1] out_d,
    double[::1] out_l,
    Py_ssize_t start,
    Py_ssize_t end,
):
    """Compute -M^-1.

    For the square matrix M = LU = U'L', solve the diagonal and lower
    sub-diagonal of the negative inverse of M.

    Parameters
    ----------
    lu
        The tri-diagonal LU decomposition for the square matrix M.  Only a
        subset is used, e.g. the slice lu.d[start:end] for the diagonal.
    ul
        The tri-diagonal UL decomposition for the square matrix M.  Only a
        subset is used, e.g. the slice ul.d[start:end] for the diagonal.
    out_d
        The output array for the diagonal of the negative inverse of M.  Its
        length must be the same as the order of M.
    out_l
        The output array for the lower sub-diagonal of the negative inverse
        of M.  Its length must be at least one less than the order of M.
    start
        The first index of the slice.
    end
        One plus the last index of the slice.
    """
    # WHR Appendix B.2: Computing Diagonal and Sub-diagonal Terms of H^-1
    cdef Py_ssize_t i

    # out_d[i] = d'[i+1] / (b[i] b'[i] - d[i] d'[i+1])
    cdef double t
    for i in range(start, end - 1):
        t = lu.b[i] * ul.b[i]
        t -= lu.d[i] * ul.d[i + 1]
        out_d[i] = ul.d[i + 1] / t

    # base case
    out_d[end - 1] = -1.0 / lu.d[end - 1]

    for i in range(start, end - 1):
        out_l[i] = -lu.a[i] * out_d[i + 1]
