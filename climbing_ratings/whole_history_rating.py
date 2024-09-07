"""Modified Whole-History Rating model for climbing"""

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


import copy
import itertools
import numpy as np
from .bradley_terry import (
    BradleyTerry,
    expand_to_slices,
    expand_to_slices_sparse,
)
from . import derivatives
from .slices import Slices
from numpy.typing import ArrayLike, NDArray
from typing import List, NamedTuple, Optional, Tuple, Union, cast


_Array = NDArray[np.float64]


class Hyperparameters(NamedTuple):
    """Hyperparameters for the climbing ratings model.

    Encapsulates values that are common to all climber or route priors.

    A normal distribution is used as a prior on the natural rating for each
    climber, in the first time period (page) in which they record an ascent.

    A Wiener process is used as a prior on how each climber's natural rating
    may vary over time.

    Normal distributions are used as priors on the natural rating for each
    route.  While the mean can be informed on a per-route basis, the variance
    is common.

    Route ratings are assumed to be time-invariant.

    Attributes
    ----------
    climber_prior_mean
        Mean of the prior distribution for the initial natural rating of each
        climber.
    climber_prior_variance
        Variance of the prior distribution for the initial natural rating of
        each climber.
    climber_wiener_variance
        Variance of the prior process for all climbers ratings over time.
    route_prior_mean
        Mean of the prior distribution for the initial natural rating of each
        route.
    route_prior_variance
        Variance of the prior distribution for the natural rating of each
        route.
    style_prior_mean
        Variance of the prior distribution for the effect of each style on the
        climber's rating.
    style_prior_variance
        Variance of the prior process for the effect of each style on the
        climber's rating over time.
    """

    climber_prior_mean: float
    climber_prior_variance: float
    climber_wiener_variance: float
    route_prior_variance: float
    style_prior_variance: float
    style_wiener_variance: float


class AscentsTable:
    """Normalized table of ascents.

    The table must be ordered by page.

    Attributes
    ----------
    route : ndarray of intp
        The 0-based ID of the route for each ascent.
    clean : _Array
        1 for a clean ascent, 0 otherwise, for each ascent.  The implied
        ascents must be in page order.
    page : ndarray of intp
        The 0-based ID of the page for each ascent.
    style_page : ndarray of intp
        The 0-based ID of the style-page for each ascent.
    """

    __slots__ = ("route", "clean", "page", "style_page")

    def __init__(
        self,
        route: ArrayLike,
        clean: ArrayLike,
        page: ArrayLike,
        style_page: ArrayLike,
    ):
        """Initializes an AscentsTable.

        Parameters
        ----------
        route
            The 0-based ID of the route for each ascent.
        clean
            1 for a clean ascent, 0 otherwise, for each ascent.  The implied
            ascents must be in page order.
        page
            The 0-based ID of the page for each ascent.
        style_page
            The 0-based ID of the style-page for each ascent.
        """
        self.route: NDArray[np.intp] = np.array(route, np.intp)
        self.clean: _Array = np.array(clean, float)
        self.page: NDArray[np.intp] = np.array(page, np.intp)
        self.style_page: NDArray[np.intp] = np.array(style_page, np.intp)

    def __len__(self) -> int:
        """Return the number of ascents in the table."""
        return cast(int, self.route.shape[0])


class PagesTable:
    """Normalized table of pages.

    Pages must be sorted lexicographically by climber, timestamp and style.
    Hence pages belonging to the same climber are all contiguous, and pages
    for the same climber and timestamp are all contiguous.  Style may be
    omitted.

    Attributes
    ----------
    climber : ndarray of intp
        The 0-based ID of the climber (or climber_style) for each page.
    timestamp
        The time of the ascents for each page.
    """

    __slots__ = ("climber", "timestamp")

    def __init__(self, climber: ArrayLike, timestamp: ArrayLike):
        """Initialize a PagesTable.

        Parameters
        ----------
        climber
            The 0-based ID of the climber (or climber_style) for each page.
        timestamp
            The time of the ascents for each page.
        """
        self.climber: NDArray[np.intp] = np.asarray(climber, np.intp)
        self.timestamp: _Array = np.asarray(timestamp, float)

    def __len__(self) -> int:
        """Return the number of pages in the table."""
        return cast(int, self.climber.shape[0])


class PageRatingsTable(NamedTuple):
    """Encapsulates the model for a particular page-slicing.

    This model generalizes over both base pages and style-pages.  The rows
    correspond 1:1 with rows in a PagesTable (in the same order).

    Attributes
    ----------
    ratings
        Current estimate of the natural rating of each page.
    var
        Estimate of the variance of the natural rating of each page.
    cov
        Estimate of the covariance between the natural rating of each page and
        the next page.  The covariance for the last page of each climber (or
        climber_style) is not meaningful.
    """

    ratings: _Array
    var: _Array
    cov: _Array


class WholeHistoryRating:
    """Performs optimization for route and climber ratings.

    Initializes models for climbers and routes from ascent tables.
    Stores an estimate for each rating and performs optimization using Newton's
    method.

    We use two orderings for ascents:
    - page-order: ascents are ordered by page index
    - route-order: ascents are ordered by route index

    Climbers have one ratings process per style in which they have logged
    ascents.  Internally, these are represented as a base-style rating
    (style-page -1) and a style rating for any other style that climber
    has logged.  The style rating is added to the base-style rating to get the
    natural rating for a particular ascent.
    """

    __slots__ = (
        "_bases",
        "_styles",
        "_has_styles",
        "_route_ratings",
        "_route_var",
        "_route_ascents",
        "_route_priors",
        "_route_bt",
    )

    def __init__(
        self,
        hparams: Hyperparameters,
        ascents: AscentsTable,
        pages: PagesTable,
        style_pages: PagesTable,
        routes_rating: Union[_Array, List[float]],
    ):
        """Initialize a WHR model.

        Parameters
        ----------
        hparams
            Parameter values for use in priors across climbers and routes.
        ascents
            Input ascents table.
        pages
            Input pages table.
        style_pages
            Input style-pages table.
        routes_rating
            Initial natural ratings for each route.
        """
        self._bases = _Pages(
            ascents,
            pages,
            ascents.page,
            hparams.climber_prior_mean,
            hparams.climber_prior_variance,
            hparams.climber_wiener_variance,
        )
        self._styles = _Pages(
            ascents,
            style_pages,
            ascents.style_page,
            0.0,
            hparams.style_prior_variance,
            hparams.style_wiener_variance,
        )
        self._has_styles = len(style_pages) > 0
        self._route_ratings: _Array = np.array(routes_rating)
        self._route_var = np.empty_like(self._route_ratings)
        self._route_ascents = _make_route_ascents(
            self._bases.ascents, len(routes_rating)
        )
        self._route_priors = derivatives.MultiNormalDistribution(
            self._route_ratings, hparams.route_prior_variance
        )
        self._route_bt = BradleyTerry(
            self._route_ascents.slices,
            len(ascents),
            len(routes_rating),
        )

    def __copy__(self) -> "WholeHistoryRating":
        """Copies a snapshot of the current model estimates.

        The estimates (base/style/route ratings and variance) are deep-copied,
        effectively creating a snapshot of the current estimates that can be
        updated independently.  References to the other (immutable) fields are
        copied as is.
        """
        whr: WholeHistoryRating = self.__class__.__new__(self.__class__)
        whr._bases = copy.copy(self._bases)
        whr._styles = copy.copy(self._styles)
        whr._has_styles = self._has_styles
        whr._route_ratings = self._route_ratings.copy()
        whr._route_var = self._route_var.copy()
        whr._route_ascents = self._route_ascents
        whr._route_priors = self._route_priors
        whr._route_bt = self._route_bt
        return whr

    @property
    def page(self) -> PageRatingsTable:
        """The estimated ratings of the (base) pages.

        The returned arrays are references to the internal state, so they will
        reflect model updates.
        """
        pages = self._bases
        return PageRatingsTable(pages.ratings, pages.var, pages.cov)

    @property
    def style_page(self) -> PageRatingsTable:
        """The estimated ratings of the style pages.

        The returned arrays are references to the internal state, so they will
        reflect model updates.
        """
        pages = self._styles
        return PageRatingsTable(pages.ratings, pages.var, pages.cov)

    @property
    def route_ratings(self) -> _Array:
        """The natural rating of each route."""
        return self._route_ratings

    @property
    def route_var(self) -> _Array:
        """The variance of each route's natural rating."""
        return self._route_var

    def __get_page_bt_derivatives(
        self, pages: "_Pages", aux_pages: Optional["_Pages"]
    ) -> Tuple[_Array, _Array]:
        """Gets the Bradley-Terry terms for page/style-page ratings.

        Gets the first and second derivative of the Bradley-Terry model with
        respect to the player's natural rating.  Auxiliary ratings allow the
        player to be modeled with either base-ratings or style-ratings.

        Parameters
        ----------
        pages
            A slicing of ascents for the player's ratings.
        aux_pages
            An optional slicing of ascents for the auxiliary ratings.  None is
            equivalent to auxiliary ratings of 0 for all pages.

        Returns
        -------
            A pair of _Arrays (d1, d2) of the first and second derivative of
            the Bradley-Terry log-likelihood the player wins, with respect to
            the player's natural rating.
        """
        ascents = pages.ascents
        ratings = pages.ratings
        num_ascents = len(ascents.adversary)

        expand_to_slices(ratings, ascents.slices, pages.bradley_terry.ratings)

        if aux_pages is not None:
            np.add(
                pages.bradley_terry.ratings,
                expand_to_slices_sparse(
                    aux_pages.ratings, aux_pages.ascents.slices, num_ascents
                ),
                pages.bradley_terry.ratings,
            )

        route_ratings = self.route_ratings[ascents.adversary]

        return pages.bradley_terry.get_derivatives(
            ascents.win,
            route_ratings,
        )

    def __update_page_ratings(
        self,
        pages: "_Pages",
        aux_pages: Optional["_Pages"],
        only_variance: bool,
    ) -> None:
        """Update the ratings of all pages.

        Parameters
        ----------
        pages
            Pages of the player to update.
        aux_pages
            Optional auxiliary pages for the player.  None is equivalent to an
            auxiliary rating of 0 for all pages.
        only_variance
            If true, only updates the page variance and covariance.
            The ratings estimates are left unchanged.
        """
        bt_d1, bt_d2 = self.__get_page_bt_derivatives(pages, aux_pages)

        if only_variance:
            pages.model.update_covariance(bt_d1, bt_d2)
        else:
            pages.model.update_ratings(bt_d1, bt_d2)

    def update_base_ratings(self, only_variance: bool = False) -> None:
        """Update the ratings of all (base) pages.

        Parameters
        ----------
        only_variance
            If true, only updates the base variance and covariance.
            The ratings estimates are left unchanged.
        """
        styles = self._styles if self._has_styles else None
        self.__update_page_ratings(self._bases, styles, only_variance)

    def update_style_ratings(self, only_variance: bool = False) -> None:
        """Update the ratings of all style pages.

        Parameters
        ----------
        only_variance
            If true, only updates the style variance and covariance.
            The ratings estimates are left unchanged.
        """
        self.__update_page_ratings(self._styles, self._bases, only_variance)

    def update_route_ratings(self, only_variance: bool = False) -> None:
        """Update the ratings of all routes.

        Parameters
        ----------
        only_variance
            If true, only updates the "route_var" attribute.
            The ratings estimates are left unchanged.
        """

        expand_to_slices(
            self._route_ratings,
            self._route_ascents.slices,
            self._route_bt.ratings,
        )

        page_ratings = self._bases.ratings[self._route_ascents.adversary]

        # Bradley-Terry terms.
        d1, d2 = self._route_bt.get_derivatives(
            self._route_ascents.win,
            page_ratings,
        )

        # Gaussian prior terms.
        self._route_priors.add_gradient(self._route_ratings, d1)
        d2 += self._route_priors.d2()

        if only_variance:
            np.reciprocal(d2, self._route_var)
            np.negative(self._route_var, self._route_var)
        else:
            delta = d1  # output parameter
            np.divide(d1, d2, delta)

            # r2 = r1 - delta
            self._route_ratings -= delta

    def update_ratings(self) -> None:
        """Update ratings for all routes and pages."""
        # Update pages first because we have better priors/initial values for
        # the routes.
        self.update_base_ratings()
        self.update_route_ratings()
        if self._has_styles:
            self.update_style_ratings()

    def update_covariance(self) -> None:
        """Update covariance estimates for all routes and pages."""
        self.update_base_ratings(True)
        self.update_route_ratings(True)
        self.update_style_ratings(True)

    def get_log_likelihood(self) -> float:
        """Calculate the marginal log-likelihood.

        Evaluates the marginal log-likelihood from the Bradley-Terry model at
        the current ratings.

        This value generally increases toward zero as the model improves.
        However, its maximum may not coincide with the posterior distribution's
        maximum due to the influence of the other priors.

        Returns
        -------
        float
            Marginal log-likelihood.
        """
        # WHR 2.2: Bradley-Terry Model
        #   P(player i beats player j) = exp(r_i) / (exp(r_i) + exp(r_j))
        #                              = 1 / (exp(r_j - r_i) + 1)

        pages = self._bases

        ascent_page_ratings = pages.ascent_ratings()
        ascent_page_ratings += self._styles.ascent_ratings()
        ascent_route_ratings = self._route_ratings[pages.ascents.adversary]

        # log(P) = sum over ascents[ log( 1 / (1 + exp(loser - winner)) ) ]
        #        = - sum ( log( exp(loser - winner) + 1 ) )

        x = ascent_route_ratings
        x -= ascent_page_ratings
        x *= pages.ascents.win
        np.exp(x, x)
        x += 1.0
        np.log(x, x)

        return cast(float, -np.sum(x))


class _SlicedAscents(NamedTuple):
    """Stores ascents organized into contiguous slices.

    Ascents are organized into player-order, where the player is a route or
    a page.  Hence ascents with the same player are contiguous and can be
    addressed by a slice.

    Attributes
    ----------
    slices
        Slices defining the slice in the player-ordered ascents, for each
        player.
    adversary
        The index of the adversary for each player-ordered ascent.
    win
        Each element is 1 if the ascent was a win for the player, -1 otherwise,
        for each ascent.
    """

    slices: Slices
    adversary: _Array
    win: _Array


class _Pages:
    """Encapsulates the ascents and page ratings model for climbers.

    One _Pages object may store the state for all pages for all climbers.

    Attributes
    ----------
    ascents : _SlicedAscents
        Ascents in page order.
    bradley_terry :  BradleyTerry
        Bradley-Terry estimation state for all ascents.
    model : PageModel
        Estimation state for all pages.
    """

    __slots__ = ("ascents", "bradley_terry", "model")

    def __init__(
        self,
        ascents: AscentsTable,
        pages: PagesTable,
        ascents_page: NDArray[np.intp],
        prior_mean: float,
        prior_var: float,
        wiener_var: float,
    ):
        """Initialize a _Pages.

        Parameters
        ----------
        ascents
            All ascents.
        pages
            Slicing of the ascents (may be sparse).
        ascents_page
            Page ID for each ascent (e.g. ascents.pages or ascents.style_pages).
        prior_mean
            Mean of the prior distribution for the initial natural rating of
            each process.
        prior_var
            Variance of the prior distribution for the initial natural rating of
            each process.
        wiener_var
            Variance of the prior process for natural ratings over time.
        """
        num_pages = len(pages)
        num_climbers = 0 if len(pages.climber) == 0 else pages.climber[-1] + 1
        self.ascents = self.__make_page_ascents(ascents, ascents_page, num_pages)
        self.bradley_terry = BradleyTerry(self.ascents.slices, len(ascents), num_pages)

        initial = derivatives.NormalDistribution(prior_mean, prior_var)
        gaps = _get_pages_gap(pages.timestamp)
        slices = Slices(_extract_slices(pages.climber, num_climbers))
        w = derivatives.WienerProcess(gaps, slices, wiener_var)
        invariants = derivatives.PageInvariants(initial, w, slices)
        self.model = derivatives.PageModel(invariants, np.zeros(num_pages))

    def __copy__(self) -> "_Pages":
        """Returns a copy of the model.

        The ratings, var and cov arrays will be deep-copied, while the other
        (immutable or transient) fields are shallow-copied.
        """
        pages: _Pages = self.__class__.__new__(self.__class__)
        pages.ascents = self.ascents
        pages.bradley_terry = self.bradley_terry
        pages.model = copy.copy(self.model)
        return pages

    @property
    def ratings(self) -> _Array:
        return self.model.ratings

    @property
    def var(self) -> _Array:
        return self.model.var

    @property
    def cov(self) -> _Array:
        return self.model.cov

    @staticmethod
    def __make_page_ascents(
        ascents: AscentsTable,
        ascents_page: NDArray[np.intp],
        num_pages: int,
    ) -> _SlicedAscents:
        """Slice ascents by pages."""
        ascents_page_slices = _extract_slices(ascents_page, num_pages)
        # Transform {0, 1} clean values to {-1, 1} win values.
        win: _Array = ascents.clean - 0.5
        np.sign(win, win)
        return _SlicedAscents(Slices(ascents_page_slices), np.array(ascents.route), win)

    def ascent_ratings(self) -> _Array:
        """Page ratings for each ascent."""
        num_ascents = len(self.ascents.adversary)
        return expand_to_slices_sparse(
            self.ratings,
            self.ascents.slices,
            num_ascents,
        )


def _get_pages_gap(pages_timestamp: _Array) -> _Array:
    """Calculate the time difference from each page to the following page.

    Parameters
    ----------
    pages_timestamp
        The time of the ascents for each page.

    Returns
    -------
        Time interval from each page to the next page.  The gap for the last
        page of each climber is undefined.
    """
    pages_gap: _Array = np.array(pages_timestamp)
    pages_gap[:-1] = pages_gap[1:] - pages_gap[:-1]
    return pages_gap


_Slice = Tuple[int, int]


def _extract_slices(
    values: Union[List[int], NDArray[np.intp]], num_slices: int
) -> List[_Slice]:
    """Extract slices of contiguous IDs.

    Parameters
    ----------
    values
        Values are IDs such that 0 <= value < num_slices, or as a special case,
        -1.  -1 may appear anywhere; all occurences of any other value must be
        contiguous within the list.
    num_slices
        The length of the list to return.

    Returns
    -------
        Returns a list x such that x[i] is a tuple (start, end) where start is
        the earliest index of the least value >= i, and end is one plus the
        latest index of the greatest value <= i.

        If "values" is monotonic, then "values[x[i][0]:x[i][1]]"
        is the slice containing (and only containing) all values of "i".
    """
    slices: List[_Slice] = [None] * num_slices  # type: ignore
    start = end = 0
    prev = -1
    for j, value in enumerate(itertools.chain(values, [num_slices])):
        if prev != value:
            if prev != -1:
                slices[prev] = (start, end)

            prev = value
            start = j

        end = j + 1

    # Populate slices for IDs that never ocurred in values.
    end = 0
    for i, slice in enumerate(slices):
        if slice is None:
            slices[i] = (end, end)
        else:
            _, end = slice

    return slices


def _make_route_ascents(ascents: _SlicedAscents, num_routes: int) -> _SlicedAscents:
    """Create a permutation of ascents in route-order.

    Parameters
    ----------
    ascents
        Page-slicing of the ascents.  The adversary field should correspond to
        the route ID.
    num_routes
        Number of routes.  Route indices must be in the interval
        [0, num_routes).  Routes may have zero ascents.

    Returns
    -------
        Ascents ordered by (and sliced by) route.  The "slices" list will have
        length num_routes.  The "clean" attribute is unpopulated.
    """
    num_ascents = len(ascents.adversary)
    rascents_route_slices = []
    rascents_page = [0] * num_ascents
    rascents_win = [0] * num_ascents

    permutation = [(route, a) for a, route in enumerate(ascents.adversary)]
    permutation.sort()
    ascent_to_rascent = [0] * num_ascents

    # Add an additional ascent so the loop adds all routes.
    permutation = itertools.chain(permutation, [(num_routes, -1)])  # type: ignore

    start = end = 0
    i = 0

    for j, (route, a) in enumerate(permutation):
        if 0 <= a:
            ascent_to_rascent[a] = j
            rascents_win[j] = -ascents.win[a]

        if i < route:
            rascents_route_slices.append((start, end))
            # Routes with no ascents:
            rascents_route_slices.extend([(end, end)] * (route - i - 1))

            i = route
            start = j

        end = j + 1

    for page, (start, end) in enumerate(ascents.slices):
        for a in range(start, end):
            rascents_page[ascent_to_rascent[a]] = page

    return _SlicedAscents(
        Slices(rascents_route_slices),
        np.array(rascents_page, dtype=np.intp),
        np.array(rascents_win),
    )
