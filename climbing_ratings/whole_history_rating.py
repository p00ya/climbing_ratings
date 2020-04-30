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


import collections
import itertools
import numpy as np
from .bradley_terry import expand_to_slices, get_bt_derivatives
from .normal_distribution import NormalDistribution
from .process import Process


class Hyperparameters(
    collections.namedtuple(
        "Hyperparameters",
        [
            "climber_prior_mean",
            "climber_prior_variance",
            "climber_wiener_variance",
            "route_prior_variance",
        ],
    )
):
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
    climber_prior_mean : float
        Mean of the prior distribution for the initial natural rating of each
        climber.
    climber_prior_variance : float
        Variance of the prior distribution for the initial natural rating of
        each climber.
    climber_wiener_variance : float
        Variance of the prior process for all climbers ratings over time.
    route_prior_variance : float
        Variance of the prior distribution for the natural rating of each
        route.
    """


class AscentsTable:
    """Normalized table of ascents.

    The table must be ordered by page.

    Attributes
    ----------
    route : ndarray of intp
        The 0-based ID of the route for each ascent.
    clean : ndarray
        1 for a clean ascent, 0 otherwise, for each ascent.  The implied
        ascents must be in page order.
    page : ndarray of intp
        The 0-based ID of the page for each ascent.
    style_page : ndarray of intp, or None
        The 0-based ID of the style-page for each ascent.
    """

    __slots__ = ("route", "clean", "page", "style_page")

    def __init__(self, route, clean, page, style_page):
        """Initializes an AscentsTable.

        Parameters
        ----------
        route : array_like of int
            The 0-based ID of the route for each ascent.
        clean : array_like of float
            1 for a clean ascent, 0 otherwise, for each ascent.  The implied
            ascents must be in page order.
        page : array_like of int
            The 0-based ID of the page for each ascent.
        style_page : array_like of int
            The 0-based ID of the style-page for each ascent.
        """
        self.route = np.array(route, np.intp)
        self.clean = np.array(clean, np.float_)
        self.page = np.array(page, np.intp)
        self.style_page = np.array(style_page, np.intp)

    def __len__(self):
        """Return the number of ascents in the table."""
        return self.route.shape[0]


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
    timestamp : ndarray
        The time of the ascents for each page.
    """

    __slots__ = ("climber", "timestamp")

    def __init__(self, climber, timestamp):
        """Initialize a PagesTable.

        Parameters
        ----------
        climber : array_like of int
            The 0-based ID of the climber (or climber_style) for each page.
        timestamp : array_like of float
            The time of the ascents for each page.
        """
        self.climber = np.array(climber, np.intp)
        self.timestamp = np.array(timestamp)

    def __len__(self):
        """Return the number of pages in the table."""
        return self.climber.shape[0]


class WholeHistoryRating:
    """Performs optimization for route and climber ratings.

    Initializes models for climbers and routes from ascent tables.
    Stores an estimate for each rating and performs optimization using Newton's
    method.

    We use two orderings for ascents:
    - page-order: ascents are ordered by page index
    - route-order: ascents are ordered by route index
    """

    def __init__(self, hparams, ascents, pages, routes_rating):
        """Initialize a WHR model.

        Parameters
        ----------
        hparams: Hyperparameters
            Parameter values for use in priors across climbers and routes.
        ascents : AscentsTable
            Input ascents table.
        pages : PagesTable
            Input pages table.
        routes_rating : list
            Initial natural ratings for each route.
        """
        num_pages = len(pages)
        self._route_ratings = np.array(routes_rating)
        self._bases = _PageModel(
            ascents,
            pages,
            hparams.climber_prior_mean,
            hparams.climber_prior_variance,
            hparams.climber_wiener_variance,
        )
        self._route_var = np.empty_like(self._route_ratings)
        self._route_ascents = _make_route_ascents(
            ascents.clean, self._bases.ascents.slices, ascents.route, len(routes_rating)
        )
        self._route_priors = NormalDistribution(
            self._route_ratings, hparams.route_prior_variance
        )

    @property
    def page_ratings(self):
        """The natural rating of each page, as an ndarray."""
        return self._bases.ratings

    @property
    def page_var(self):
        """The variance of each page's natural rating, as an ndarray."""
        return self._bases.var

    @property
    def page_cov(self):
        """The covariance between pages.

        The returned ndarray contains the covariance between the natural rating
        of each page and the next page, for each page.  The covariance for the
        last page of each climber is not meaningful.
        """
        return self._bases.cov

    @property
    def route_ratings(self):
        """The natural rating of each route, as an ndarray."""
        return self._route_ratings

    @property
    def route_var(self):
        """The variance of each route's natural rating, as an ndarray."""
        return self._route_var

    def update_page_ratings(self, should_update_covariance=False):
        """Update the ratings of all pages.

        Parameters
        ----------
        should_update_covariance : boolean
            If true, updates the "page_var" and "page_cov" attributes.
            This has no effect on rating estimation.
        """
        pages = self._bases

        ascent_page_gammas = expand_to_slices(
            np.exp(pages.ratings), pages.ascents.slices, len(pages.ascents.adversary)
        )
        ascent_route_gammas = self._route_ratings[pages.ascents.adversary]
        np.exp(ascent_route_gammas, ascent_route_gammas)

        bt_d1, bt_d2 = get_bt_derivatives(
            pages.ascents.slices,
            pages.ascents.wins,
            ascent_page_gammas,
            ascent_route_gammas,
        )

        for i, (start, end) in enumerate(self._bases.slices):
            if start == end:
                continue
            climber = pages.processes[i]
            delta = climber.get_ratings_adjustment(
                pages.ratings[start:end], bt_d1[start:end], bt_d2[start:end]
            )
            # r2 = r1 - delta
            pages.ratings[start:end] -= delta

            if should_update_covariance:
                climber.get_covariance(
                    pages.ratings[start:end],
                    bt_d1[start:end],
                    bt_d2[start:end],
                    pages.var[start:end],
                    pages.cov[start : end - 1],
                )

    def update_route_ratings(self, should_update_variance=False):
        """Update the ratings of all routes.

        Parameters
        ----------
        should_update_variance : boolean
            If true, updates the "route_var" attribute.  This has no effect on
            rating estimation.
        """

        rascents_route_gammas = expand_to_slices(
            np.exp(self._route_ratings),
            self._route_ascents.slices,
            len(self._route_ascents.adversary),
        )

        rascents_page_gammas = self._bases.ratings[self._route_ascents.adversary]
        np.exp(rascents_page_gammas, rascents_page_gammas)

        # Bradley-Terry terms.
        d1, d2 = get_bt_derivatives(
            self._route_ascents.slices,
            self._route_ascents.wins,
            rascents_route_gammas,
            rascents_page_gammas,
        )

        # Prior terms.
        prior_d1, prior_d2 = self._route_priors.get_derivatives(self._route_ratings)
        d1 += prior_d1
        d2 += prior_d2

        delta = d1  # output parameter
        np.divide(d1, d2, delta)

        # r2 = r1 - delta
        self._route_ratings -= delta

        if should_update_variance:
            np.reciprocal(d2, self._route_var)
            np.negative(self._route_var, self._route_var)

    def update_ratings(self, should_update_variance=False):
        """Update ratings for all routes and pages.

        Parameters
        ----------
        should_update_variance : boolean
            If true, updates page_var, page_cov and route_var attributes.
        """
        # Update pages first because we have better priors/initial values for
        # the routes.
        self.update_page_ratings(should_update_variance)
        self.update_route_ratings(should_update_variance)

    def get_log_likelihood(self):
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

        # While we could use ascents.wins to make evaluation of the numerator
        # O(pages + routes), to avoid numeric overflow it's better to iterate
        # over ascents (which is unavoidable for the denominator anyway).
        pages = self._bases

        ascent_page_ratings = expand_to_slices(
            pages.ratings, pages.ascents.slices, len(pages.ascents.adversary)
        )
        ascent_route_ratings = self._route_ratings[pages.ascents.adversary]

        # log(P) = sum over ascents[ log( exp(r_i) / (exp(r_i) + exp(r_j)) ) ]
        #        = sum[ r_winner - log( exp(r_i) + exp(r_j) ) ]

        # Collect the rating of each winner for the numerator.
        clean = pages.ascents.clean
        x = clean * ascent_page_ratings
        x += (1.0 - clean) * ascent_route_ratings

        # Sum the exponential terms for the denominator, reusing the ratings
        # arrays.
        np.exp(ascent_page_ratings, ascent_page_ratings)
        np.exp(ascent_route_ratings, ascent_route_ratings)
        denominator = ascent_page_ratings
        denominator += ascent_route_ratings

        np.log(denominator, denominator)

        x -= denominator
        return np.sum(x)


class _SlicedAscents(
    collections.namedtuple("_SlicedAscents", ["wins", "slices", "adversary", "clean"])
):
    """Stores ascents organized into contiguous slices.

    Ascents are organized into player-order, where the player is a route or
    a page.  Hence ascents with the same player are contiguous and can be
    addressed by a slice.

    Attributes
    ----------
    wins : ndarray
        Count of wins for each player.
    slices : list of pairs
        (start, end) pairs defining the slice in the player-ordered ascents,
        for each player.
    adversary : ndarray of intp
        The index of the adversary for each player-ordered ascent.
    clean : ndarray or None
        Each element is 1 if the ascent was clean, 0 otherwise for each ascent.
    """


class _PageModel:
    """Encapsulates the model for a particular page-slicing.

    Attributes
    ----------
    ratings : ndarray
        Current estimate of the natural rating of each page.
    var : ndarray
        Estimate of the variance of the natural rating of each page.
    cov : ndarray
        Estimate of the covariance between the natural rating of each page and
        the next page.  The covariance for the last page of each climber is
        not meaningful.
    ascents : _SlicedAscents
        Ascents in page order.
    slices : list of pairs
        Start and end indices in ratings for each climber.
    processes : list of Process
        Process models (in the same order as slices).
    """

    __slots__ = ("ratings", "var", "cov", "ascents", "slices", "processes")

    def __init__(self, ascents, pages, prior_mean, prior_var, wiener_var):
        """Initialize a _PageModel.

        Parameters
        ----------
        ascents : AscentsTable
        pages : PagesTable
        prior_mean : float
            Mean of the prior distribution for the initial natural rating of
            each process.
        prior_var : float
            Variance of the prior distribution for the initial natural rating of
            each process.
        wiener_var : float
            Variance of the prior process for natural ratings over time.
        """
        num_pages = len(pages)
        self.ratings = np.full(num_pages, prior_mean)
        self.var = np.empty(num_pages)
        self.cov = np.zeros(num_pages)
        self.ascents = self.__make_page_ascents(ascents, num_pages)
        self.slices = _extract_slices(pages.climber, pages.climber[-1] + 1)
        self.processes = self.__make_processes(
            prior_var, wiener_var, pages, self.slices
        )

    @staticmethod
    def __make_page_ascents(ascents, num_pages):
        """Slice ascents by pages."""
        ascents_page_slices = _extract_slices(ascents.page, num_pages)
        wins = np.array(_sum_slices(ascents.clean, ascents_page_slices))
        return _SlicedAscents(
            wins, ascents_page_slices, np.array(ascents.route), np.array(ascents.clean)
        )

    @staticmethod
    def __make_processes(var, wiener_var, pages, page_slices):
        """Make processes for each slice of pages."""
        pages_gap = _get_pages_gap(pages.timestamp)
        prior = NormalDistribution(0.0, var)
        return [
            Process(wiener_var, prior, pages_gap[start : end - 1])
            for (start, end) in page_slices
        ]


def _get_pages_gap(pages_timestamp):
    """Calculate the time difference from each page to the following page.

    Parameters
    ----------
    pages_timestamp : array_like of float
        The time of the ascents for each page.

    Returns
    -------
    ndarray
        Time interval from each page to the next page.  The gap for the last
        page of each climber is undefined.
    """
    pages_gap = np.array(pages_timestamp)
    pages_gap[:-1] = pages_gap[1:] - pages_gap[:-1]
    return pages_gap


def _extract_slices(values, num_slices):
    """Extract slices of contiguous values.

    Parameters
    ----------
    values : list of int
        A list of values in ascending order.
    num_slices : int
        The length of the list to return.

    Returns
    -------
    list of (start, end) tuples
        Returns a list x such that x[i] is a tuple (start, end) where start is
        the earliest index of the least value >= i, and end is the latest index
        of the greatest value <= i.
    """
    slices = []
    start = end = 0
    i = 0
    for j, value in enumerate(itertools.chain(values, [num_slices])):
        if i < value:
            slices.append((start, end))
            # Add missing values:
            slices.extend([(end, end)] * (value - i - 1))
            i = value
            start = j

        end = j + 1

    return slices


def _make_route_ascents(ascents_clean, ascents_page_slices, ascents_route, num_routes):
    """Create a permutation of ascents in route-order.

    Parameters
    ----------
    ascents_clean : array_like of float
        1 if the ascent was clean, 0 otherwise.
    ascents_route : ndarray of intp
        Route index of each ascent.
    ascents_page : ndarray of intp
        Page index of each ascent.
    num_routes : integer
        Number of routes.  Route indices must be in the interval
        [0, num_routes).  Routes may have zero ascents.

    Returns
    -------
    Ascents
        Ascents ordered by (and sliced by) route.  The "slices" list will have
        length num_routes.  The "clean" attribute is unpopulated.
    """
    num_ascents = len(ascents_route)
    route_wins = []
    rascents_route_slices = []
    rascents_page = [0] * num_ascents

    permutation = [(route, a) for a, route in enumerate(ascents_route)]
    permutation.sort()
    ascent_to_rascent = [0] * num_ascents

    # Add an additional ascent so the loop adds all routes.
    permutation = itertools.chain(permutation, [(num_routes, -1)])

    start = end = 0
    i = 0
    wins = 0.0

    for j, (route, a) in enumerate(permutation):
        if 0 <= a:
            ascent_to_rascent[a] = j
        if i < route:
            rascents_route_slices.append((start, end))
            route_wins.append(wins)

            # Routes with no ascents:
            rascents_route_slices.extend([(end, end)] * (route - i - 1))
            route_wins.extend([0.0] * (route - i - 1))

            i = route
            start = j
            wins = 0.0
        end = j + 1
        wins += 1.0 - ascents_clean[a]

    for page, (start, end) in enumerate(ascents_page_slices):
        for a in range(start, end):
            rascents_page[ascent_to_rascent[a]] = page

    return _SlicedAscents(
        np.array(route_wins),
        rascents_route_slices,
        np.array(rascents_page, dtype=np.intp),
        None,
    )


def _sum_slices(values, slices):
    """Sum the values in each slice."""
    sum = np.sum  # save repeated lookup overhead
    return [sum(values[start:end]) for start, end in slices]
