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


class AscentsTable(
    collections.namedtuple("AscentsTable", ["route", "clean", "page", "style_page"])
):
    """Normalized table of ascents.

    The table must be ordered by page.

    Attributes
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


class PagesTable(collections.namedtuple("PagesTable", ["climber", "timestamp"])):
    """Normalized table of pages.

    Pages must be sorted lexicographically by climber and timestamp.
    Hence pages belonging to the same climber are all contiguous.

    Attributes
    ----------
    climber : array_like of int
        The 0-based ID of the climber (or climber_style) for each page.
    timestamp : array_like of float
        The time of the ascents for each page.
    """


class WholeHistoryRating:
    """Performs optimization for route and climber ratings.

    Initializes models for climbers and routes from ascent tables.
    Stores an estimate for each rating and performs optimization using Newton's
    method.

    We use two orderings for ascents:
    - page-order: ascents are ordered by page index
    - route-order: ascents are ordered by route index

    Attributes
    ----------
    page_ratings : ndarray
        Current estimate of the natural rating of each page.
    route_ratings : ndarray
        Current estimate of the natural rating of each route.
    page_var : ndarray
        Estimate of the variance of the natural rating of each page.
    page_cov : ndarray
        Estimate of the covariance between the natural rating of each page and
        the next page.  The covariance for the last page of each climber is
        not meaningful.
    route_var : ndarray
        Estimate of the variance of the natural rating of each route.
    """

    # Private Attributes
    # ------------------
    # _page_ascents : _SlicedAscents
    #     Ascents in page order.
    # _route_ascents : _SlicedAscents
    #     Ascents in route order (no clean).
    # _pages_climber_slices : list of pairs
    #     Start and end indices in page_ratings for each climber.
    # _route_priors : NormalDistribution
    #     Distributions for the prior on each route's natural rating.
    # _climbers : list of Process
    #     Climbers (in the same order as _pages_climber_slices).

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
        num_pages = len(pages.climber)
        ascents_page_slices = _extract_slices(ascents.page, num_pages)
        pages_climber_slices = _extract_slices(pages.climber, pages.climber[-1] + 1)

        self.route_ratings = np.array(routes_rating)
        self.page_ratings = np.full(num_pages, hparams.climber_prior_mean)
        self.page_var = np.empty(num_pages)
        self.page_cov = np.zeros(num_pages)
        self.route_var = np.empty_like(self.route_ratings)

        self._pages_climber_slices = pages_climber_slices

        page_wins = []
        for (start, end) in ascents_page_slices:
            page_wins.append(np.add.reduce(ascents.clean[start:end]))

        self._page_ascents = _SlicedAscents(
            np.array(page_wins),
            ascents_page_slices,
            np.array(ascents.route),
            np.array(ascents.clean),
        )
        self._route_ascents = _make_route_ascents(
            ascents.clean, ascents_page_slices, ascents.route, len(routes_rating)
        )

        self._route_priors = NormalDistribution(
            self.route_ratings, hparams.route_prior_variance
        )

        climber_prior = NormalDistribution(
            hparams.climber_prior_mean, hparams.climber_prior_variance
        )

        pages_gap = _get_pages_gap(pages.timestamp)

        self._climbers = []
        for start, end in pages_climber_slices:
            climber = Process(
                hparams.climber_wiener_variance,
                climber_prior,
                pages_gap[start : end - 1],
            )
            self._climbers.append(climber)

    def update_page_ratings(self, should_update_covariance=False):
        """Update the ratings of all pages.

        Parameters
        ----------
        should_update_covariance : boolean
            If true, updates the "page_var" and "page_cov" attributes.
            This has no effect on rating estimation.
        """

        ascent_page_gammas = expand_to_slices(
            np.exp(self.page_ratings), self._page_ascents.slices
        )
        ascent_route_gammas = self.route_ratings[self._page_ascents.adversary]
        np.exp(ascent_route_gammas, ascent_route_gammas)

        bt_d1, bt_d2 = get_bt_derivatives(
            self._page_ascents.slices,
            self._page_ascents.wins,
            ascent_page_gammas,
            ascent_route_gammas,
        )

        for i, (start, end) in enumerate(self._pages_climber_slices):
            if start == end:
                continue
            climber = self._climbers[i]
            delta = climber.get_ratings_adjustment(
                self.page_ratings[start:end], bt_d1[start:end], bt_d2[start:end]
            )
            # r2 = r1 - delta
            self.page_ratings[start:end] -= delta

            if should_update_covariance:
                climber.get_covariance(
                    self.page_ratings[start:end],
                    bt_d1[start:end],
                    bt_d2[start:end],
                    self.page_var[start:end],
                    self.page_cov[start : end - 1],
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
            np.exp(self.route_ratings), self._route_ascents.slices
        )

        rascents_page_gammas = self.page_ratings[self._route_ascents.adversary]
        np.exp(rascents_page_gammas, rascents_page_gammas)

        # Bradley-Terry terms.
        d1, d2 = get_bt_derivatives(
            self._route_ascents.slices,
            self._route_ascents.wins,
            rascents_route_gammas,
            rascents_page_gammas,
        )

        # Prior terms.
        prior_d1, prior_d2 = self._route_priors.get_derivatives(self.route_ratings)
        d1 += prior_d1
        d2 += prior_d2

        delta = d1  # output parameter
        np.divide(d1, d2, delta)

        # r2 = r1 - delta
        self.route_ratings -= delta

        if should_update_variance:
            np.reciprocal(d2, self.route_var)
            np.negative(self.route_var, self.route_var)

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

        ascent_page_ratings = expand_to_slices(
            self.page_ratings, self._page_ascents.slices
        )
        ascent_route_ratings = self.route_ratings[self._page_ascents.adversary]

        # log(P) = sum over ascents[ log( exp(r_i) / (exp(r_i) + exp(r_j)) ) ]
        #        = sum[ r_winner - log( exp(r_i) + exp(r_j) ) ]

        # Collect the rating of each winner for the numerator.
        clean = self._page_ascents.clean
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
