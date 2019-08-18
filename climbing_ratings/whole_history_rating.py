"""Modified Whole-History Rating model for climbing"""

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


import collections
import numpy as np
from .bradley_terry import get_bt_derivatives
from .climber import Climber
from .gamma_distribution import GammaDistribution


def expand_to_slices(values, slices, dtype=None):
    """Expand normalized values to contiguous blocks

    Parameters
    ----------
    values : ndarray
        The normalized values.
    slices : list of pairs
        The (start, end) pairs corresponding to a slice in the output.  The
        implied slices must be contiguous and in ascending order.
    """
    _, n = slices[-1]
    expanded = np.empty([n], dtype=dtype)
    for i, (start, end) in enumerate(slices):
        expanded[start:end] = values[i]
    return expanded


class Ascents(collections.namedtuple("Ascents", ["wins", "slices", "adversary"])):
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
    """


def make_route_ascents(ascents_clean, ascents_page_slices, ascents_route):
    """Create a permutation of ascents in route-order.

    Parameters
    ----------
    ascents_clean : array_like of float
        1 if the ascent was clean, 0 otherwise.
    ascents_route : ndarray of intp
        Route index of each ascent.
    ascents_page : ndarray of intp
        Page index of each ascent.

    Returns
    -------
    rascents : Ascents
        Ascents ordered by (and sliced by) route.
    """
    num_ascents = len(ascents_route)
    route_wins = []
    rascents_route_slices = []
    rascents_page = [0] * num_ascents

    permutation = [(route, a) for a, route in enumerate(ascents_route)]
    permutation.sort()
    ascent_to_rascent = [0] * num_ascents

    prev_route = 0
    start = 0
    wins = 0.0
    for ra, (route, a) in enumerate(permutation):
        ascent_to_rascent[a] = ra
        if route != prev_route:
            # Flush the last block.
            rascents_route_slices.append((start, ra))
            route_wins.append(wins)
            # Start a new block.
            wins = 0.0
            start = ra
            prev_route = route

        wins += 1.0 - ascents_clean[a]

    rascents_route_slices.append((start, num_ascents))
    route_wins.append(wins)

    for page, (start, end) in enumerate(ascents_page_slices):
        for a in range(start, end):
            rascents_page[ascent_to_rascent[a]] = page

    return Ascents(
        np.array(route_wins, dtype=np.float64),
        rascents_route_slices,
        np.array(rascents_page, dtype=np.intp),
    )


def clip_ratings(ratings):
    """Clip ratings to range."""
    np.clip(
        ratings,
        WholeHistoryRating.clip_ratings[0],
        WholeHistoryRating.clip_ratings[1],
        ratings,
    )


class WholeHistoryRating:
    """Performs optimization for route and climber ratings.

    Initializes models for climbers and routes from raw ascent data.
    Stores an estimate for each rating and performs optimization using Newton's
    method.

    We use two orderings for ascents:
    - page-order: ascents are ordered by page index
    - route-order: ascents are ordered by route index

    Attributes
    ----------
    page_ratings : ndarray
        Current estimate of the rating of each page.
    route_ratings : ndarray
        Current estimate of the rating of each route.  route_ratings[0] is
        always 1.
    page_var : ndarray
        Estimate of the variance of the natural rating of each page.
    page_cov : ndarray
        Estimate of the covariance between the natural rating of each page and
        the next page.  The covariance for the last page of each climber is
        not meaningful.
    route_var : ndarray
        Estimate of the variance of the natural rating of each route.
        route_var[0] is zero by assumption.
    """

    # Minimum and maximum ratings.
    clip_ratings = (1.0 / 1024.0, 1024.0)

    # Private Attributes
    # ------------------
    # _page_ascents : Ascents
    #     Ascents in page order.
    # _route_ascents : Ascents
    #     Ascents in route order.
    # _pages_climber_slices : list of pairs
    #     Start and end indices in _page_ratings for each climber.
    # _pages_gap : ndarray
    #     Interval of time between consecutive pages of a climber.
    # _route_priors : GammaDistribution
    #     Distributions for the gamma prior on each route's rating.
    # _climbers : list of Climber
    #     Climbers (in the same order as _pages_climber_slices).

    def __init__(
        self,
        ascents_route,
        ascents_clean,
        ascents_page_slices,
        pages_climber_slices,
        routes_grade,
        pages_gap,
    ):
        """Initialize a WHR model.

        Parameters
        ----------
        ascents_route : array_like of int
            The 0-based ID of the route for each ascent.  The implied ascents
            must be in page order.
        ascents_clean : array_like of float
            1 for a clean ascent, 0 otherwise, for each ascent.  The implied
            ascents must be in page order.
        ascents_page_slices : list of pairs
            Each (start, end) entry defines the slice of the ascents for a page.
        pages_climber_slices : list of pairs
            Each (start, end) entry defines the slice of the pages for a
            climber.
        routes_grade : list
            Normalized grades of each route.  The first route has an implied
            grade of 1.
        pages_gap : array_like of float
            Interval of time between each page and the next page.  The gap for
            the last page of each climber is not used.
        """
        num_pages = len(ascents_page_slices)
        self.route_ratings = np.array(routes_grade, dtype=np.float64)
        self.page_ratings = np.full(num_pages, 1.0)
        self.page_var = np.empty(num_pages)
        self.page_cov = np.empty(num_pages)
        self.route_var = np.empty_like(self.route_ratings)
        self.route_var[0] = 0.0

        self._pages_climber_slices = pages_climber_slices

        page_wins = []
        for (start, end) in ascents_page_slices:
            page_wins.append(np.add.reduce(ascents_clean[start:end]))

        self._page_ascents = Ascents(
            np.array(page_wins), ascents_page_slices, np.array(ascents_route)
        )
        self._route_ascents = make_route_ascents(
            ascents_clean, ascents_page_slices, ascents_route
        )

        self._pages_gap = pages_gap

        self._route_priors = GammaDistribution(self.route_ratings)

        self._climbers = []
        for start, end in pages_climber_slices:
            self._climbers.append(Climber(pages_gap[start : end - 1]))

    def update_page_ratings(self, should_update_covariance=False):
        """Update the ratings of all pages.

        Parameters
        ----------
        should_update_covariance : boolean
            If true, updates the "page_var" and "page_cov" attributes.
            This has no effect on rating estimation.
        """

        ascent_page_ratings = expand_to_slices(
            self.page_ratings, self._page_ascents.slices
        )
        ascent_route_ratings = self.route_ratings[self._page_ascents.adversary]

        bt_d1, bt_d2 = get_bt_derivatives(
            self._page_ascents.slices,
            self._page_ascents.wins,
            ascent_page_ratings,
            ascent_route_ratings,
        )

        for i, (start, end) in enumerate(self._pages_climber_slices):
            climber = self._climbers[i]
            delta = climber.get_ratings_adjustment(
                self.page_ratings[start:end], bt_d1[start:end], bt_d2[start:end]
            )
            # r2 = r1 - delta
            # gamma2 = exp(log(gamma1) - delta) = gamma exp(-delta)
            np.negative(delta, delta)
            np.exp(delta, delta)
            self.page_ratings[start:end] *= delta

            if should_update_covariance:
                climber.get_covariance(
                    self.page_ratings[start:end],
                    bt_d1[start:end],
                    bt_d2[start:end],
                    self.page_var[start:end],
                    self.page_cov[start : end - 1],
                )

        clip_ratings(self.page_ratings)

    def update_route_ratings(self, should_update_variance=False):
        """Update the ratings of all routes.

        Parameters
        ----------
        should_update_variance : boolean
            If true, updates the "route_var" attribute.  This has no effect on
            rating estimation.
        """

        rascents_route_ratings = expand_to_slices(
            self.route_ratings, self._route_ascents.slices
        )

        rascents_page_ratings = self.page_ratings[self._route_ascents.adversary]

        # Bradley-Terry terms.
        d1, d2 = get_bt_derivatives(
            self._route_ascents.slices,
            self._route_ascents.wins,
            rascents_route_ratings,
            rascents_page_ratings,
        )

        # Gamma terms.
        gamma_d1, gamma_d2 = self._route_priors.get_derivatives(self.route_ratings)
        d1 += gamma_d1
        d2 += gamma_d2

        delta = d1[1:]  # output parameter
        np.divide(d1[1:], d2[1:], delta)

        # r2 = r1 - delta
        # gamma2 = exp(log(gamma1) - delta) = gamma exp(-delta)
        np.negative(delta, delta)
        np.exp(delta, delta)
        self.route_ratings[1:] *= delta
        clip_ratings(self.route_ratings)

        if should_update_variance:
            np.reciprocal(d2[1:], self.route_var[1:])
            np.negative(self.route_var[1:], self.route_var[1:])

    def update_ratings(self):
        """Update ratings for all routes and pages"""
        # Update pages first because we have better priors/initial values for
        # the routes.
        self.update_page_ratings()
        self.update_route_ratings()
