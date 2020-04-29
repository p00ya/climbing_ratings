"""Tests for the whole_history_rating module"""

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

import unittest
import math
import numpy as np
from ..whole_history_rating import (
    AscentsTable,
    Hyperparameters,
    PagesTable,
    WholeHistoryRating,
    _extract_slices,
    _make_route_ascents,
)
from .assertions import assert_close


# Hyperparameters defaults.
_hparams = Hyperparameters(0.0, 1.0, 1.0, 1.0)


class TestWholeHistoryRatingFunctions(unittest.TestCase):
    """Tests for functions in the whole_history_rating module"""

    def test_extract_slices(self):
        """Test _extract_slices()"""
        self.assertSequenceEqual([(0, 1)], _extract_slices([0], 1))
        self.assertSequenceEqual([(0, 2)], _extract_slices([0, 0], 1))
        self.assertSequenceEqual([(0, 1), (1, 2)], _extract_slices([0, 1], 2))
        self.assertSequenceEqual([(0, 0)], _extract_slices([], 1))
        self.assertSequenceEqual([(0, 0), (0, 1)], _extract_slices([1], 2))

    def test_make_route_ascents(self):
        """Test _make_route_ascents()"""
        ascents_clean = [0, 0, 0, 0, 0]
        ascents_page_slices = [(0, 5)]
        ascents_route = [0, 1, 0, 1, 0]

        ascents = _make_route_ascents(
            ascents_clean, ascents_page_slices, ascents_route, 2
        )
        self.assertSequenceEqual([3.0, 2.0], ascents.wins.tolist())
        self.assertSequenceEqual([(0, 3), (3, 5)], ascents.slices)
        self.assertSequenceEqual([0, 0, 0, 0, 0], ascents.adversary.tolist())

    def test_make_route_ascents_sparse(self):
        """Test _make_route_ascents() for routes without ascents"""
        ascents_clean = [0, 0, 0, 0, 0]
        ascents_page_slices = [(0, 5)]
        ascents_route = [1, 2, 1, 2, 1]

        ascents = _make_route_ascents(
            ascents_clean, ascents_page_slices, ascents_route, 4
        )
        self.assertSequenceEqual([0.0, 3.0, 2.0, 0.0], ascents.wins.tolist())
        self.assertSequenceEqual([(0, 0), (0, 3), (3, 5), (5, 5)], ascents.slices)
        self.assertSequenceEqual([0, 0, 0, 0, 0], ascents.adversary.tolist())


class TestWholeHistoryRatingStable(unittest.TestCase):
    """Tests for the WholeHistoryRating class with stable ratings.

    1 climber, 1 page, 3 routes at grade "1", all with 1 clean and 1
    non-clean ascent.
    """

    def setUp(self):
        np.seterr(all="raise")
        self.assert_close = assert_close.__get__(self, self.__class__)
        ascents = AscentsTable(
            route=[0, 0, 1, 1, 2, 2],
            clean=np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            page=[0, 0, 0, 0, 0, 0],
            style_page=[-1, -1, -1, -1, -1, -1],
        )
        pages = PagesTable(climber=[0], timestamp=[0.0])
        routes_grade = [0.0, 0.0, 0.0]
        self.whr = WholeHistoryRating(_hparams, ascents, pages, routes_grade)

    def test_initialization(self):
        """Test WholeHistoryRating initialization"""
        route_ratings = self.whr.route_ratings
        self.assert_close([0.0, 0.0, 0.0], route_ratings, "route_ratings")

        page_ratings = self.whr.page_ratings
        self.assert_close([0.0], page_ratings, "page_ratings")

    def test_update_page_ratings(self):
        """Test WholeHistoryRating.update_page_ratings"""
        self.whr.update_page_ratings(True)
        self.assert_close([0.0, 0.0], self.whr.page_ratings, "page_ratings")
        self.assert_close([0.4], self.whr.page_var, "page_var")

    def test_update_route_ratings(self):
        """Test WholeHistoryRating.update_route_ratings is stable"""
        self.whr.update_route_ratings(True)
        # Ratings should not change: both ascents had a 50% probability assuming
        # the initial ratings.
        self.assert_close([0.0, 0.0, 0.0], self.whr.route_ratings, "route_ratings")
        self.assert_close(
            [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], self.whr.route_var, "route_var"
        )

    def test_update_ratings(self):
        """Test WholeHistoryRating.update_ratings"""
        self.whr.update_ratings(True)
        self.assert_close([0.4], self.whr.page_var, "page_var")
        self.assert_close(
            [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], self.whr.route_var, "route_var"
        )

    def test_get_log_likelihood(self):
        """Test WholeHistoryRating.get_log_likelihood"""
        log_lik = self.whr.get_log_likelihood()
        self.assert_close(6.0 * math.log(0.5), log_lik, "log_lik")


class TestWholeHistoryRatingStableMultipage(unittest.TestCase):
    """Tests for the WholeHistoryRating class with multiple pages.

    1 climber, 2 pages, 3 routes at grade "1", all with 1 clean and 1
    non-clean ascent.
    """

    def setUp(self):
        np.seterr(all="raise")
        self.assert_close = assert_close.__get__(self, self.__class__)
        ascents = AscentsTable(
            route=[0, 0, 1, 1, 2, 2],
            clean=np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            page=[0, 0, 1, 1, 1, 1],
            style_page=[-1, -1, -1, -1, -1, -1],
        )
        pages = PagesTable(climber=[0, 0], timestamp=np.array([0.0, 1.0]))
        routes_grade = [0.0, 0.0, 0.0]
        self.whr = WholeHistoryRating(_hparams, ascents, pages, routes_grade)

    def test_update_page_ratings(self):
        """Test WholeHistoryRating.update_page_ratings is stable"""
        self.whr.update_page_ratings(True)
        self.assert_close([0.0, 0.0], self.whr.page_ratings, "page_ratings")
        self.assert_close([0.5, 0.625], self.whr.page_var, "page_var")

    def test_get_log_likelihood(self):
        """Test WholeHistoryRating.get_log_likelihood"""
        log_lik = self.whr.get_log_likelihood()
        self.assert_close(6.0 * math.log(0.5), log_lik, "log_lik")


class TestWholeHistoryRatingUpdates(unittest.TestCase):
    """Tests for the WholeHistoryRating class with updates.

    1 climber, 1 page, 3 routes at grade "1", with all clean ascents.
    """

    def setUp(self):
        np.seterr(all="raise")
        self.assert_close = assert_close.__get__(self, self.__class__)
        ascents = AscentsTable(
            route=[0, 0, 1, 1, 2, 2],
            clean=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            page=[0, 0, 0, 0, 0, 0],
            style_page=[-1, -1, -1, -1, -1, -1],
        )
        pages = PagesTable(climber=[0], timestamp=np.array([0.0]))
        routes_grade = [0.0, 0.0, 0.0]
        self.whr = WholeHistoryRating(_hparams, ascents, pages, routes_grade)

    def test_update_page_ratings(self):
        """Test WholeHistoryRating.update_page_ratings converges"""
        for _ in range(4):
            self.whr.update_page_ratings(True)
        self.assert_close([1.29253960], self.whr.page_ratings, "page_ratings")
        self.assert_close([0.49650051], self.whr.page_var, "page_var")


class TestWholeHistoryRatingUpdatesDifferentGrades(unittest.TestCase):
    """Tests for the WholeHistoryRating class with updates.

    1 climber, 1 page, 3 routes with grades 1, 2 and 2, with all clean ascents.
    """

    def setUp(self):
        np.seterr(all="raise")
        self.assert_close = assert_close.__get__(self, self.__class__)
        ascents = AscentsTable(
            route=[0, 0, 1, 1, 2, 2],
            clean=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            page=[0, 0, 0, 0, 0, 0],
            style_page=[-1, -1, -1, -1, -1, -1],
        )
        pages = PagesTable(climber=[0], timestamp=np.array([0.0]))
        routes_grade = np.log([1.0, 2.0, 2.0])
        self.whr = WholeHistoryRating(_hparams, ascents, pages, routes_grade)

    def test_update_page_ratings(self):
        """Test WholeHistoryRating.update_page_ratings"""
        for _ in range(4):
            self.whr.update_page_ratings(True)
        self.assert_close([1.54631420], self.whr.page_ratings, "page_ratings")
        self.assert_close([0.47001792], self.whr.page_var, "page_var")


class TestWholeHistoryRatingUpdatesMultipage(unittest.TestCase):
    """Tests for the WholeHistoryRating class with multiple pages.

    1 climber, 2 pages, 3 routes at grade "1", with all clean ascents.
    """

    def setUp(self):
        np.seterr(all="raise")
        self.assert_close = assert_close.__get__(self, self.__class__)
        ascents = AscentsTable(
            route=[0, 0, 1, 1, 2, 2],
            clean=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            page=[0, 0, 0, 0, 1, 1],
            style_page=[-1, -1, -1, -1, -1, -1],
        )
        pages = PagesTable(climber=[0, 0], timestamp=np.array([0.0, 1.0]))
        routes_grade = [0.0, 0.0, 0.0]
        self.whr = WholeHistoryRating(_hparams, ascents, pages, routes_grade)

    def test_update_page_ratings(self):
        """Test WholeHistoryRating.update_page_ratings converges"""
        for _ in range(4):
            self.whr.update_page_ratings(True)
        self.assert_close([1.2394699, 1.5808267], self.whr.page_ratings, "page_ratings")
        self.assert_close([0.5216223, 1.0962046], self.whr.page_var, "page_var")
