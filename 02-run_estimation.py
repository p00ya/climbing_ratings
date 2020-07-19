#!/usr/bin/env python3

"""Reads a pre-processed table of ascents and iteratively runs a WHR model.

Should be called as:

    run_estimation.py DATA_DIR

DATA_DIR should be a directory containing the files 'ascents.csv', 'routes.csv',
'pages.csv'.  The two files 'route_ratings.csv' and 'page_ratings.csv' will be
written to the output directory (which defaults to DATA_DIR).  Each of the
files are described below.

The first line of each of the CSV files is assumed to be a header.

ascents.csv
-----------
route
    0-based route ID of the route ascended.  Acts as an index into the routes
    table.
clean
    1 for a clean ascent, 0 otherwise.
page
    0-based page ID.  A page identifies a climber at an interval in time.  Acts
    as an index into the pages table.
style_page
    0-based style-page ID.  A style-page identifies a climber ascending in a
    particular style, at an interval in time.  Acts as an index into the
    style-pages table.  -1 indicates no style page (an ascent in the base
    style).

pages.csv
---------
A page identifies a climber at an interval in time.  Pages must be sorted by
lexicographically by climber and timestamp.

climber
    0-based climber ID.  While there is no separate climbers table, the IDs
    should behave like an index into such a table.
timestamp
    Number representing the time of all ascents for this page.  The time unit
    must be consistent with the "wiener-variance" option.

style_pages.csv
---------------
A style-page identifies a climber ascending in a particular style, at an
interval in time.  Pages must be sorted lexicographically by climber
and timestamp.

climber_style
    0-based climber/style ID.  This ID should be unique for each combination
    of climber and style, and IDs for the same climber should be contiguous.
timestamp
    Number representing the time of all ascents for this page.  The time unit
    must be consistent with the "wiener-variance" option.

routes.csv
----------
route
    Arbitrary route tag.  Not used.
rating
    Initial natural rating for each route.  The ratings should be reals, with
    the interpretation under the Bradley-Terry model that for grades A and B,
    if a climber can cleanly ascend grade A with even probability, then the
    probability of cleanly ascending grade B = exp(B) / (exp(A) + exp(B)).

route_ratings.csv
-----------------
Output file.

route
    Tag from routes.csv.
rating
    WHR natural rating for each route.
var
    Variance of the natural rating for each route.

page_ratings.csv
----------------
Output file.

climber
    0-based climber ID for each page.  Consistent with pages.csv.
rating
    WHR natural rating for each page.
var
    Variance of the natural rating for each page.
cov
    Covariance of the natural rating for each page with the natural rating
    of the climber's next page.  The value for the last page of a climber is
    meaningless.

style_page_ratings.csv
----------------------
Output file.

climber_style
    0-based climber/style ID for each style-page.  Consistent with
    style_pages.csv.
rating
    WHR natural rating for each style-page.
var
    Variance of the natural rating for each style-page.
cov
    Covariance of the natural rating for each style-page with the natural rating
    of the climber-style's next style-page.  The value for the last style-page
    of a climber-style is meaningless.
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

import argparse
import csv
import numpy as np
import os
import sys
from climbing_ratings.whole_history_rating import (
    AscentsTable,
    Hyperparameters,
    PagesTable,
    WholeHistoryRating,
)


def read_ascents(dirname):
    """Read the ascents table."""
    filename = os.path.join(dirname, "ascents.csv")
    routes = []
    cleans = []
    pages = []
    style_pages = []
    with open(filename, newline="") as fp:
        reader = iter(csv.reader(fp))

        assert next(reader) == ["route", "clean", "page", "style_page"]
        for line in reader:
            route, clean, page, style_page = line
            routes.append(int(route))
            cleans.append(float(clean))
            pages.append(int(page))
            style_pages.append(int(style_page))

    return (routes, cleans, pages, style_pages)


def read_routes(dirname):
    """Read the routes table."""
    filename = os.path.join(dirname, "routes.csv")
    names = []
    ratings = []
    with open(filename, newline="") as fp:
        reader = iter(csv.reader(fp))

        assert next(reader) == ["route", "rating"]
        for line in reader:
            name, rating = line
            names.append(name)
            ratings.append(float(rating))

    return names, ratings


def read_pages(dirname):
    """Read the pages table."""
    filename = os.path.join(dirname, "pages.csv")
    climbers = []
    timestamps = []
    with open(filename, newline="") as fp:
        reader = iter(csv.reader(fp))

        assert next(reader) == ["climber", "timestamp"]
        for line in reader:
            climber, timestamp = line
            climbers.append(int(climber))
            timestamps.append(float(timestamp))

    return (climbers, timestamps)


def read_style_pages(dirname):
    """Read the style-pages table."""
    filename = os.path.join(dirname, "style_pages.csv")
    climber_styles = []
    timestamps = []
    with open(filename, newline="") as fp:
        reader = iter(csv.reader(fp))

        assert next(reader) == ["climber_style", "timestamp"]
        for line in reader:
            climber_style, timestamp = line
            climber_styles.append(int(climber_style))
            timestamps.append(float(timestamp))

    return (climber_styles, timestamps)


def write_route_ratings(dirname, routes_name, route_ratings, route_var):
    filename = os.path.join(dirname, "route_ratings.csv")
    with open(filename, "w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n", delimiter=",")
        writer.writerow(["route", "rating", "var"])
        for route, rating, var in zip(routes_name, route_ratings, route_var):
            writer.writerow([route, rating, var])


def _write_page_ratings(filename, climber_field, dirname, pages_climber, page):
    filename = os.path.join(dirname, filename)
    with open(filename, "w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n", delimiter=",")
        writer.writerow([climber_field, "rating", "var", "cov"])
        for climber, rating, var, cov in zip(
            pages_climber, page.ratings, page.var, page.cov
        ):
            writer.writerow([climber, rating, var, cov])


def write_page_ratings(dirname, pages_climber, page):
    _write_page_ratings("page_ratings.csv", "climber", dirname, pages_climber, page)


def write_style_page_ratings(dirname, pages_climber, page):
    _write_page_ratings(
        "style_page_ratings.csv", "climber_style", dirname, pages_climber, page
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="""
        Reads a pre-processed table of ascents and iteratively runs a WHR model.
        """
    )
    parser.add_argument("data_dir", metavar="DATA_DIR", help="input data directory")
    parser.add_argument(
        "--output", metavar="DIR", help="output data directory; defaults to DATA_DIR"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="do not write output files"
    )
    parser.add_argument(
        "--max-iterations",
        metavar="N",
        type=int,
        default=64,
        help="maximum number of Newton-Raphson iterations",
    )
    parser.add_argument(
        "--wiener-variance",
        metavar="w2",
        type=float,
        # Assume a 1 point variance over 1 year, with time units of 1s.
        default=1.0 / 86400.0 / 364.0,
        help="variance of climber ratings per time unit",
    )
    parser.add_argument(
        "--climber-prior-mean",
        metavar="mu",
        type=float,
        default=0.0,
        help="mean of climbers' natural ratings prior",
    )
    parser.add_argument(
        "--climber-prior-variance",
        metavar="sigma2",
        type=float,
        default=1.0,
        help="variance of climbers' natural ratings prior",
    )
    parser.add_argument(
        "--route-prior-variance",
        metavar="sigma2",
        type=float,
        default=1.0,
        help="variance of routes' natural ratings prior",
    )
    parser.add_argument(
        "--style-prior-variance",
        metavar="sigma2",
        type=float,
        # Should be less than the climber prior variance so that the majority
        # of variation between climbers is captured in their base rating.
        default=0.01,
        help="variance of climber style natural ratings priors",
    )
    parser.add_argument(
        "--style-wiener-variance",
        metavar="w2",
        type=float,
        # Should be less than climber Wiener variance so that the majority
        # of variation in a climber's strength over time is captured in the
        # climber's base rating.
        default=0.01 / 86400.0 / 364.0,
        help="variance of climber style ratings per time unit",
    )
    return parser.parse_args(argv)


def main(argv):
    """Read tables and perform estimation."""
    args = parse_args(argv[1:])
    data = args.data_dir

    ascents = AscentsTable(*read_ascents(data))
    pages = PagesTable(*read_pages(data))
    style_pages = PagesTable(*read_style_pages(data))
    routes_name, routes_rating = read_routes(data)

    hparams = Hyperparameters(
        args.climber_prior_mean,
        args.climber_prior_variance,
        args.wiener_variance,
        args.route_prior_variance,
        args.style_prior_variance,
        args.style_wiener_variance,
    )

    whr = WholeHistoryRating(hparams, ascents, pages, style_pages, routes_rating)

    np.seterr(all="ignore")
    last_log_lik = float("-inf")
    for i in range(args.max_iterations):
        whr.update_ratings()
        if i % 8 == 0:
            log_lik = whr.get_log_likelihood()
            print(log_lik)
            if 0.0 < abs(log_lik - last_log_lik) < 1.0:
                # Detect early convergence.
                break
            last_log_lik = log_lik

    whr.update_ratings(True)

    print(whr.get_log_likelihood())

    if not args.dry_run:
        output = args.output
        if output is None:
            output = data

        write_route_ratings(output, routes_name, whr.route_ratings, whr.route_var)
        write_page_ratings(output, pages.climber, whr.page)
        write_style_page_ratings(output, pages.climber, whr.style_page)


if __name__ == "__main__":
    main(sys.argv)
