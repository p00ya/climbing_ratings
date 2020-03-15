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

pages.csv
---------
climber
    0-based climber ID.  While there is no separate climbers table, the IDs
    should behave like an index into such a table.
gap
    Time interval from this page to the next page.  The gap for the last page
    of each climber is not used.

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
    Tag from routes.csv
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
import itertools
import numpy as np
import sys
from climbing_ratings.climber import Climber
from climbing_ratings.whole_history_rating import WholeHistoryRating


def read_ascents(dirname):
    """Read the ascents table."""
    filename = "%s/ascents.csv" % dirname
    routes = []
    cleans = []
    pages = []
    with open(filename, newline="") as fp:
        reader = iter(csv.reader(fp))

        assert next(reader) == ["route", "clean", "page"]
        for line in reader:
            route, clean, page = line
            routes.append(int(route))
            cleans.append(float(clean))
            pages.append(int(page))

    return (routes, cleans, pages)


def read_routes(dirname):
    """Read the routes table."""
    filename = "%s/routes.csv" % dirname
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
    filename = "%s/pages.csv" % dirname
    climbers = []
    gaps = []
    with open(filename, newline="") as fp:
        reader = iter(csv.reader(fp))

        assert next(reader) == ["climber", "gap"]
        for line in reader:
            climber, gap = line
            climbers.append(int(climber))
            gaps.append(float(gap))

    return (climbers, gaps)


def extract_slices(values, num_slices):
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


def write_route_ratings(dirname, routes_name, route_ratings, route_var):
    filename = "%s/route_ratings.csv" % dirname
    with open(filename, "w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n", delimiter=",")
        writer.writerow(["route", "rating", "var"])
        for route, rating, var in zip(routes_name, route_ratings, route_var):
            writer.writerow([route, rating, var])


def write_page_ratings(dirname, pages_climber, page_ratings, page_var):
    filename = "%s/page_ratings.csv" % dirname
    with open(filename, "w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n", delimiter=",")
        writer.writerow(["climber", "rating", "var"])
        for climber, rating, var in zip(pages_climber, page_ratings, page_var):
            writer.writerow([climber, rating, var])


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
        metavar="w",
        type=float,
        # Assume a 1 point variance over 1 year, with time units of 1 week.
        default=1.0 / 52.0,
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
        metavar="sigma",
        type=float,
        default=1.0,
        help="standard deviation of climbers' natural ratings prior",
    )
    parser.add_argument(
        "--route-prior-variance",
        metavar="sigma",
        type=float,
        default=1.0,
        help="standard deviation of routes' natural ratings prior",
    )
    return parser.parse_args(argv)


def main(argv):
    """Read tables and perform estimation."""
    args = parse_args(argv[1:])
    data = args.data_dir

    ascents_route, ascents_clean, ascents_page = read_ascents(data)
    pages_climber, pages_gap = read_pages(data)
    routes_name, routes_rating = read_routes(data)

    ascents_page_slices = extract_slices(ascents_page, len(pages_climber))
    pages_climber_slices = extract_slices(pages_climber, pages_climber[-1] + 1)

    Climber.wiener_variance = args.wiener_variance
    WholeHistoryRating.climber_mean = args.climber_prior_mean
    WholeHistoryRating.climber_variance = args.climber_prior_variance
    WholeHistoryRating.route_variance = args.route_prior_variance

    whr = WholeHistoryRating(
        ascents_route,
        ascents_clean,
        ascents_page_slices,
        pages_climber_slices,
        routes_rating,
        pages_gap,
    )

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

    whr.update_page_ratings(should_update_covariance=True)
    whr.update_route_ratings(should_update_variance=True)

    print(whr.get_log_likelihood())

    if not args.dry_run:
        output = args.output
        if output is None:
            output = data

        write_route_ratings(output, routes_name, whr.route_ratings, whr.route_var)
        write_page_ratings(output, pages_climber, whr.page_ratings, whr.page_var)


if __name__ == "__main__":
    main(sys.argv)
