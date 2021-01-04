#!/usr/bin/env python3
"""Reads a pre-processed table of ascents and iteratively runs a WHR model.

Should be called as:

    run_estimation.py DATA_DIR

DATA_DIR should be a directory containing the files 'ascents.csv', 'routes.csv',
'pages.csv', and 'style_pages.csv'.  The three files 'route_ratings.csv',
'page_ratings.csv' and 'style_page_ratings.csv' will be written to the output
directory (which defaults to DATA_DIR).  Each of the files are described below.

The first line of each of the CSV files is assumed to be a header.  The columns
can be in any order, and additional columns are permitted.

Some columns are optional, in which case the entire column may be omitted.  If
the column is present in the input, every row must have a value.

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
    style).  Column is optional (values default to -1).

pages.csv
---------
A page identifies a climber at an interval in time.  Pages must be sorted
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

This file may be omitted.  If it does not exist, no style-pages are defined, and
all ascents must use the base style.

climber_style
    0-based climber/style ID.  This ID should be unique for each combination
    of climber and style, and IDs for the same climber should be contiguous.
timestamp
    Number representing the time of all ascents for this page.  The time unit
    must be consistent with the "wiener-variance" option.

routes.csv
----------
route
    Arbitrary route tag.  Copied to the output route_ratings.csv, but otherwise
    not used.
rating
    Initial natural rating for each route.  The ratings should be reals, with
    the interpretation under the Bradley-Terry model that for grades A and B,
    if a climber can cleanly ascend grade A with even probability, then the
    probability of cleanly ascending grade B = exp(B) / (exp(A) + exp(B)).
    Column is optional (values default to 0).

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
import copy
import csv
import math
import numpy as np
import os
import sys
from climbing_ratings.whole_history_rating import (
    AscentsTable,
    Hyperparameters,
    PagesTable,
    WholeHistoryRating,
)


class TableReader:
    """Reads a table from a CSV file.

    The reader is robust to additional columns, missing columns, and changes in
    column order.
    """

    __slots__ = ("_colspecs",)

    def __init__(self, colspecs):
        """Configures a reader that expects the given columns.

        Parameters
        ----------
        colspecs : list
            Output column specifications.  Each member is a tuple of
            (name, type, default).  Name is a string.  The type can be called
            like a function with a string argument (like int or float).  The
            default should be a value of type, used if the input table does not
            contain the corresponding column.  If default is None, then the
            read method will throw an error if the column is missing in the
            input.
        """
        self._colspecs = colspecs
        for i, (name, type_, default) in enumerate(colspecs):
            assert isinstance(name, str)
            assert isinstance(type_, type)
            assert default is None or isinstance(default, type_)

    def read(self, filename):
        """Read the table from the given CSV file.

        Parameters
        ----------
        filename : string
            The file name of the input CSV file.

        Returns
        -------
        tuple of lists
            Returns a tuple with the same number of members as the column
            specifications the reader was initialized with.  Each of the
            inner lists will have the same length, corresponding to the number
            of rows in the CSV (not including the header).
        """
        output = tuple([[] for _ in self._colspecs])

        with open(filename, newline="") as fp:
            reader = iter(csv.reader(fp))

            header = next(reader)
            input_index = self.__get_input_index(filename, header)

            for line in reader:
                # Append a value for this row to each of the output columns.
                for i, (_, type_, value) in enumerate(self._colspecs):
                    if input_index[i] >= 0:
                        value = type_(line[input_index[i]])

                    output[i].append(value)

        return output

    def __get_input_index(self, filename, header):
        """Returns a mapping from output columns to input columns.

        Parameters
        ----------
        filename : str
            The input CSV file, for error messages.
        header : list of str
            Names of each of the columns in the input table.

        Returns
        -------
        list of int
            For the column with index i in the output table, the return value[i]
            is the index of the column with the same name in the input table.
            If the input table does not contain the given column, value[i] will
            be -1.
        """
        # d[name] is the index of column "name" in the input table.
        d = dict([(name, i) for i, name in enumerate(header)])
        index = []
        for (name, _, default) in self._colspecs:
            if name in d:
                index.append(d[name])
            else:
                if default is None:
                    raise ValueError(f'{filename}: missing required column "{name}"')
                index.append(-1)

        return index


def read_ascents(dirname):
    """Read the ascents table."""
    reader = TableReader(
        [
            ("route", int, None),
            ("clean", float, None),
            ("page", int, None),
            ("style_page", int, -1),
        ]
    )
    return reader.read(os.path.join(dirname, "ascents.csv"))


def read_routes(dirname):
    """Read the routes table."""
    reader = TableReader(
        [
            ("route", str, ""),
            ("rating", float, 0.0),
        ]
    )
    return reader.read(os.path.join(dirname, "routes.csv"))


def read_pages(dirname):
    """Read the pages table."""
    reader = TableReader(
        [
            ("climber", int, None),
            ("timestamp", float, None),
        ]
    )
    return reader.read(os.path.join(dirname, "pages.csv"))


def read_style_pages(dirname):
    """Read the style-pages table."""
    filename = os.path.join(dirname, "style_pages.csv")
    if not os.path.exists(filename):
        # Tolerate a missing style-pages table.
        return ([], [])

    reader = TableReader(
        [
            ("climber_style", int, None),
            ("timestamp", float, None),
        ]
    )
    return reader.read(os.path.join(dirname, "style_pages.csv"))


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


def guess_iterations(num_entities):
    """Guesses the number of iterations required for convergence.

    This is just a heuristic; the actual number depends heavily on the data
    and hyperparameters.

    Parameters
    ----------
    num_entities : int
        The number of ratings that need to be estimated.

    Returns
    -------
    int
        An estimate of the number of iterations of Newton's method required for
        the WHR estimates to converge.
    """
    return max(64, 2 * int(math.sqrt(1 + num_entities)))


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
        "--progress", action="store_true", help="print log-likelihood periodically"
    )
    parser.add_argument(
        "--max-iterations",
        metavar="N",
        type=int,
        default=None,
        help="maximum number of Newton-Raphson iterations (default scales with input)",
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
        default=4.0,
        help="variance of climbers' natural ratings prior",
    )
    parser.add_argument(
        "--route-prior-variance",
        metavar="sigma2",
        type=float,
        default=4.0,
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

    if args.max_iterations is None:
        max_iterations = guess_iterations(
            len(routes_rating) + pages.climber.size + style_pages.climber.size
        )
    else:
        max_iterations = args.max_iterations

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

    max_log_lik = float("-inf")
    best_model = whr

    if args.progress:
        print("iteration,log_lik")

    for i in range(max_iterations + 1):
        if i % 8 == 0 or i == args.max_iterations:
            log_lik = whr.get_log_likelihood()

            if args.progress:
                print(f"{i},{log_lik}")

            if log_lik > max_log_lik:
                max_log_lik = log_lik
                best_model = copy.copy(whr)
            elif log_lik - max_log_lik > 1.0:
                print(f"warning: divergent model (iteration {i})", file=sys.stderr)

            if 0.0 < abs(log_lik - last_log_lik) < 1.0:
                # Detect early convergence.
                break

            last_log_lik = log_lik

        if i < max_iterations:
            whr.update_ratings()
        else:
            print(
                f"warning: reached iteration limit {max_iterations}",
                file=sys.stderr,
            )

    best_model.update_covariance()

    if not args.dry_run:
        output = args.output
        if output is None:
            output = data

        write_route_ratings(
            output, routes_name, best_model.route_ratings, best_model.route_var
        )
        write_page_ratings(output, pages.climber, best_model.page)
        write_style_page_ratings(output, pages.climber, best_model.style_page)


if __name__ == "__main__":
    main(sys.argv)
