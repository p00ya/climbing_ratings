#!/usr/bin/env python3

"""Reads a pre-processed table of ascents and iteratively runs a WHR model.

Should be called as:

    run_estimation.py DATA

DATA should be a directory containing the files 'ascents.csv', 'routes.csv',
'pages.csv'.  The two files 'route_ratings.csv' and 'page_ratings.csv' will be
written to the directory.  Each of the files are described below.

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
grade
    Initial "grade" estimate for each route.  The first route should have a
    grade of 1.  The other grades should be positive reals, with the
    interpretation under the Bradley-Terry model that for grades A and B, if
    a climber can cleanly ascend grade A with even probability, then the
    probability of cleanly ascending grade B = B / (A + B).

route_ratings.csv
-----------------
Output file.

route
    Tag from routes.csv
gamma
    WHR gamma-rating for each route.
var
    Variance of the natural (log gamma) rating for each route.

page_ratings.csv
----------------
Output file.

climber
    0-based climber ID for each page.  Consistent with pages.csv.
gamma
    WHR gamma-rating for each page.
var
    Variance of the natural (log gamma) rating for each page.
"""

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

import csv
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
        reader = csv.reader(fp)

        for i, line in enumerate(reader):
            if i == 0:
                assert line == ["route", "clean", "page"]
                continue
            route, clean, page = line
            routes.append(int(route))
            cleans.append(float(clean))
            pages.append(int(page))

    return (routes, cleans, pages)


def read_routes(dirname):
    """Read the routes table."""
    filename = "%s/routes.csv" % dirname
    names = []
    grades = []
    with open(filename, newline="") as fp:
        reader = csv.reader(fp)

        for i, line in enumerate(reader):
            if i == 0:
                assert line == ["route", "grade"]
                continue
            name, grade = line
            names.append(name)
            grades.append(float(grade))

    return names, grades


def read_pages(dirname):
    """Read the pages table."""
    filename = "%s/pages.csv" % dirname
    climbers = []
    gaps = []
    with open(filename, newline="") as fp:
        reader = csv.reader(fp)

        for i, line in enumerate(reader):
            if i == 0:
                assert line == ["climber", "gap"]
                continue
            climber, gap = line
            climbers.append(int(climber))
            gaps.append(float(gap))

    return (climbers, gaps)


def extract_slices(values):
    """Extract slices of contiguous values."""
    slices = []
    prev_value = 0
    start = 0
    for i, value in enumerate(values):
        if value != prev_value:
            slices.append((start, i))
            start = i
            prev_value = value
    slices.append((start, len(values)))
    return slices


def write_route_ratings(dirname, routes_name, route_ratings, route_var):
    filename = "%s/route_ratings.csv" % dirname
    with open(filename, "w", newline="") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(["route", "gamma", "var"])
        for route, rating, var in zip(routes_name, route_ratings, route_var):
            writer.writerow([route, rating, var])


def write_page_ratings(dirname, pages_climber, page_ratings, page_var):
    filename = "%s/page_ratings.csv" % dirname
    with open(filename, "w", newline="") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(["climber", "gamma", "var"])
        for climber, rating, var in zip(pages_climber, page_ratings, page_var):
            writer.writerow([climber, rating, var])


def main(argv):
    """Read tables and perform estimation."""
    data = argv[1]

    ascents_route, ascents_clean, ascents_page = read_ascents(data)
    ascents_page_slices = extract_slices(ascents_page)

    routes_name, routes_grade = read_routes(data)

    pages_climber, pages_gap = read_pages(data)
    pages_climber_slices = extract_slices(pages_climber)

    # Assume a variance over 1 year of 1 point, and pages at 1-week intervals.
    Climber.wiener_variance = 1.0 / 52.0

    whr = WholeHistoryRating(
        ascents_route,
        ascents_clean,
        ascents_page_slices,
        pages_climber_slices,
        routes_grade,
        pages_gap,
    )

    np.seterr(all="ignore")
    for _ in range(100):
        whr.update_ratings()

    whr.update_page_ratings(should_update_covariance=True)
    whr.update_route_ratings(should_update_variance=True)

    write_route_ratings(data, routes_name, whr.route_ratings, whr.route_var)
    write_page_ratings(data, pages_climber, whr.page_ratings, whr.page_var)


if __name__ == "__main__":
    main(sys.argv)
