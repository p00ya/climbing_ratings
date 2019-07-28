#!/usr/bin/env python3

"""Reads a pre-processed table of ascents and iteratively runs a WHR model.

Should be called as:

    run_estimation.py ascents.csv routes.csv pages.csv \
        route_ratings.csv page_ratings.csv

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
rating
    WHR (gamma) rating for each route.


page_ratings.csv
----------------
Output file.

climber
    0-based climber ID for each page.  Consistent with pages.csv.
rating
    WHR (gamma) rating for each page.
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


def read_ascents(filename):
    """Read the ascents table."""
    routes = []
    cleans = []
    pages = []
    with open(filename, newline='') as fp:
        reader = csv.reader(fp)

        for i, line in enumerate(reader):
            if i == 0:
                continue
            route, clean, page = line
            routes.append(int(route))
            cleans.append(float(clean))
            pages.append(int(page))

    return (routes, cleans, pages)


def read_routes(filename):
    """Read the routes table."""
    names = []
    grades = []
    with open(filename, newline='') as fp:
        reader = csv.reader(fp)

        for i, line in enumerate(reader):
            if i == 0:
                continue
            name, grade = line
            names.append(name)
            grades.append(float(grade))

    return names, grades


def read_pages(filename):
    """Read the pages table."""
    climbers = []
    gaps = []
    with open(filename, newline='') as fp:
        reader = csv.reader(fp)

        for i, line in enumerate(reader):
            if i == 0:
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


def write_route_ratings(filename, routes_name, route_ratings):
    with open(filename, 'w', newline='') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(['route', 'rating'])
        for route, rating in zip(routes_name, route_ratings):
            writer.writerow([route, rating])


def write_page_ratings(filename, pages_climber, page_ratings):
    with open(filename, 'w', newline='') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(['climber', 'rating'])
        for climber, rating in zip(pages_climber, page_ratings):
            writer.writerow([climber, rating])


def main(argv):
    """Read tables and perform estimation."""
    ascents_route, ascents_clean, ascents_page = read_ascents(argv[1])
    ascents_page_slices = extract_slices(ascents_page)

    routes_name, routes_grade = read_routes(argv[2])

    pages_climber, pages_gap = read_pages(argv[3])
    pages_climber_slices = extract_slices(pages_climber)

    # Assume a variance over 1 year of 1 point, and pages at 1-week intervals.
    Climber.wiener_variance = 1. / 52.

    whr = WholeHistoryRating(
        ascents_route, ascents_clean, ascents_page_slices,
        pages_climber_slices, routes_grade, pages_gap)

    np.seterr(all='ignore')
    for _ in range(100):
        whr.update_ratings()

    write_route_ratings(argv[4], routes_name, whr.route_ratings)
    write_page_ratings(argv[5], pages_climber, whr.page_ratings)


if __name__ == "__main__":
    main(sys.argv)
