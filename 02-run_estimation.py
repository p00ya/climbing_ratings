#!/usr/bin/env python3

"""Reads a pre-processed table of ascents and iteratively runs a WHR model.

This is just a stub; see the documentation for the climbing_ratings.__main__
module instead.
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

from climbing_ratings.__main__ import main
import sys

if __name__ == "__main__":
    print(
        'warning: 02-run_estimation.py is deprecated, use "python3 -m climbing_ratings" instead',
        file=sys.stderr,
    )
    main(sys.argv)
