# Climbing Ratings

Climbing Ratings is software that estimates ratings for the sport of rock climbing.  The ratings can be used to predict route difficulty and climber performance on a particular route.

Climbing Ratings is Copyright 2019-2022 the Contributors to the Climbing Ratings project.

The algorithms are based on the "WHR" paper:

> Rémi Coulom, "Whole-History Rating: A Bayesian Rating System for Players of Time-Varying Strength", <https://www.remi-coulom.fr/WHR/WHR.pdf>.

Equivalences to the WHR model are:

-   Climbers are players.
-   Ascents are games.
-   A clean ascent is a "win" for the climber.

Notable differences are:

-   Routes are like players except their rating does not change with time.
-   The normal distribution is used for the prior distribution of route and initial climber ratings.
-   A "page" is the model of a climber in a particular time interval (like a page in a climber's logbook).  This is equivalent to a player on a particular day in WHR, except that the time may be quantized with lower resolution (e.g. a week).
-   Ascents can be labelled with a style.  Climbers' relative proficiency in alternative styles is modelled as a Wiener process.  The process is independent of their proficiency in the "base" style.  Hence a "climber-style" is like a climber and a "page-style" is like a page, but for a specific style.

Results of analyzing a database of Australian ascents with an earlier version of this software are discussed in the paper:

> Dean Scarff, "Estimation of Climbing Route Difficulty using Whole-History Rating", [arXiv:2001.05388](https://arxiv.org/abs/2001.05388) [stat.AP], 2020.

## Contents and Usage

### Python library

The estimation algorithms are implemented in Python and Cython, in the `climbing_ratings` package.  Some effort has been taken to optimize parts of the code for speed, namely by leveraging numpy for vectorized operations and using Cython to reduce Python overheads and allow C compilers to generate vectorized CPU instructions.

Python 3.7+ with pip is required; Python 3.9 is recommended.  The additional dependencies can be installed with:

```sh
python3 -m pip install Cython numpy pytest
```

The package can be built for the local system using:

```sh
export CFLAGS="-march=native -mtune=native"
python3 setup.py build
python3 setup.py build_ext --inplace
```

Unit tests can be run using:

```sh
python3 -X dev -W error -m pytest climbing_ratings
```

Type checking can be run with `mypy`:

```sh
python3 -m mypy -p climbing_ratings
```

### Estimation script

The `climbing_ratings` module can be run as an estimation script.  It reads in a set of CSV files and writes out the estimated ratings for pages and routes as CSV files.  To read and write CSV files from the `data/` directory, it can be run like:

```sh
python3 -m climbing_ratings data
```

To understand the format for the data files, see the documentation in `climbing_ratings/__main__.py` and the examples in `tests/testdata`.

For legacy reasons, the `02-run_estimation.py` script provides a stub as an alternate way to call the estimation script.

It will typically run in less than 5 seconds per 100,000 ascents (measured on an Intel Core i5-8210Y).

Integration tests can be run using:

```sh
python3 -X dev -m pytest tests
```

### R package

The `climbr` sub-directory contains an R package with utility functions for data preparation and results analysis.  Those functions are called from the top-level R scripts.

Tests can be run using:

```sh
Rscript --vanilla -e 'devtools::check("climbr")'
```

### R scripts

A collection of R scripts are used for data preparation and results analysis.  They can be sourced into an R session.  Most of the logic is in the `climbr` package, which can be used in-place (without installation) using the "devtools" package.  The scripts also use several libraries from the "tidyverse" collection.  Additionally, to read JSON data from theCrag API responses, the "jsonlite" package is required.  To perform cross-validation, the "caret" package is required.  Rcpp and a C++ toolchain are required to build some optimized data preparation code.  The packages can be installed from R:

```R
install.packages(c("devtools", "tidyverse", "jsonlite", "caret", "Rcpp"))
```

`01-data_prep.R` creates appropriate input CSV files for the `climbing_ratings` Python script.

`03-post_estimation.R` merges the estimation results with the data frames created by `01-data_prep.R`, and produces some plots that can be used to analyze the model fit.

The `cross_validation.R` script performs repeated k-fold cross-validation on the model.

With the file `data/raw_ascents.csv` already present, the entire pipeline can be run from R:

```R
data_dir <- "data"
devtools::load_all("climbr")
library(climbr)
source("01-data_prep.R")
system2("python3", c("-m", "climbing_ratings", data_dir))
source("03-post_estimation.R")
```

The `raw_ascents.csv` file can be regenerated from a directory containing CSV logbook exports from theCrag.  After attaching the `climbr` package, read the logbooks from the directory "logbooks":

```R
write.csv(
  ReadLogbooks("logbooks"),
  "data/raw_ascents.csv",
  quote = FALSE,
  row.names = FALSE
)
```

Instead of reading the ascents data from logbook exports, it can instead be read from the JSON responses returned by theCrag's API.  With files like `data/ascents-01.json` present, replace `source("01-data_prep.R")` in the pipeline above with:

```R
source("01-data_prep_json.R")
```

Note that processing the JSON can be quite slow (on the order of 1 minute for 1 million ascents), so as a convenience, the results of the data preparation script can be read from a file:

```R
data_dir <- "data"
dfs <- readRDS(file.path(data_dir, "dfs.rds"))
```

Tests can be run using:

```sh
TZ=UTC Rscript --vanilla -e 'testthat::test_dir("tests")'
```

## Interpreting ratings

Ratings come in two flavours; the `gamma` rating and the "natural" rating `r`; they are related by `gamma = exp(r)`.

Suppose a climber whose current rating is `gamma_i` attempts to climb a route with rating `gamma_j`.  The model predicts the probability of the ascent being clean as:

```
    P(clean) = gamma_i / (gamma_i + gamma_j)
```
