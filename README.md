# Climbing Ratings

Climbing Ratings is software that estimates ratings for the sport of rock climbing.  The ratings can be used to predict route difficulty and climber performance on a particular route.

Climbing Ratings is Copyright 2019, 2020 the Contributors to the Climbing Ratings project.

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

Results of analyzing a database of Australian ascents with this software are discussed in the paper:

> Dean Scarff, "Estimation of Climbing Route Difficulty using Whole-History Rating", [arXiv:2001.05388](https://arxiv.org/abs/2001.05388) [stat.AP], 2020.

## Contents and Usage

### Python library

The estimation algorithms are implemented in Python and Cython, in the `climbing_ratings` package.  Some effort has been taken to optimize parts of the code for speed, namely by leveraging numpy for vectorized operations and using Cython to reduce Python overheads and allow C compilers to generate vectorized CPU instructions.

The package can be built using:

```
python3 setup.py build
```

Unit tests can be run using:

```
python3 setup.py test
```

### Estimation script

The Python script `02-run_estimation.py` reads in a set of CSV files and writes out the estimated ratings for pages and routes as CSV files.  To read and write CSV files from the `data/` directory, it can be run like:

```
python3 02-run_estimation.py data
```

It will typically run in less than 5 seconds per 100,000 ascents (measured on an Intel Core i5-8210Y).

Tests can be run using:

```
python3 -m unittest discover -s tests
```

### R package

The `climbr` sub-directory contains an R package with utility functions for data preparation and results analysis.  Those functions are called from the top-level R scripts.

Tests can be run using:

```
Rscript --vanilla -e 'devtools::check("climbr")'
```

### R scripts

A collection of R scripts are used for data preparation and results analysis.  They can be sourced into an R session.  Most of the logic is in the `climbr` package, which can be used in-place (without installation) using the "devtools" package.  The scripts also use several libraries from the "tidyverse" collection.  Additionally, to read JSON data from theCrag API responses, the "jsonlite" package is required.  To perform cross-validation, the "caret" package is required.  The packages can be installed from R:

```
install.packages(c("devtools", "tidyverse", "jsonlite", "caret"))
```

`01-data_prep.R` creates appropriate input CSV files for `02-run_estimation.py`.

`03-post_estimation.R` merges the estimation results with the data frames created by `01-data_prep.R`, and produces some plots that can be used to analyze the model fit.

The `cross_validation.R` script performs repeated k-fold cross-validation on the model.

With the file `data/raw_ascents.csv` already present, the entire pipeline can be run from R:

```
data_dir <- "data"
devtools::load_all("climbr")
library(climbr)
source("01-data_prep.R")
system2("./02-run_estimation.py", data_dir)
source("03-post_estimation.R")
```

The `raw_ascents.csv` file can be regenerated from a directory containing CSV logbook exports from theCrag.  Having sourced `00-data_prep_functions.R`, and with the logbooks in the directory "logbooks":

```
write.csv(
  ReadLogbooks("logbooks"),
  "data/raw_ascents.csv",
  quote = FALSE,
  row.names = FALSE
)
```

Instead of reading the ascents data from logbook exports, it can instead be read from the JSON responses returned by theCrag's API.  With files like `data/ascents-01.json` present, replace `source("01-data_prep.R")` in the pipeline above with:

```
source("01-data_prep_json.R")
```

Tests can be run using:

```
Rscript --vanilla -e 'testthat::test_dir("tests")'
```

## Interpreting ratings

Ratings come in two flavours; the `gamma` rating and the "natural" rating `r`; they are related by `gamma = exp(r)`.

Suppose a climber whose current rating is `gamma_i` attempts to climb a route with rating `gamma_j`.  The model predicts the probability of the ascent being clean as:

```
    P(clean) = gamma_i / (gamma_i + gamma_j)
```
