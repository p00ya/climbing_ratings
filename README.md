# climbing_ratings

`climbing_ratings` estimates ratings for the sport of rock climbing.  The ratings can be used to predict route difficulty and climber performance on a particular route.  It is written by Dean Scarff.

The algorithms are based on the "WHR" paper:

> Rémi Coulom, "Whole-History Rating: A Bayesian Rating System for Players of Time-Varying Strength", <https://www.remi-coulom.fr/WHR/WHR.pdf>.

Equivalences to the WHR model are:

-   Climbers are players.
-   Ascents are games.
-   A clean ascent is a "win" for the climber.

Notable differences are:

-   Routes are like players except their rating does not change with time.
-   The gamma distribution is used for the prior distribution of route and initial climber ratings.
-   A "page" is the model of a climber in a particular time interval (like a page in a climber's logbook).  This is equivalent to a player on a particular day in WHR, except that the time may be quantized with lower resolution (e.g. a week).

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

The Python script `02-run_estimation.py` reads in set of CSV files and writes out the estimated ratings for pages and routes as CSV files.  To read and write CSV files from the `data/` directory, it can be run like:

```
python3 02-run_estimation.py data
```

It will typically take seconds (measured on an Intel Core i5) for every ten thousand ascents.

### Data preparation and results analysis

The `01-data_prep.R` script creates appropriate input CSV files for `02-run_estimation.py`.

The `03-post_estimation.R` script merges the estimation results with the data frames created by `01-data_prep.R`, and produces some plots that can be used to analyze the model fit.

With the file `data/raw_ascents.csv` already present, the entire pipeline can be run from R:

```
data_dir <- "data"
source("01-data_prep.R")
system("python3 02-run_estimation.py data")
source("03-post_estimation.R")
```

## Interpreting ratings

Ratings come in two flavours; the `gamma` rating and the "natural" rating `r`; they are related by `gamma = exp(r)`.

Suppose a climber whose current rating is `gamma_i` attempts to climb a route with rating `gamma_j`.  The model predicts the probability of the ascent being clean as:

```
    P(clean) = gamma_i / (gamma_i + gamma_j)
```
