# climbing_ratings

`climbing_ratings` estimates ratings for the sport of rock climbing.  The ratings can be used to predict route difficulty and climber performance on a particular route.

The algorithms are based on the "WHR" paper:

Rémi Coulom, "Whole-History Rating: A Bayesian Rating System for Players of
Time-Varying Strength", <https://www.remi-coulom.fr/WHR/WHR.pdf>.

Equivalences to the WHR model are:

-   climbers are players
-   ascents are games
-   a clean ascent is a "win" for the climber

Notable differences are:

-   routes are like players except their rating does not change with time.
-   routes and climbers have initial priors from the gamma-distribution.
-   a "page" (like a page in a climber's logbook) is the model of a climber in a particular time interval.  This is equivalent to a player on a particular day in WHR, except that the time may be quantized with lower resolution (e.g. a week).
