suppressMessages(source("../00-data_prep_functions.R"))

src_dir <- "testdata"
data_dir <- tempfile("dir")

# Create temporary directory and copy testdata data there.
dir.create(data_dir)
file.copy(
  file.path(src_dir, "raw_ascents.csv"),
  file.path(data_dir)
)
source("../01-data_prep.R")
