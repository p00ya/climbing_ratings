// Optimized utilities for reshaping jsonlite lists.
//
// Copyright Contributors to the Climbing Ratings project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Rcpp.h>

using Rcpp::CharacterVector;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::String;

namespace {

// Coerces `x[[idx]]` to an integer.
//
// @param x either an atomic integer vector, or a list of singleton
//     integer vectors.
// @param idx the element of x to extract the string from (1-based)
// @keywords internal
int LiftInteger(SEXP x, int idx) {
  switch (TYPEOF(x)) {
  case VECSXP: {
    List y = x;
    IntegerVector z = y.at(idx - 1);
    return z.at(0);
  }
  case INTSXP: {
    IntegerVector y = x;
    return y.at(idx - 1);
  }
  }
  return NA_INTEGER;
}

// Coerces `x[[idx]]` to a string.
//
// @param x either an atomic character vector, or a list of singleton
//     character vectors.
// @param idx the element of x to extract the string from (1-based)
// @keywords internal
String LiftString(SEXP x, int idx) {
  switch (TYPEOF(x)) {
  case VECSXP: {
    List y = x;
    CharacterVector z = y.at(idx - 1);
    return z.at(0);
  }
  case STRSXP: {
    CharacterVector y = x;
    return y.at(idx - 1);
  }
  }
  return NA_STRING;
}

} // namespace

//' Converts the list-of-singleton-integer-vector structures typical of
//' jsonlite to an atomic vector of integers.  NULLs are converted to NA.
//'
//' @param lst a list of lists of characters
//' @param idx the singleton element of the inner list (1-based) to extract
//' @keywords internal
// [[Rcpp::export]]
IntegerVector FlattenInt(List lst, int idx = 1) {
  if (idx < 1) {
    Rcpp::stop("invalid idx value %d", idx);
  }
  const int n = lst.length();
  IntegerVector flattened(n, NA_INTEGER);
  int i = 0;
  for (List::iterator it = lst.begin(); it != lst.end(); ++it, ++i) {
    try {
      flattened[i] = LiftInteger(*it, idx);
    } catch (const Rcpp::index_out_of_bounds &e) {
      Rcpp::warning("lst[%d][%d] subscript out of bounds", i + 1, idx);
    }
  }
  return flattened;
}

//' Converts the list-of-singleton-character-vector structures typical of
//' jsonlite to an atomic vector of characters.  NULLs are converted to NA.
//'
//' @param lst a list of lists of characters
//' @param idx the singleton element of the inner list (1-based) to extract
//' @keywords internal
// [[Rcpp::export]]
CharacterVector FlattenChr(List lst, int idx = 1) {
  if (idx < 1) {
    Rcpp::stop("invalid idx value %d", idx);
  }
  const int n = lst.length();
  CharacterVector flattened(n, NA_STRING);
  int i = 0;
  for (List::iterator it = lst.begin(); it != lst.end(); ++it, ++i) {
    try {
      flattened[i] = LiftString(*it, idx);
    } catch (const Rcpp::index_out_of_bounds &e) {
      Rcpp::warning("lst[%d][%d] subscript out of bounds", i + 1, idx);
    }
  }
  return flattened;
}
