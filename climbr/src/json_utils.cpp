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

#include <utility>
#include <vector>

#include <Rcpp.h>

using Rcpp::CharacterVector;
using Rcpp::DataFrame;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::Named;
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

namespace {

// Adds a pitch suffix to the base name.
//
// For example, PitchSuffix("888", "2") becomes "888P2".
// @internal
std::string PitchSuffix(String::StringProxy base, const std::string &pitch) {
  std::string s = Rcpp::as<std::string>(base);
  s.reserve(s.size() + 1 + pitch.size());
  s += "P";
  s += pitch;
  return s;
}

// Encapsulates the input and output state for expanding pitches to ascents.
class ExpandPitchesHelper {
public:
  // Initializes the instance to process ascents from the given data frame.
  explicit ExpandPitchesHelper(DataFrame df) : n(df.nrows()) {
    ascent_in = df["ascentId"];
    route_in = df["route"];
    tick_in = df["tick"];
    climber_in = df["climber"];
    timestamp_in = df["timestamp"];
    grade_in = df["grade"];
    style_in = df["style"];
    pitch_in = df["pitch"];

    // Reserve 1.05x more space than the input length.
    const int reservation = n * 105 / 100;
    ascent_out.reserve(reservation);
    route_out.reserve(reservation);
    tick_out.reserve(reservation);
    climber_out.reserve(reservation);
    timestamp_out.reserve(reservation);
    grade_out.reserve(reservation);
    style_out.reserve(reservation);
  }

  // Accumulates one output ascent for each no-pitch ascent and each pitch of
  // an ascent with pitches.
  void Expand() {
    // i loops through each ascent.
    for (int i = 0; i < n; ++i) {
      if (TYPEOF(pitch_in[i]) == VECSXP) {
        // Has pitch information.
        List pitches = pitch_in[i];

        // j loops through each pitch in the current ascent.
        for (int j = 0; j < pitches.size(); ++j) {
          List pitch = pitches[j];
          std::string pitch_label;
          String pitch_tick = NA_STRING;
          try {
            if (pitch.size() < 2 || TYPEOF(pitch[0]) != STRSXP ||
                TYPEOF(pitch[1]) != VECSXP) {
              continue;
            }
            pitch_label = LiftString(pitch, 1);
            pitch_tick = LiftString(pitch[1], 1);
          } catch (const Rcpp::index_out_of_bounds &e) {
            // The pitch JSON is expected to be like:
            //   `["1",["onsight"]]`
            // or
            //   `["1",null]`
            Rcpp::stop("bad pitch shape for ascent %s, pitch %d (length %d)",
                       Rcpp::as<std::string>(ascent_in[i]), j + 1,
                       pitch.length());
            break;
          }

          // Add suffix to ascent_id and route.
          std::string pitch_ascent = PitchSuffix(ascent_in[i], pitch_label);
          std::string pitch_route = PitchSuffix(route_in[i], pitch_label);

          Accumulate(std::move(pitch_ascent), std::move(pitch_route),
                     pitch_tick, climber_in[i], timestamp_in[i], grade_in[i],
                     style_in[i]);
        }
      } else {
        // No pitches; copy 1:1 to output.
        CopyFromInput(i);
      }
    }
  }

  // Returns a DataFrame with the accumulated ascents.
  DataFrame Wrap() const {
    return DataFrame::create(
        Named("ascentId") =
            CharacterVector(ascent_out.begin(), ascent_out.end()),
        Named("route") = CharacterVector(route_out.begin(), route_out.end()),
        Named("tick") = CharacterVector(tick_out.begin(), tick_out.end()),
        Named("climber") =
            CharacterVector(climber_out.begin(), climber_out.end()),
        Named("timestamp") =
            IntegerVector(timestamp_out.begin(), timestamp_out.end()),
        Named("grade") = IntegerVector(grade_out.begin(), grade_out.end()),
        Named("style") = CharacterVector(style_out.begin(), style_out.end()));
  }

private:
  // Accumulates the input ascent with the given (0-based) index.
  void CopyFromInput(int i) {
    ascent_out.push_back(ascent_in[i]);
    route_out.push_back(route_in[i]);
    tick_out.push_back(tick_in[i]);
    climber_out.push_back(climber_in[i]);
    timestamp_out.push_back(timestamp_in[i]);
    grade_out.push_back(grade_in[i]);
    style_out.push_back(style_in[i]);
  }

  void Accumulate(std::string &&ascent, std::string &&route, String tick,
                  String climber, int timestamp, int grade, String style) {
    ascent_out.push_back(ascent);
    route_out.push_back(route);
    tick_out.push_back(tick);
    climber_out.push_back(climber);
    timestamp_out.push_back(timestamp);
    grade_out.push_back(grade);
    style_out.push_back(style);
  }

  const int n;

  CharacterVector ascent_in;
  CharacterVector route_in;
  CharacterVector tick_in;
  CharacterVector climber_in;
  IntegerVector timestamp_in;
  IntegerVector grade_in;
  CharacterVector style_in;
  List pitch_in;

  std::vector<String> ascent_out;
  std::vector<String> route_out;
  std::vector<String> tick_out;
  std::vector<String> climber_out;
  std::vector<int> timestamp_out;
  std::vector<int> grade_out;
  std::vector<String> style_out;
};

} // namespace

//' Expands ascents that have pitch information into separate ascents.
//'
//' Ascents without pitch information will be copied 1:1 to the output.
//' Ascents with pitch information will have one ascent row in the output for
//' each pitch that includes both a label and a tick type.  The route and
//' ascent ID will have the pitch label appended to them, e.g. a "P1"
//' suffix for pitch 1.
//'
//' @param df a data.frame with columns ascentId, route, tick, climber,
//'     timestamp, grade, style, and pitch.
//' @return a data.frame with columns ascentId, route, tick, climber,
//'     timestamp, grade, and style.
// [[Rcpp::export]]
DataFrame ExpandPitches(DataFrame df) {
  ExpandPitchesHelper helper(df);

  helper.Expand();
  return helper.Wrap();
}
