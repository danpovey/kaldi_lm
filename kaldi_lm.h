#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <tr1/unordered_map>
using std::tr1::unordered_map;
using std::string;

// Parses the line into a history, predicted-token, and count.
// Will crash if line is invalid.
inline void parse_line(const char *line, int *N_out, std::string *history_out,
                       std::string *predicted_out, double *count_out) {
  // Expect input of the form
  // history\tpredicted\tcount
  // where "history" is a possibly empty sequence of words (space-separated,
  // in reverse order), "predicted" is a word, and "count" is a floating-point
  // number.
  const char *first_tab = strchr(line, '\t');
  if (!first_tab) {
    fprintf(stderr, "Invalid line: %s\n", line);
    exit(1);
  }
  const char *history = line, *predicted = first_tab+1;
  const char *second_tab = strchr(predicted, '\t');
  if (!second_tab) {
    fprintf(stderr, "Invalid line: %s\n", line);
    exit(1);
  }
  const char *count_str = second_tab+1;
  
  int ngram_order = 1 + (*history == '\t' ? 0 : 1);
  // N-gram order is 1 plus number of spaces
  // in the history, plus 1 if history is nonempty.
  // (e.g. if history is "b a", N-gram order is 3); if history
  // is "", N-gram order is 1.
  for (const char *c = history; c != first_tab; c++)
    if (*c == ' ') ngram_order++;

  double count = atof(count_str);
  if (count == 0.0 && *count_str != '0' && *count_str != '-') {
    fprintf(stderr, "discount_ngrams: bad line  %s\n", line);
    exit(1);
  }
  *N_out = ngram_order;
  *history_out = std::string(history, first_tab-history);
  *predicted_out = std::string(predicted, second_tab-predicted);
  *count_out = count;
}

// struct Counts stores the N-gram counts for a particular history.
// "history" is a history string e.g. "a b",
// and counts is a map from the predicted thing (e.g. "c", a word,
// or "*" meaning backoff count, or "%" meaning total count), 5o
// the associated count.  Programs that use this typically declare
// a global variable "ngram_counts".

struct Counts {
  string history;
  unordered_map<string, double> counts; // map from "predicted" to count.
};


