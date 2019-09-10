// see comments at top of main() which will make clear what this
// program is for.

#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <assert.h>
#include <stdlib.h>
#include <string.h>


struct NgramEntry {
  int order;
  std::string history;
  std::string predicted;
  double count;
};

// This stack is generally used with just one element,
// to store the latest N-gram count.  On rare occasions it
// will have more entries, this is when we have a sequence of
// potentially "useless" N-gram entries with "*", and we
// want to see if they end with something that makes them
// non-useless.  E.g. if we see
// a   *   2.0
// a b   *   2.0
// a b c   *   2.0
// a b c    d   3.0
// we don't know until we see the "d" entry that the previous ones weren't
// useless.
// There are three possible cases:
//  the stack is empty.
//  the stack contains a single "real" N-gram count, e.g. "a b   c   3.0"
//  the stack contains one or more "*" entries, with histories that
//  are strict prefixes of each other, like:
//   a   *   2.0
//   a b   *   2.0

std::vector<NgramEntry> stack;



inline void print_entry(const NgramEntry &entry) {
  // For now we just print out zero counts, but with a warning.
  if (entry.count == 0.0) {
    if (entry.predicted == "*") {
      fprintf(stderr, "Printing zero NgramEntry %s %s\n", entry.history.c_str(),
              entry.predicted.c_str());
    } else {
      fprintf(stderr, "Not printing zero NgramEntry %s %s\n", entry.history.c_str(),
              entry.predicted.c_str());
      return;
    }
  }
  printf("%s\t%s\t%.2f\n", entry.history.c_str(), entry.predicted.c_str(), entry.count);  
}

// This is called at the end of the program.  It prints out the last pending
// N-gram-- but not if it predicts "*", which would then be a "useless N-gram".
// The sorting guarantees that any real words in the same history-state, or
// any more-specific history-states, would appear after this-- and since they
// don't, it must be useless.
void flush_output() {
  if (stack.size() == 1 && stack[0].predicted != "*") {
    // We can only have a non-"*" N-gram count if size() == 1.
    print_entry(stack[0]);
    stack.clear();
  }
}

inline bool history_is_prefix_of(const std::string &hist_a,
                                 const std::string &hist_b) {
  // Returns true if hist_a is a prefix (not necessarily a strict prefix)
  // of string hist_b.
  // e.g. "a" is a prefix of "a"
  // "ab cd" is a prefix of "ab cd e";
  // "ab c" is not a prefix of "ab cd".
  size_t lena = hist_a.length(), lenb = hist_b.length();
  if (lenb < lena) return false;
  return ((strncmp(hist_a.c_str(), hist_b.c_str(), lena) == 0)
          && (lena == lenb || lena == 0 || hist_b.c_str()[lena] == ' '));
}


// Will crash if line is invalid.
inline void parse_line(const char *line,
                       NgramEntry *entry) {
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
  if (count == 0.0) {
    if (*count_str != '0' && *count_str != '-') {
      fprintf(stderr, "discount_ngrams: bad line  %s\n", line);
      exit(1);
    } else {
      fprintf(stderr, "discount_ngrams: warning, zero count  %s\n", line);
    }
  }
  entry->order = ngram_order;
  entry->history = std::string(history, first_tab-history);
  entry->predicted = std::string(predicted, second_tab-predicted);
  entry->count = count;
}

inline void process_line(char *line) {
  NgramEntry entry;
  parse_line(line, &entry);
  if (entry.predicted[0] == '%') {
    fprintf(stderr, "merge_ngrams_online: not expecting interpolated counts.\n");
    exit(1);
  }
  // Make sure the input is in sorted order.
#ifndef NDEBUG
  if (!stack.empty()) {
    int comp = entry.history.compare(stack.back().history);
    assert(comp > 0 || (comp == 0 && entry.predicted.compare(stack.back().predicted) >= 0));
  }
#endif
  if (stack.empty()) {
    stack.push_back(entry);
  } else {
    NgramEntry &back = stack.back();
    if (stack.back().predicted[0] == '*') {
      while (!stack.empty() && !history_is_prefix_of(stack.back().history, entry.history))
        stack.pop_back(); // Get rid of useless entries on top of stack.
      
      if (!stack.empty()) { // we know that entry on stack is prefix of current history.
        if (entry.predicted[0] == '*') {
          if (stack.back().history == entry.history) {
            stack.back().count += entry.count;
          } else { // stack history is strict prefix of cur history,
            stack.push_back(entry);
          }
        } else { // all items on stack are needed, and different from
          // "entry" (which is not "*"): print them out.
          for (size_t i = 0; i < stack.size(); i++) print_entry(stack[i]);
          stack.clear();
          stack.push_back(entry);
        }
      } else {
        stack.push_back(entry);
      }
    } else { // Stack contains a single, non-* entry.
      assert(stack.size()==1);
      if (entry.history == back.history &&
          entry.predicted == back.predicted) {
        back.count += entry.count;
      } else {
        print_entry(back);
        back = entry;
      }
    }
  }
}
  

int main(int argc, char **argv) {
  // This program takes ngram counts that have been sorted
  // (with "sort", and assuming LC_ALL=C),
  // and it merges them such that if we get ngrams on successive
  // lines that have the same value, e.g.
  // a b   c   2.0
  // a b   c   1.0
  // it will add them together, printing out
  // a b   c   3.0
  // It also removes useless n-grams, e.g.
  // a b   *   1.0
  // (such N-grams are useless when there are no "real"
  // N-grams with the same prefix, e.g. "a b   c   1.0", and no
  // higher-order N-grams with a more specific prefix, e.g. "a b c  d  1.0").
  
  if (argc != 1) {
    fprintf(stderr, "Usage: cat ngrams | sort | merge_ngrams > merged_ngrams\n"
            "See comments in code for more information\n");
    exit(1);
  }
  size_t nbytes = 100;
  char *line = (char*) malloc(nbytes + 1);
  int bytes_read;
  while ((bytes_read = getline(&line, &nbytes, stdin)) != -1) {
    process_line(line);
  }  
  flush_output();
  delete line;
}
