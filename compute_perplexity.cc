#include "kaldi_lm.h"
#include <math.h> // for log.

// This function is different from parse_line in kaldi_lm.h in that it
// accepts lines with the count field prefixed with ":", which is what
// we do for lines in the "test set".  (this ensures the ordering we
// want, so we will have already read the N-gram counts we need to
// compute the likelihood).
//
// This function will set "is_test_set_line" to true if this line is a "test set
// line" (with ":" preceding the count), and false otherwise.

inline void parse_line_special(const char *line, int *N_out,
                               std::string *history_out,
                               std::string *predicted_out,
                               double *count_out,
                               bool *is_test_set_line) {
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

  *N_out = ngram_order;
  *history_out = std::string(history, first_tab-history);
  *predicted_out = std::string(predicted, second_tab-predicted);

  if (*count_str != ':') {
    *is_test_set_line = false; // means: doesn't come from the test set.
  } else {
    *is_test_set_line = true;
    count_str++; // Skip over the ":".
  }
  double count = atof(count_str);
  if (count == 0.0 && *count_str != '0' && *count_str != '-') {
    fprintf(stderr, "discount_ngrams: bad line  %s\n", line);
    exit(1);
  }
  *count_out = count;
}    



// This variable "ngram_counts" is the key data-structure for this
// program.  It is indexed by the history-length (e.g. 0 for 1-grams, etc.).
// This program only ever keeps in memory the counts for a single history
// of a particular order.  The order it accesses the counts in ensures that
// it only ever needs the counts that it curently has in memory.
// Note: while these counts are in memory it means they are in some sense
// "pending output", or waiting to be output.
// We always maintain "ngram_counts" in a state such that the "history" of
// order h is always a shorter version of the "history" for order h+1.
// The only exception to this is if histories of orders h+1 and higher
// are all the empty string (and the counts empty).
std::vector<Counts> ngram_counts;

// The next two are configuration variables.
int num_unk_words = 1; // -log(this number) serves as a penalty on the #unk words.
bool verbose = false;

//  Below are the stats this program collects.
double tot_test_count = 0.0; // Total count of test-set n-grams, excluding unk ("C")
double tot_test_loglike = 0.0; // log_e total likelihood, excluding unk ("C").
double tot_test_unk_count = 0.0; // total count of unk ("C") in test set.
double tot_test_unk_loglike = 0.0; // log_e total likelihood of unk ("C") in test set


// Note: throughout, the history-length "h" is always one less than
// the associated n-gram order (n).

// This function increments hash[predicted]. by "count".  The same
// as (*hash)[predicted] += count, except being careful about what
// happens when "predicted" not currently a key in the hash (not
/// sure if this is undefined behaviour)
inline void increment_hash(const std::string &predicted,
                           double count, 
                           unordered_map<string,double> *hash) {
  unordered_map<string,double>::iterator iter = hash->find(predicted);
  if (iter == hash->end()) (*hash)[predicted] = count;
  else iter->second += count;
}

// Clears the counts for history-length 0 <= h < ngram_counts.size(),
// and any higher orders, and sets the histories to "".
void clear_counts(int h);

// This function
// ensures that for history-length h >= 0, and any lower values of h,
// the history we have matches "history".  This may involve clearing
// out any histories of lower or higher orders that does not match,
// and setting the "history" string for lower orders to sub-strings
// of the current history string (even if they are empty).
// This function will also resize ngram_counts if necessary.
void ensure_history_matches(int h, const std::string &history); 

// This function will add the given count to the N-gram counts in memory for
// history-length "h", and predicted-word (or "*" or "%") "predicted".  You must
// have first called ensure_history_matches, to ensure that the "history"
// variable for this history-length matches whatever you read in.
inline void add_count(int h,
                      const std::string predicted,
                      double count) {
  increment_hash(predicted, count, &ngram_counts[h].counts);
}

void process_test_set_ngram(int h_in,
                            const std::string predicted,
                            double count) {
  // This function doesn't need the history string because the
  // preceding call to "ensure_history_matches" guarantees that
  // the history of length h will be the same as the one on
  // the n-gram line.

  // It may be that this context does not exist in the n-gram
  // counts estimated from the training set.
  // The next few lines skip over these.
  int h = h_in;
  while (h > 0 && ngram_counts[h].counts.empty())
    h--;
  double prob = 1.0;
  const double inv_ln10 = 0.434294481903252;
  while(1) { // We'll break from this when we find the prob (or we'll crash).
    // "prob" will be accumulating the backoff probability, in case we have to back
    // off.
    double total_count;
    unordered_map<string,double>::const_iterator iter;
    if ( (iter=ngram_counts[h].counts.find("%")) == ngram_counts[h].counts.end()
         || (total_count = iter->second) <= 0.0) {
      fprintf(stderr, "compute_perplexity: for history-state \"%s\", no total-count %% is seen\n"
              "(perhaps you didn't put the training n-grams through interpolate_ngrams?)\n",
              ngram_counts[h].history.c_str());
      exit(1);
    }
    if ((iter=ngram_counts[h].counts.find(predicted)) !=
        ngram_counts[h].counts.end()) { // We have an entry for this word..
      prob *= iter->second / total_count;
      break;
    } else {
      if (h == 0) { // Nowhere to back off to... this shouldn't really occur
        // so for now we'll crash.  Later we can make this a non-crash if needed.
        fprintf(stderr, "compute_perplexity: no unigram-state weight for predicted word \"%s\"\n",
                predicted.c_str());
        exit(1);
      } else {
        if ((iter=ngram_counts[h].counts.find("*")) ==
            ngram_counts[h].counts.end() || iter->second <= 0.0) {
          fprintf(stderr, "compute_perplexity: no backoff weight \"*\" in history-state \"%s\"\n",
                  ngram_counts[h].history.c_str());
          exit(1);
        }
        prob *= iter->second / total_count;
        h--; // And continue on whe while() loop.. this is the only path 
        // in this code that does that.
      }
    }
  }
  // Now accumulate the stats (here is also where we correct for
  // num_unk_words.
  double log_prob; 
  if (predicted == "C") {
    prob /= num_unk_words;
    log_prob = log(prob);
    tot_test_unk_count += count;
    tot_test_unk_loglike += count * log_prob;
  } else {
    log_prob = log(prob);
    tot_test_count += count;
    tot_test_loglike += count * log_prob;
  }
  double log10_prob = inv_ln10 * log_prob;
  if (verbose) { // Give some detail comparable to what srilm's "ngram"
    // tool gives with -debug 2.  (although the words will all be our
    // special, shortened versions).
    if (h_in == 0) { // We only had unigram counts from test... no context avail.
      fprintf(stderr, "\tp( %s )\t= [1-gram] %f [ %f ] (count=%f)\n",
              predicted.c_str(), prob, log10_prob, (float)count);
    } else { // note: this prints out the context reversed.
      fprintf(stderr, "\tp( %s | %s )\t= [%d-gram] %.6g [ %.6g ] (count=%f)\n",
              predicted.c_str(), ngram_counts[h_in].history.c_str(),
              h+1, prob, log10_prob, count);
    }
  }
}

void print_stats() {
  double tot_count = tot_test_count + tot_test_unk_count,
      tot_loglike = tot_test_loglike + tot_test_unk_loglike;
  fprintf(stderr,"Perplexity over %f words is %f\n", tot_count, exp(-tot_loglike/tot_count));
  fprintf(stderr, "Perplexity over %f words (excluding %f OOVs) is %f\n",
          tot_test_count, tot_test_unk_count,
          exp(-tot_test_loglike/tot_test_count));
  printf("%f\n", exp(-tot_loglike/tot_count)); // Print just the perplexity (including OOVs)
  // on the standard output).
}


// see comments at top of main() which will make clear what this is
// doing.
inline void process_line(char *line) {
  int n;
  std::string history;
  std::string predicted;
  double count;
  bool is_test_set_line;
  
  parse_line_special(line, &n, &history, &predicted, &count, &is_test_set_line);
  
  int h = n-1; // number of words in the history.
  ensure_history_matches(h, history);
  if (!is_test_set_line) {
    add_count(h, predicted, count);
  } else {
    process_test_set_ngram(h, predicted, count); // Debug output, if any,
    // is produced from here, and from here the stats are incremented.
  } 
}

int main(int argc, char **argv) {
  // This program computes perplexity of test data given
  // discounted N-grams representing an LM, and N-grams obtained
  // from test data whose perplexity we want to evaluate.
  // The training N-grams would be output from interpolate_ngrams

  for (int x = 1; x <= 2; x++) { // warning: this changes argc and argv.
    if (argc > 1 && !strcmp(argv[1], "-v")) {
      argc--;
      argv++;
      verbose = true;
    }
    if (argc > 2 && !strcmp(argv[1], "-u")) {
      char *endptr;
      num_unk_words = strtol(argv[2], &endptr, 10);
      if (num_unk_words < 1 || *endptr != '\0') {
        fprintf(stderr, "compute_perplexity: invalid argument to -u option: %s\n",
                argv[2]);
        exit(1);
      }
      argc -= 2;
      argv += 2;
    }
  }
  if (argc != 1) { // If we have any remaining command-line arguments that were not
    // parsed....
    fprintf(stderr, "Usage:\n"
            " cat lm_ngrams test_ngrams | sort | compute_perplexity [options]\n"
           "Note: the n-grams in test_ngrams should have a \":\" before the count, to\n"
           "distinguish them, e.g.:\n"
           "b a\tc\t:1.0\n"
           "Options:\n"
           "   -v     Print extra information, including likelihood and base-10\n"
           "          log-likelihood for each N-gram\n"
           "   -u n   Number of unknown-words we assume there to be (default 1);\n"
           "          -log(n) gets added to unknown-word log-likelihoods\n");
    exit(1);
  }

  size_t nbytes = 100;
  char *line = (char*) malloc(nbytes + 1);
  int bytes_read;
  while ((bytes_read = getline(&line, &nbytes, stdin)) != -1) {
    process_line(line);
  }
  print_stats();
  delete line;
}


// The recursive function "get_prob" returns the probability
// of word "predicted" given this history-state (which will
// be less than the full history length).  We just give the
// #words h in the history state, because the data-structures
// and ordering ensure that we only ever call this when
// the history we have in "ngram_counts" matches what we need.
// This function assumes that "interpolate_if_needed" has already
// been called for this order, so it can look for the total-count
// stored as "%".
// This function looks to see if this word was actually seen
// in this history state, and if so it works out the probability
// directly from the total-count "%" and the word's count; otherwise
// it goes to backoff, calling itself for h-1.

double get_prob(int h, const std::string &predicted) {
  assert(predicted[0] >= 'A'); // Check it's a "real word", not "%" or "*", or
  // empty.
  assert(h >= 0 && h < ngram_counts.size()-1); // -1 because we just
  // don't expect this to be called for the highest-order history length..
  // it should work in that case anyway, though.
  unordered_map<string, double> &counts = ngram_counts[h].counts;
  double total_count;
  { // Find total_count.
    unordered_map<string, double>::iterator iter = counts.find("%");
    assert(iter != counts.end()); // would be coding error if this fails.
    total_count = iter->second;
    assert(total_count != 0); // or something is wrong.
  }
  // Find count of current word.
  unordered_map<string, double>::iterator iter = counts.find(predicted);
  if (iter != counts.end()) {
    return iter->second / total_count; // Return count for this word / total_count.
    // Note: since we already did interpolatoin, this would contain
    // any probability mass taken by interpolating with lower orders.
  } else { // No count for this word -> go to backoff.
    if (h == 0) { // can't go to backoff.. something went wrong.
      fprintf(stderr, "No count for word %s in unigram history state.\n",
              predicted.c_str());
      exit(1);
    }
    iter = counts.find("*"); // find the backoff weight.
    if (iter == counts.end()) { // Backoff weight not specified...
      // maybe user supplied input that was not discounted.
      fprintf(stderr, "No count for backoff symbol \"*\" in history "
              "state \"%s\": maybe you gave non-discounted N-grams to "
              "interpolate_ngrams?\n",  ngram_counts[h].history.c_str());
      exit(1);
    }
    double backoff_weight = iter->second / total_count;
    assert(backoff_weight != 0.0);
    return backoff_weight * get_prob(h-1, predicted);
  }
}


// See comments by declaration of this function.  Clears
// counts and histories of this history-length and higher.
void clear_counts(int h_in) {
  assert(h_in >= 0 && h_in < ngram_counts.size());
  for (int h = ngram_counts.size() - 1; h >= h_in; h--) {
    // Flush higher orders first, so we always maintain the property of the
    // counts that matching lower-order histories are always present if
    // higher-order ones are.
    if (h > 0 && ngram_counts[h].history == "")  continue; // Empty; nothing to do.
    ngram_counts[h].history = ""; // Set the history to empty (invalidating this n-gram order)
    unordered_map<string,double> tmp;
    tmp.swap(ngram_counts[h].counts); // Clearing the counts this way stops the
    // hashes getting too large, which would cause us to spend too much time
    // in clear().
  }
}

// See comments by this function's declaration for its purpose.
void ensure_history_matches(int h, const std::string &history) {
  assert(h >= 0 && (h == 0 || !history.empty()));
  if (h >= ngram_counts.size()) ngram_counts.resize(h+1);
  if (ngram_counts[h].history == history) return; // Nothing to do.
  else { // note: if h==0, then both should always be "", so h must be > 0.
    if (ngram_counts[h].history.compare(history) > 0) { // a check.
      // i.e. previous history comes later in C order than this history,
      // which means they were not in sorted order (note: "" is first,
      // so we don't have to treat it as a special case.
      fprintf(stderr, "error: histories are not in sorted order, \"%s\" > \"%s\"\n",
              ngram_counts[h].history.c_str(), history.c_str());
      exit(1);
    }
    std::string sub_history;
    if (h > 1) {
      size_t pos = history.find_last_of(' ');
      assert(pos != std::string::npos); // would be invalid history of order >= 2.
      sub_history = std::string(history, 0, pos);
    } // else sub_history is empty
    ensure_history_matches(h-1, sub_history); // This guarantees that the history
    // of lower order matches, so we don't have to worry about it.
    clear_counts(h); // Clear out the counts of this order and higher, and
    // set histories to empty.
    // At this point, ngram_counts[h].history == "" (and also for higher
    // values of h).
    ngram_counts[h].history = history;
  }
}

