#include "kaldi_lm.h"
#include "math.h"

// This program does entropy pruning of language models, while they are
// represented as N-gram counts.  See extended comment at the bottom for
// mathematical details.  It is assumed that you will call it with
// the output of discount_ngrams, and then put it into interpolate_ngrams
// before converting it into an arpa.
// You give it a pruning threshold that is comparable to a Stolcke-pruning
// type threshold (e.g. 10^-9) multiplied by the total number of predicted-words
// in your training set.
// You can compute the total number of predicted-words in the training set
// by, in the output of e.g. uniq_to_ngrams, merge_ngram_counts or
// discount_ngram_counts, adding up all the counts except (those where
// the predicted-word is '*' that have a non-empty history).


// The function compute_divergence computes (an upper bound on) the
// K-L divergence [times total training-data count] caused by taking
// the count of a in history h1 (c_a_h1) and turning it into backoff
// mass in that history state, and increasing the count of a (c_a_hbo) in the
// backoff history state "h_{bo}", by the same quantity.  It's more
// exact than Stolcke pruning because we also modify the backoff
// distribution.  Unlike Stolcke pruning, it's specialized for LMs
// "with interpolation", e.g. interpolated Kneser-Ney.
double compute_divergence(double c_a_h1, double c_all_h1, double c_bo_h1,
                            double c_a_hbo, double c_all_hbo);


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
double threshold = 1.0; // The change in (total train-data count * KL-divergence),
// below which we will prune a parameter.  Divide by the total train-data count
// to get a threshold comparable to that in Stolcke pruning.
bool prune_whole_states = true;

// Some stats accumulated while pruning.
double total_divergence = 0.0;
int num_params_pruned = 0;


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

// reads the wordlist into "wordlist".  This is used for
// assigning weights from backoff to all words, in the unigram case.
void read_wordlist(const char *wordlist_filename);

// This is called from flush_counts; we prune this history-state
// (i.e. the history state of history-length h currently in ngram_counts),
// before writing out the counts.  Requires 0 < h < ngram_counts.size().
void try_removing_ngrams(int h);

// this is called from flush_counts; it writes out any counts that
// are currently stored, with history-length h >= 0.  It requires that
// the history be valid (e.g. not empty, for orders > 0).  It does
// not write out the total-counts ("%"), as these were just needed
// to make the pruning more efficient.
void write_counts(int h);


// Called from try_removing_ngrams(), this ensure that the total-count ("%")
// exists for the currently loaded history-state of this length.
// If this history-state is empty it just returns zero; else it computes
// the total-count, stores it as "%", and returns it.
double get_total_count(int h);

// Called from try_removing_ngrams(), this returns the count for string "predicted" in
// the currently loaded history-state with history length h, or zero if there is
// no such count.
inline double get_count(int h, const std::string &predicted);

// Flush the counts for history-length h (with 0 <= h < ngram_counts.size()),
// and any higher orders.  This will prune the counts (for h>0),
// write them out, and set to empty the history and the "counts" of that order
// (and higher).  Calls try_removing_ngrams and write_counts.
void flush_counts(int h);

// This function
// ensures that for history-length h >= 0, and any lower values of h,
// the history we have matches "history".  This is typically because
// we want to increment the counts for that history and we're concerned that
// a different history might be there for that hisory position.  If the history
// does not match, this program will call "flush_counts" to get rid of counts
// for any order of history that doesn't match the supplied one,
// and will then set this order's history to "history", and any lower orders to match.
// Note: if it matches for history-length h, it must match for lower
// history-lengths, because we always maintain it in that way.
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


// see comments at top of main() which will make clear what this is
// doing.
inline void process_line(char *line) {
  int n;
  std::string history;
  std::string predicted;
  double count;
  
  parse_line(line, &n, &history, &predicted, &count);

  int h = n-1; // number of words in the history.
  ensure_history_matches(h, history);
  if (*(predicted.c_str()) == '%') {
    fprintf(stderr, "You cannot call prune_ngrams with ngrams that have already "
            "been interpolated.\n");
    exit(1);
  }
  if (count != 0.0)
    add_count(h, predicted, count);
  // Note: most of the output of this program gets produced in
  // "ensure_history_matches", which produces output (via
  // flush_counts -> write_counts) when a history does not match
  // and needs to be got rid of.  We also produce output
  // at the very end when we flush the last histories (including
  // unigram).
}

void test_divergence();
void test_divergence_remove_all();
void print_stats();

int main(int argc, char **argv) {
  test_divergence_remove_all();
  test_divergence();

  // This program applies entropy-based pruning (similar to
  // Stolcke pruning, but more exact) to N-gram counts that are
  // to be used to build an LM of the "interpolated Kneser-Ney"
  // type (well, of that general class).
  // The threshold is not the same as the Stolcke style of
  // threshold (e.g. 10^-9) as it's multiplied by the total
  // number of counts.  E.g. something in the range 1-10 might
  // be typical.
  
  char *endptr;
  if (argc == 2) 
    threshold = strtod(argv[1], &endptr);

  if (argc != 2 || endptr == argv[1] || threshold < 0.0) {
    fprintf(stderr, "Usage: prune_ngrams thresh <ngrams >pruned_ngrams\n"
            "This program removes N-gram counts (actually, completely discounts the\n"
            "to the next history state), as long as doing so would cause less than\n"
            "\"thresh\" K-L divergence [times total training-data count] versus the\n"
            "original model.\n"
            "Similar to Stolcke pruning but more exact.\n"
            "This would typically be applied to the output of discount_ngrams|sort.\n"
            "You would then pipe the output into sort|merge_ngrams, to get rid of\n"
            "\"useless\" backoff probabilities.\n");
    exit(1);
  }
  if (threshold < 1.0e-05) {
    fprintf(stderr, "Warning: the threshold is very small: make sure you are not\n"
            "trying to use a Stolcke-style threshold (our threshold are\n"
            "multiplied by the total data-count\n");
  }

  size_t nbytes = 100;
  char *line = (char*) malloc(nbytes + 1);
  int bytes_read;
  while ((bytes_read = getline(&line, &nbytes, stdin)) != -1) {
    process_line(line);
  }
  if (ngram_counts.size() != 0)
    flush_counts(0); // Flushes the last counts that were read, and the unigram
  // history state.
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
  assert((unsigned char)predicted[0] >= (unsigned char)'A'); // Check
  // it's a "real word", not "%" or "*", or empty.
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

void write_counts(int h) {
  assert(h < ngram_counts.size() && h >= 0);
  const char *history_str = ngram_counts[h].history.c_str();
  if (h > 0) { assert(*history_str != '\0'); } else { assert(*history_str == '\0'); }
  unordered_map<string, double> &counts = ngram_counts[h].counts;  
  for (unordered_map<string, double>::iterator iter = counts.begin();
       iter != counts.end();
       ++iter) {
    const char *predicted_str = iter->first.c_str();
    if (*predicted_str == '%') continue; // Don't write "%".
    double count = iter->second;
    assert(count >= 0); // there's no valid way we could have
    // negative counts.
    if(count >= 0.005) { // else it would be 0 when written as %.2f.
      printf("%s\t%s\t%.2f\n", history_str, predicted_str, count);
    }
  }
}

// See comments by declaration of this function.  Discounts and writes out
// counts of this history-length and higher.
void flush_counts(int h_in) {
  assert(h_in >= 0 && h_in < ngram_counts.size());
  for (int h = ngram_counts.size() - 1; h >= h_in; h--) {
    // Flush higher orders first, so we always maintain the property of the
    // counts that matching lower-order histories are always present if
    // higher-order ones are.
    if (h > 0 && ngram_counts[h].history == "")  continue; // Empty; nothing to do.
    if (h > 0)
      try_removing_ngrams(h);
    write_counts(h); // Write the counts out.
    ngram_counts[h].history = ""; // Set the history to empty (invalidating this n-gram order)
    unordered_map<string, double> temp;
    temp.swap(ngram_counts[h].counts); // This method of clearing the counts
    // keeps the "counts" hashes small.   Otherwise, if we call clear() each time,
    // we spend a lot of time in clear().
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
    flush_counts(h); // Flush out the counts of this order and higher.
    // at this point, ngram_counts[h].history == "" (and also for higher
    // values of h).
    ngram_counts[h].history = history;
  }
}

/*
  This form of entropy pruning is intended for the case where the counts
  are to be interpreted in the style of Kneser-Ney with interpolation.
  That is-- the backoff probability mass goes to the backoff distribution
  including all words, not just those that were not directly seen.
  
  What this program reads are the discounted counts, in the normal sorted order,
  e.g. after discount_ngrams but before interpolate_ngrams.  It's convenient to
  work with the counts and not the language model directly, because the counts
  give us information about how much we saw each history-state, that's relevant
  to the entropy calculation.

  The idea of this pruning is: we want to get rid of some entries in the LM
  (to save space), and we want to do so in a sensible way.  A sensible way
  seems to be to use the K-L divergence between distributions: that is,
  given observation generated from the original LM, we want to not reduce
  the log-likelihood of them too much, when evaluating them with the
  pruned LM.  This is the one-sided K-L divergence, e.g. if p_o(x) is the
  original distribution and p_p(x) is the pruned distribution, we'd want
  to prune in such a way that the divergence
     \sum_x p_o(x) log(p_o(x) / p_p(x))
  is minimized.  But this is a bit more complicated because the LM has
  a history state.  Suppose the distribution of the history-states in the training
  data is given by p(h).  We'd actually be doing the above as:

     \sum_h p_o(h)  \sum_x p_o(x|h) log(p_o(x|h) / p_p(x|h))    (1)

  We could actually compute the above, but it would involve a very large
  sum over histories and predicted words.  There are a few simplifications
  we make.  One is that after partially pruning the LM, we let p_o be
  the already-partially-pruned LM.  This makes things easier, and should
  not really affect the results for small pruning beams.  The next
  thing is we have to make an approximation that will give us this
  divergence reasonably efficiently.  We end up computing not the
  actual divergence, but an upper bound on it.  Roughly, the idea is
  this.  Suppose we distinguish the symbols produced from a history-state
  when they arise from different backoff states, e.g. instead of just "a"
  we would have a_3 for trigram entries, a_2 for bigram, a_1 for unigram.
  Then the symbol "a" is produced by mapping a_3->a, a_2->a, a_1->a.  We
  can show that the divergence measured on the distribution over the
  marked symbols is >= that measured over the normal symbols.

  In general we can choose which symbols to "mark" and which to consider
  as the same symbol.  Another thing is, some sets of phones "act the
  same way" in terms of our problem, so we can lump them together without
  affecting the results.

  For concreteness, let's assume the n-gram entry we're trying to get
  rid of is an entry like:
x y   a    2.0
  i.e. the predicted-word is a and the history state is "x y".
   (note: histories are reversed, so this means b is the closest word to a).
  The other history-states we'll consider are "b" (the bigram history)
  and "" (the unigram history). 

  Let's use the subscripts 3, 2 and 1
  to refer to these history-states, and counts coming from them.
  The quantities involved in this computation are as follow (interpret "_"
  in the LaTeX manner, as a subscripting device):
  
  c(a,3)      The count of a in the trigram history state; 2.0 in the example here.
  c(*,3)      The backoff count in the trigram history state; e.g. "x y  *  10.0"
  c(%,3)      The total count in the trigram history state (we'd have to sum to get this).
  c(a,2)      The count of a in the bigram history state, e.g. from an entry like
           "x   a   1.0" (may be zero).
  c(%,2)      The total count in the bigram history state (obtained by summing).


  By lumping symbols together that "behave the same way", we can get the following
  picture.  We imagine there is another trigram state backing off to "x"; say,
  this is state "x z".  We don't need to consider any direct trigrams from
  this state, of the form "x z   s" since they don't affect anything.  We just
  conisider the symbols that come via backoff from this state.  Suppose there
  are two symbols like this: "a" itself, and "b".  And let "c" represent all the symbols
  apart from "a" that are produced from state "x y" via bigram or unigram backoff.
  
  Let "r" be the raw counts in the training data of something occurring in a trigram
  history state.   We can work out these raw counts (bearing in mind that they are
  w.r.t. symbols that are not "real" symbols) as:

   r(a, "x y") = c(a,3) + c(*,3) * c(a,2)/c(%,2)
   r(c, "x y") = c(*,3) * (c(%,2) - c(a,2))/c(%,2)
   r(b, "x z") = c(%,2) - c(a,2) - r(c, "x y")
   r(a, "x z") = c(a,2) - c(*,3) * c(a,2)/c(%,2)

  There are no other counts that are relevant to this computation (i.e. that
  change in likelihood when we discount the count c(a,3).  The (bound on the)
  K-L divergence [* total count] is as follows: each term below looks like:
     [count of class observations] * log(p(this class | old model)/p(tihs class | new model))

  r(a, "x y") * log(  (c(a,3) + c(*,3) * c(a,2)/c(%,2)) /   %% Note: we canceled 1/c(%,3)
                      (c(*,3)+c(a,3)) * (c(a,2)+c(a,3))/(c(%,2)+c(a,3)) )
+ r(c, "x y") * log ( (c(*,3) * (c(%,2)-c(a,2))/c(%,2)) /   %% note: we canceled 1/c(%,3)
                       c(*,3)+c(a,3) * (c(%,2)-c(a,2))/(c(%,2)+c(a,3)) )
+ r(b, "x z") * log ( (c(%,2)-c(a,2))/c(%,2) /
                       (c(%,2)-c(a,2)/(c(%,2)+c(a,3))) )
+ r(a, "x z") * log ( (c(a,2)/c(%,2)) /
                      ((c(a,2)+c(a,3))/(c(%,2)+c(a,3))) )
                    
*/

/*
  See the extended comment above for derivation (well, proto-derivation).
  Let's let the history state "x y" (sometimes "3", or trigram) be
  h1, alternative history state "x z" be h2, and backoff state "x" be
  hb.
  
  Here, c_a_h1 is c(a,3),
        c_all_h1 is c(%,3),
        c_bo_h1 is c(%,3),
        c_a_hbo is c(a,2),
        c_all_hbo is c(%,2).
        
*/
       
inline void check_divergence_params(double *c_a_h1, double *c_all_h1, double *c_bo_h1,
                                    double *c_a_hbo, double *c_all_hbo) {
  assert(*c_a_h1 > 0.0);
  assert(*c_all_h1 > 0.0);
  assert(*c_bo_h1 > 0.0);
  assert(*c_a_hbo >= 0.0);
  if (*c_a_hbo == 0.0) *c_a_hbo = 1.0e-20; // Avoids NaN's.
  assert(*c_all_hbo > 0.0);
  if (*c_all_h1 - *c_bo_h1 - *c_a_h1 < -0.0) {
    if (*c_all_h1 - *c_bo_h1 - *c_a_h1 < -0.05) {
      fprintf(stderr, "Remaining probability mass is <-0.05 in this state\n");
    }
    *c_all_h1 = *c_bo_h1 + *c_a_h1;
  }
  if (*c_all_hbo <= *c_a_hbo) { // Should be some mass left for backoff:
    // this is not right.
    fprintf(stderr, "No probability mass left in backoff state\n");
    exit(1);
  }
  if (*c_all_hbo < 0.98*(*c_bo_h1 - 0.2)) { // If this later causes problems, we
    // can make it a warning.
    fprintf(stderr, "Error: backoff mass in history-state is more than total mass "
            "in backoff state: %f < %f\n",
            *c_all_hbo, *c_bo_h1);
    exit(1);
  }
}

double compute_divergence(double c_a_h1, double c_all_h1, double c_bo_h1,
                            double c_a_hbo, double c_all_hbo) {
  // The next call will fix up the arguments if there are small errors.
  check_divergence_params(&c_a_h1, &c_all_h1, &c_bo_h1, &c_a_hbo, &c_all_hbo);
  
  // r_h_h1 is #times symbol a appears (in data generated from model, times #train-counts)
  // from history-state h1.
  double r_a_h1 = c_a_h1 + c_bo_h1 * c_a_hbo / c_all_hbo;
  // r_c_h1 is #times phantom-symbol "c" (representing all not-a symbols generated via
  // backoff) appears from history-state h1.
  double r_c_h1 = c_bo_h1 * ((c_all_hbo-c_a_hbo)/c_all_hbo);
  // r_c_h2 is #times phantom-symbol "b" (representing all all not-a symbols generated
  // via backoff) appears from "phantom history-state" h2 (representing all history-states
  // backing off to h_{bo})).
  double r_b_h2 = c_all_hbo - c_a_hbo - r_c_h1; // I.e. mass in hbo due to all symbols but
  // "a", but excluding that attriibuted to state h1.
  double r_a_h2 = c_a_hbo - c_bo_h1 * c_a_hbo/c_all_hbo; // I.e. mass in hbo for a, but
  // subtracting what comes via backoff from h1.

  double c_nota_hbo = c_all_hbo - c_a_hbo;
  
  double
      term1 = r_a_h1 * log(  (c_a_h1 + c_bo_h1*c_a_hbo/c_all_hbo) /
                             ((c_bo_h1+c_a_h1)*(c_a_hbo+c_a_h1)/(c_all_hbo+c_a_h1)) ),
      term2 = r_c_h1 * log(  (c_bo_h1 * c_nota_hbo/c_all_hbo) /
                             ((c_bo_h1+c_a_h1) * c_nota_hbo/(c_all_hbo+c_a_h1) ) ),
      term3 = r_b_h2 * log(  (c_nota_hbo/c_all_hbo) /
                             (c_nota_hbo/(c_all_hbo+c_a_h1)) ),
      term4 = r_a_h2 * log(   (c_a_hbo/c_all_hbo) /
                              ((c_a_hbo+c_a_h1)/(c_all_hbo+c_a_h1)) );
  double ans = term1 + term2 + term3 + term4;
  return ans;
}

// This function returns the entropy difference we would get by
// removing all counts from this state
// and moving them to the backoff state.  We do this separately from
// "compute_divergence" which just does them one by one, because it
// could be that removing them all is possible even if removing none
// of them individually is possible.
// The change is quite simple to compute (actually this is a bound on
// the change, like the other one)...
// It's a sum over all words seen from this state [...]
// old-prob=[sum[words in this state]..
// +sum [our-words-seen-from-backoff-state]
// +[remaining-words-from-backoff-state]..
// new-prob=sum[words in this state].. (prob | backoff state)
//  +[remaining-words-from-backoff-state]...

// The input to this function is: firstly a list of pairs, where
// the first member of each pair is the count of that word in the
// history-state (h_1) we're considering removing, and the second
// it its count in the backoff state;  c_bo_h1 is the
// total backoff count in history-state h1; and c_tot_bo is the
// total count in the backoff state (which includes "*" and other
// words not seen in state h_1).
// Preconditions include: c_bo_h1 <= c_tot_hbo;
// tot(count_pairs[i].second)  < c_tot_hbo
double compute_divergence_remove_all(
    const std::vector<std::pair<double,double> > &count_pairs,
    double c_bo_h1,
    double c_tot_hbo) {
  double c_tot_h1 = c_bo_h1, c_seen_hbo = 0.0;
  for (size_t i = 0; i < count_pairs.size(); i++) {
    c_tot_h1 += count_pairs[i].first;
    c_seen_hbo += count_pairs[i].second;
  }
  assert(c_tot_h1 != 0.0 && c_seen_hbo < c_tot_hbo);
  if (!(c_bo_h1 <= c_tot_hbo+0.2)) {
    fprintf(stderr, "Warning: c_bo_h1 > c_tot_hbo: %f > %f\n",
            c_bo_h1, c_tot_hbo);
  }
  double divergence = 0.0;
  for (size_t i = 0; i < count_pairs.size(); i++) {
    // count of word (say, "w") in state h1 or h_bo.
    double c_w_h1 = count_pairs[i].first, c_w_hbo = count_pairs[i].second;
    // variables starting r_ are raw-counts of how many times we would observe
    // words in particular history states (sum over direct + backoff mass).
    double r_w_h1 = c_w_h1 + c_bo_h1*(c_w_hbo/c_tot_hbo); // Raw observation count
    // of word "w" in history h1.
    double r_w_h2 = c_w_hbo - c_bo_h1*(c_w_hbo/c_tot_hbo); // Raw observation count of
    // word "w" in history-state h2 (a fake history state encompassing all states that
    // also backoff to h_bo, or were removed and became h_bo).

    double like_w_h1_old = r_w_h1 / c_tot_h1, // this is a simple way of writing the expression.
        like_w_h1_new = (c_w_h1+c_w_hbo) / (c_tot_hbo + (c_tot_h1-c_bo_h1));
    double like_w_h2_old = (c_w_hbo/c_tot_hbo),
        like_w_h2_new = like_w_h1_new;
    divergence += r_w_h1 * log(like_w_h1_old / like_w_h1_new);
    if (r_w_h2 != 0.0)
      divergence += r_w_h2 * log(like_w_h2_old / like_w_h2_new);

  }
  // word "x" is a fake word consisting of all words that we don't have
  // explicit counts for.
  double r_x_h1 = c_bo_h1 * ((c_tot_hbo-c_seen_hbo)/c_tot_hbo);
  double like_x_h1_old = r_x_h1 / c_tot_h1,
      like_x_h1_new = ((c_tot_hbo-c_seen_hbo)/(c_tot_hbo + (c_tot_h1-c_bo_h1)));
  if (r_x_h1 != 0.0)
    divergence += r_x_h1 * log(like_x_h1_old / like_x_h1_new);

  // To get count for x in history-state h2, take total
  // count in hbo and to subtract amount coming from h1.
  double r_x_h2 = c_tot_hbo - c_seen_hbo - r_x_h1;
  double like_x_h2_old = ((c_tot_hbo - c_seen_hbo) / c_tot_hbo),
      like_x_h2_new = ((c_tot_hbo - c_seen_hbo) / (c_tot_hbo + (c_tot_h1-c_bo_h1)));
  if (r_x_h2 != 0.0)
    divergence += r_x_h2 * log(like_x_h2_old / like_x_h2_new);
  assert(divergence >= -0.1);
  return divergence;
}


void test_divergence() {
  for (int i = 0; i < 100; i++) {
    double c_a_h1 = (1 + rand() % 5) * 0.2; // if zero, would not have to prune->
    // don't allow zero.
    double c_bo_h1 = (2 + rand() % 5) * 0.2;
    double c_extra_h1 = (rand() % 5) * 0.2;
    double c_all_h1 = c_a_h1 + c_bo_h1 + c_extra_h1;

    double c_a_hbo = (rand() % 5) * 0.2;
    double c_extra_hbo = (1 + rand() % 4) * 0.2; // can't be zero,
    // as it includes backoff mass.
    double c_all_hbo = c_a_hbo + c_extra_hbo;
    if (c_all_hbo < c_bo_h1) { // This is not possible, so fix it.
      c_all_hbo = c_bo_h1;
    }
    double divergence = compute_divergence(c_a_h1, c_all_h1, c_bo_h1,
                                               c_a_hbo, c_all_hbo);
    // printf("Entropy diff from pruning is: %f\n", divergence);
    assert(divergence >= -0.00001); // make sure not negative.
  }
}

void test_divergence_remove_all() {
  for (int i = 0; i < 100; i++) {
    int num_points = rand() % 4;
    std::vector<std::pair<double,double> > count_pairs;
    double tot1=0, tot2=0.0;
    for (int j = 0; j < num_points; j++) {
      double count1 = (1 + rand() % 4) * 0.4, //count1 must be nonzero.
          count2 = (rand() % 3) * 0.4;  // count2 may be zero.
      count_pairs.push_back(std::make_pair(count1,count2));
      tot1 += count1; tot2 += count2;
    }
    double c_bo_h1 = 1.0 + rand () % 3; // Don't let this be zero.
    double extra_count_hbo = 1.0 + rand() % 3; // don't allow this to be zero either.
    double c_tot_hbo = extra_count_hbo + tot2;
    if (c_tot_hbo < c_bo_h1) c_tot_hbo = c_bo_h1; // this is a condition that couldn't
    // happen in real stats, that is checked for.
    double divergence = compute_divergence_remove_all(count_pairs,
                                                        c_bo_h1,
                                                        c_tot_hbo);
    // printf("Divergence is %f\n", divergence);
    assert(divergence >= -0.00001);
  }
}

double get_total_count(int h) {
  assert(h>=0 && h < ngram_counts.size());
  assert(h == 0 || !ngram_counts[h].history.empty()); // Make sure not
  // called for empty history-state.
  unordered_map<string,double> &counts = ngram_counts[h].counts;
  unordered_map<string,double>::iterator iter = counts.find("%");
  if (iter != counts.end()) return iter->second;
  else {
    double total_count = 0.0;
    for (iter = counts.begin(); iter != counts.end(); ++iter)
      total_count += iter->second;
    if (total_count <= 0.0) {
      fprintf(stderr, "Total count is zero or negative (perhaps "
              "you supplied a broken LM?)\n");
      exit(1);
    }
    counts["%"] = total_count;
    return total_count;
  }
}

inline double get_count(int h, const std::string &predicted) {
  unordered_map<string,double>::iterator iter = ngram_counts[h].counts.find(predicted);
  if (iter == ngram_counts[h].counts.end()) return 0.0;
  else return iter->second;
}

// Removes the specified list of predicted-words from the history-state
// of the specified order, giving their counts to the backoff history
// state.  
void remove_words_from_history_state(const std::vector<std::string> &words,
                                     int h) {
  assert(h>0);
  double tot_count_removed = 0.0;
  for (size_t i = 0; i < words.size(); i++) {
    const std::string &predicted = words[i];
    double count = get_count(h, predicted);
    assert(count > 0.0);
    add_count(h-1, predicted, count); // add this count to backoff state.
    ngram_counts[h].counts.erase(predicted); // Erase count from this state.
    tot_count_removed += count;
  }
  // Increase the total count due to backoff in this history state.
  add_count(h, "*", tot_count_removed);
  // Keep updated the total-count in the backoff history state.
  add_count(h-1, "%", tot_count_removed);
}

// this is called from flush_counts; we do the pruning for a particular
// history-length just before writing out the counts. 
void try_removing_ngrams(int h) {
  assert(h > 0 && h < ngram_counts.size() && ngram_counts[h].history != "");  
  // This routine tries removing individual n-grams from
  // history-state with length h.  I.e. it tests whether the
  // K-L divergence [*total-train-data-count] from removing each
  // N-gram and giving its mass to the backoff state, would
  // exceed the threshold.  Actually to avoid complexities relating
  // to the order of removing them, it tests the thresholds
  // individually, comparing with the original distribution,
  // and then removes all that were below the threshold
  // vs. the original distribution.

  unordered_map<string,double> &counts = ngram_counts[h].counts;
  unordered_map<string,double>::iterator iter;

  assert(counts.count("*") == 1); // make sure we have backoff count.
  double c_all_h1 = get_total_count(h),
      c_bo_h1 = counts["*"],
      c_all_hbo = get_total_count(h-1);

  double this_total_divergence = 0.0; // total divergence
  // from things we plan to remove.
  std::vector<std::string> words_to_remove;
  int total_num_words = 0;
  for (iter = counts.begin(); iter != counts.end(); iter++) {
    const std::string &predicted = iter->first;
    if (*(predicted.c_str()) != '*' && *(predicted.c_str()) != '%') {
      // a real word... suppose it's called "a" (this is what we call it
      // in the math).
      double c_a_h1 = iter->second,
          c_a_hbo = get_count(h-1, predicted);
      double divergence = compute_divergence(c_a_h1, c_all_h1, c_bo_h1,
                                             c_a_hbo, c_all_hbo);
      total_num_words++;
      if (divergence <= threshold) {
        this_total_divergence += divergence;
        words_to_remove.push_back(predicted);
      }
    }
  }
  remove_words_from_history_state(words_to_remove, h);
  total_divergence += this_total_divergence;
  num_params_pruned += (int)words_to_remove.size();
}


void print_stats() { // Print stats of what we removed.
  fprintf(stderr, "Removed %d parameters, total divergence %f\n", num_params_pruned, total_divergence);
  if (num_params_pruned > 0)
    fprintf(stderr, "Average divergence per parameter is %f, versus threshold %f\n",
            total_divergence/num_params_pruned, threshold);
}
