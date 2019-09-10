#include "kaldi_lm.h"


struct DiscountParameters {
  float D; // absolute discounting.  D>=0.0
  float tau; // We'll eventually discount by D+tau.   tau>=0.0
  float phi;  // phi >= 1.0, increases the length of time it takes tau to
  // fully kick in... (larger phi = slower increase).
  DiscountParameters(): D(0.0), tau(0.0), phi(0.0) {}
};

// The following variable stores the config parameter we read in, i.e.
// the values of D and tau for each N-gram order (actually it is indexed
// by history length, which is one less than the n-gram order).
std::vector<DiscountParameters> discount_params;

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

// Note: throughout, the history-length "h" is always one less than
// the associated n-gram order (n).

// this is called from flush_counts; we do the discounting for a particular
// history-length just before writing out the counts.  Note: this
// will also write to the counts of one-lower order. 
void apply_discounts(int h);

// this is called from flush_counts; it writes out any counts that
// are currently stored, with history-length h >= 0.  It requires that
// the history be valid (e.g. not empty, for orders > 0).
void write_counts(int h);

// Flush the counts for history-length h (with 0 <= h < ngram_counts.size()),
// and any higher orders.  This will discount the counts, write them out, and set to empty
// the history and the "counts" of that order (and higher).  Calls apply_discounts
// and write_counts.
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
inline void add_count(int h, const std::string predicted, double count);


double apply_discount(double *count, float D, float tau, float phi) {
  if (*count <= D) {
    double ans = *count;
    *count = 0;
    return ans;
  } else {
    assert(phi >= 1.0 && tau >= 0);
    *count -= D;
    double tau_discount = tau * *count / (*count + (phi*tau));
    *count -= tau_discount;
    return D + tau_discount;
  }
}

// see comments by declaration.
inline void add_count(int h, 
                      const std::string predicted,
                      double count) {
  unordered_map<string, double> &counts = ngram_counts[h].counts;
  unordered_map<string, double>::iterator iter = counts.find(predicted);
  if (iter == counts.end())
    counts.insert(std::pair<const std::string, double>(predicted, count));
  else
    iter->second += count;
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
  add_count(h, predicted, count);
  // Note: most of the output of this program gets produced in
  // "ensure_history_matches", which produces output (via
  // flush_counts -> write_counts) when a history does not match
  // and needs to be got rid of.  We also produce output
  // at the very end when we flush the last histories (including
  // unigram).
}

void read_parameters(const char *config_file,
                     std::vector<DiscountParameters> *v) {
  v->clear();
  FILE *f = fopen(config_file, "r");
  if (!f) {
    fprintf(stderr, "Could not open config file %s\n", config_file);
    exit(1);
  }
  DiscountParameters params;
  int n;
  while ((n=fscanf(f, "D=%f tau=%f phi=%f\n", &params.D, &params.tau, &params.phi)) == 3) {
    v->push_back(params);
    fprintf(stderr, "discount_ngrams: for n-gram order %d, D=%f, tau=%f phi=%f\n", (int)v->size(),
            params.D, params.tau, params.phi);
    assert(params.D >= 0.0 && params.tau >= 0.0 && params.phi >= 1.0);
  }
  if (n != EOF) {
    fprintf(stderr, "Bad line in config file (or junk at end of file)\n");
    exit(1);
  }
  fclose(f);
}


int main(int argc, char **argv) {
  // This program discounts sorted N-gram counts, discounting multiple
  // orders at the same time.  It uses some nice properties of the sorted
  // order to avoid using much memory;  its memory usage is limited to
  // O(vocab-size * N), where N is the N-gram order (e.g. 3 or 4).
  // You have to sort its output to get a valid N-gram counts file.

  // This program takes as its single command-line argument a file
  // containing information about how to discount.  The file should
  // contain a series of lines, one for each N-gram order that
  // must be discounted (1, 2, 3..),
  // and each line looks like (e.g.:
  // D=1.0 tau=0.5 phi=2.0
  // i.e. it contains the D value and the tau value
  // for that order.  D is absolute discounting, tau is a discounting
  // method inspired by relevance map.
  //
  // rest of comment assumes phi=1.0
  // Suppose for N=3, we have D=1.0 and tau 2.0, and this program sees
  // a line like:
  // b a   c   10.0
  // (where the multiple-spaces represent tabs).
  // [Note: it would not touch a line like "b a * 5.0" because "*" is
  // special: it represents an amount discounted from an LM state].
  // This program would work out the amount to discount as follows.
  // Firstly  it absolutely
  // discounts by D, so we discount by 1.0.  Then we further discount by
  //  count * tau / (count+tau),
  // or in this case, 9 * 2.0 / (9+2.0) = 1.63.  So the total amount
  // discounted would be 2.63.  It would print:
  // b a   c   7.37
  // b    c   2.63
  // b a   *   2.63
  // But this is only the case if that's the only input with those
  // histories, etc.  It adds everything up for you; it just outputs
  // it in the wrong order (hence the need for 'sort').
  
  if (argc != 2) {
    fprintf(stderr, "Usage: discount_ngrams_online config_file <ngrams\n"
           "See comments in code and see also discount_ngrams which\n"
           "can discount for multiple orders at the same time\n");
    exit(1);
  }

  read_parameters(argv[1], &discount_params);
  ngram_counts.reserve(discount_params.size() + 4); // you can ignore this; it's just
  // a way to help avoid copying hashes around.
  
  size_t nbytes = 100;
  char *line = (char*) malloc(nbytes + 1);
  int bytes_read;
  while ((bytes_read = getline(&line, &nbytes, stdin)) != -1) {
    process_line(line);
  }
  if (ngram_counts.size() != 0)
    flush_counts(0); // Flushes the last counts that were read, and the unigram
  // history state.
  delete line;
}


// this is called from flush_counts; we do the discounting for a particular
// history-length just before writing out the counts.  Note: this
// will also write to the counts of one-lower order.
void apply_discounts(int h) {
  static std::vector<string> counts_to_erase; // static so we don't
  // alloc memory each time.
  counts_to_erase.clear();
  assert(h < ngram_counts.size());
  if (h >= discount_params.size() ||
      (discount_params[h].D == 0 && discount_params[h].tau == 0))
    return; // Nothing to do, as no discounting specified for this order.
  float D = discount_params[h].D, tau = discount_params[h].tau, phi = discount_params[h].phi;
  unordered_map<string, double> &counts = ngram_counts[h].counts;
  unordered_map<string, double> &last_counts =
      ngram_counts[h>0 ? h-1:0].counts; // if h==0 we won't be using this.
  double discounted_amount = 0.0; // This will get written to "*".
  for (unordered_map<string, double>::iterator iter = counts.begin();
       iter != counts.end();
       ++iter) {
    const std::string &predicted = iter->first;
    double count = iter->second;
    assert(*(predicted.c_str()) != '%' && "You shouldn't be discounting already-interpolated N-gram counts"); // Note:
    // "interpolated" in this sense is as in interpolated Kneser-Ney.
    if (*(predicted.c_str()) == '*') continue; // Don't discount from "*", which
    // is the discounted-amount.

    double discounted = apply_discount(&count, D, tau, phi);
    discounted_amount += discounted; // will add to the "*" entry.
    if (discounted != 0 && h > 0) { // Add to the count of lower order.
      unordered_map<string, double>::iterator last_iter = last_counts.find(predicted);
      if (last_iter == last_counts.end())
        last_counts.insert(std::pair<const std::string, double>(predicted, discounted));
      else
        last_iter->second += discounted;
    }
    if (count == 0) counts_to_erase.push_back(predicted);
    else iter->second = count;
  }
  for (std::vector<string>::iterator iter = counts_to_erase.begin();
       iter != counts_to_erase.end();
       ++iter) // these counts went to zero: remove them from the hash.
    counts.erase(*iter);

  assert(discounted_amount >= 0.0);
  if (discounted_amount > 0.0) { // We discounted something..
    unordered_map<string, double>::iterator iter = counts.find("*");
    if (iter == counts.end())
      counts.insert(std::pair<const std::string, double>("*", discounted_amount));
    else
      iter->second += discounted_amount;
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
    apply_discounts(h); // Apply discounts for this order history.
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
    if (ngram_counts[h].history.compare(history) > 0) {
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
