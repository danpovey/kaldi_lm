#include "kaldi_lm.h"
#include "math.h"

// see main() for top-level comments.



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
std::vector<std::string> wordlist; // list of all words except "A" and "C" (begin-of-sent and <UNK>).
double unk_fraction = 0.0;
bool write_arpa = false; // write output that can easily be converted to Arpa format.

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

// This is called from flush_counts; we make sure for each history that
// it has been interpolated, before writing out the counts.  It checks
// by looking for the predicted-string "%" whether the interpolation
// has already been done.  If not, it interpolates by adding probability
// mass taken from lower orders.  It first calls interpolated_if_needed
// for the one-lower order (if h>0), because its main operation needs
// the one-lower order to be interpolated.
void interpolate_if_needed(int h);

// this is called from flush_counts; it writes out any counts that
// are currently stored, with history-length h >= 0.  It requires that
// the history be valid (e.g. not empty, for orders > 0).
void write_counts(int h);

// this version of write_counts is called if you give the --arpa
// option; it outputs in a format that can be easily converted
// to Arpa.
void write_counts_arpa(int h);

// Flush the counts for history-length h (with 0 <= h < ngram_counts.size()),
// and any higher orders.  This will interpolate the counts, write them out, and set to empty
// the history and the "counts" of that order (and higher).  Calls
// interpolate_if_needed and write_counts.
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
    fprintf(stderr, "You cannot call interpolate_ngrams with ngrams that have already "
            "been interpolated.\n");
    exit(1);
  }
  add_count(h, predicted, count);
  // Note: most of the output of this program gets produced in
  // "ensure_history_matches", which produces output (via
  // flush_counts -> write_counts) when a history does not match
  // and needs to be got rid of.  We also produce output
  // at the very end when we flush the last histories (including
  // unigram).
}

int main(int argc, char **argv) {
  // This program applies interpolation to N-gram counts, in the fashion of
  // "interpolated Kneser-Ney".  It is not a mechanism to interpolate
  // different sources of N-gram counts (this does not exist but would
  // probably basically consist of scaling one of them and then applying sort
  // and merge_counts_online).
  //
  // This program is generally invoked with its input as the output of
  // discount_ngram_counts or prune_ngram_counts, and its output
  // will in turn be read.
  //
  // The code is similar to discount_ngrams.cc.
  // It uses memory O(N * #words).
  //
  // For each history state, this program works out the total probability
  // mass (summing up predicted-words and "*") and stores it as "%".  Then,
  // for all actual predicted-words (i.e. not "*" or "%"), it adds in
  // to the count, the probability generated by "backoff with interpolation",
  // representing this as a count.  
  //
  // With the --arpa option, it outputs something that can, with minimal
  // processing, be turned into an arpa file.  It will output lines
  // that can quite easily be sorted and then converted to ARPA format:
  // a line
  // 00 3
  // where the "3" is the N-gram order; then lines like
  // 01 a -0.043
  // for the backoff prob, and
  // 01 a #-3.432
  // for the N-gram probs, or e.g.
  // 03 a b c #-2.832
  // This should be piped into sort and then finalize_arpa.pl
  
  if (argc>1 && !strcmp(argv[1], "--arpa")) {
    write_arpa = true;
    argc--;
    argv++;
  }
  
  if (argc != 3) {
    fprintf(stderr, "Usage: interpolate_ngrams [--arpa] wordlist unk-fraction <ngrams\n"
            "See comments in code; this is interpolation in the sense of\n"
            "\"interpolated Kneser-Ney\" and is typically necessary prior to converting to\n"
            "Arpa format or evaluating perplexities.\n"
            "The wordlist and unk-fraction relate to what happens to the discounted\n"
            "mass in unigram state: unk-fraction of it goes to <UNK> (actually, C, in\n"
            "our special format), and the rest goes equally to all words in the wordlist\n"
            " (apart from begin-of-sentence \"A\" and unk-symbol \"C\")\n"
            "The --arpa option outputs something that can easily be turned into\n"
            "an ARPA format language model; see comments in code for details.\n");
    exit(1);
  }

  read_wordlist(argv[1]); // reads into wordlist, which is a global variable.
  char *tail;
  unk_fraction = strtof(argv[2], &tail); // unk_fraction is a global variable.
  if (unk_fraction < 0 || unk_fraction > 1.0 || tail == argv[2] || *tail != '\0') {
    fprintf(stderr, "Bad unk-fraction: %s\n", argv[2]);
    exit(1);
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

// this is called from flush_counts; we do the interpolation for a particular
// history-length just before writing out the counts.  Note: this function may
// recursively call itself with a lower value of h, since it needs the
// lower-order history states to be interpolated.

void interpolate_if_needed(int h) {
  // First comes the "if_needed" part: we look for the presence of
  // "%" in the hash to tell us if we already did interpolation for
  // this history state.
  assert(h < ngram_counts.size());
  assert(h==0 || ngram_counts[h].history != "");
  unordered_map<string, double> &counts = ngram_counts[h].counts;
  
  if (counts.find("%") != counts.end())
    return; // Nothing to do; we've done it before.
  if (h > 0) // Ensure that for lower-order histories, we've interpolated.
    interpolate_if_needed(h-1);

  // OK, now the real work begins.
  unordered_map<string, double>::iterator iter = counts.find("*");
  if (iter == counts.end()) {
    fprintf(stderr, "No count for backoff symbol \"*\" in history "
            "state \"%s\": maybe you gave non-discounted N-grams to "
            "interpolate_ngrams?\n",  ngram_counts[h].history.c_str());
    exit(1);
  }
  double backoff_count = iter->second;
  double total_count = 0.0;
  for (iter = counts.begin(); iter != counts.end(); iter++)
    total_count += iter->second; // everything, including "*"
  assert(total_count > 0.0);
  if (h > 0) {
    // Backoff counts we add for words
    // are backoff_count * probability of that word given the backoff state.
    for (iter = counts.begin(); iter != counts.end(); iter++) {
      const std::string &predicted = iter->first;
      if (predicted[0] == '*') continue; // Not a real word.
      iter->second += backoff_count * get_prob(h-1, predicted); //  *** Here is where the
      // main work gets done***
    }
  } else { // h = 0: we're in the unigram history state, so we
    // need to do something special with the backoff mass.  "unk_fraction'
    // we assign to <UNK> ("C" in our special alphabet), and 1.0-unk_fraction
    double extra_unk_count = unk_fraction * backoff_count;
    if (extra_unk_count != 0.0) increment_hash("C", extra_unk_count, &counts);
    assert(wordlist.size() != 0);
    double extra_word_count = ((1.0-unk_fraction)*backoff_count) / wordlist.size();
    if (extra_word_count != 0.0)
      for (size_t i = 0; i < wordlist.size(); i++)
        increment_hash(wordlist[i], extra_word_count, &counts);
    counts.erase("*"); // Erase the backoff count from the counts of the
    // unigram history; it no longer has any meaning (e.g. it would not
    // be used when evaluating probabilities with the model)
  }
  counts["%"] = total_count; // Make a record of the total count.  This is done
  // in the unigram and non-unigram cases.  It's needed to form probabilities from
  // counts efficiently.
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

// Given one of our reversed histories, this
// gives history in the conventional order (with a space at the end if nonempty).
std::string reverse_history(const std::string &history) {
  size_t pos;
  if ( (pos=history.find(' ')) != std::string::npos) {
    std::string s1(history, 0, pos),
        s2(history, pos+1);
    return reverse_history(s2) + std::string(" ") + s1;
  } else {
    return history;
  }
}

void write_counts_arpa(int h) {
  // We output the following numbered lines (not in numbered order; you should
  // put them through "sort", and then process them with finalize_arpa.pl to
  // make the final Arpa file.  This has to fill in "holes" in the arpa,
  // to ensure that if a backoff prob exists for a word sequence, an explicit
  // N-gram prob also exists.
  // Note: the things in square brackets are our
  // comments about this, they're not really written out.
  // Note: the word-sequences are in the standard order as in ARPA files,
  // reflecting the order the words appeared.
  //
  // 00 3 [this is the N-gram order]
  // 01 a #-3.634 [this is the unigram prob]
  // 01 a -0.43142 [this is the backoff prob]
  // 02 a b #-4.31432
  // 02 a b -0.4768 [this is the backoff prob]
  // 03 a b c  #-3.26532
  
  assert(h < ngram_counts.size() && h >= 0);
  std::string history = reverse_history(ngram_counts[h].history.c_str()); // This
  // gives history in the conventional order
  if (history != "") history = history + std::string(" ");
  const char *history_str = history.c_str();
  double inv_log10 = 1.0/log(10.0);
  unordered_map<string, double> &counts = ngram_counts[h].counts;
  assert(counts.count("%") != 0);
  double tot_count = counts["%"];
  for (unordered_map<string, double>::iterator iter = counts.begin();
       iter != counts.end();
       ++iter) {
    const char *predicted_str = iter->first.c_str();
    double count = iter->second;
    assert(count >= 0); // there's no valid way we could have
    // negative counts.
    double prob = count / tot_count,
        log10_prob = log(prob) * inv_log10;
    if (*predicted_str == '*') { // print backoff prob...
      assert(h>0); // shouldn't have this for h==0.
      printf("%02d %s%.5f\n", h, history_str,
             log10_prob); // e.g. "02 a b -0.736"; note, history_str has space.
    } else if (*predicted_str != '%') {
      printf("%02d %s%s #%.5f\n", h+1, history_str,
             predicted_str, log10_prob); // e.g. 03 a b c <-3.634
    }
  }

  if (h == 0) { // Since this function only gets called once
    // for h=0, we use this an opportunity to print out the N-gram order.
    printf("00 %d\n", (int)ngram_counts.size());
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
    interpolate_if_needed(h); // If needed (i.e. if not already called while
    // writing out a higher-order history), apply interpolation to these
    // counts.
    if (write_arpa) write_counts_arpa(h); // Write the counts out in  arpa-ready format.
    else write_counts(h); // Write the counts out in ngram format.
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

void read_wordlist(const char *wordlist_filename) {
  FILE *f = fopen(wordlist_filename, "r");
  if (!f) {
    fprintf(stderr, "interpolate_ngrams: could not open word-list from %s\n",
            wordlist_filename);
    exit(1);
  }
  wordlist.clear();
  size_t nbytes = 100;
  char *line = (char*) malloc(nbytes + 1);
  int bytes_read;
  while ((bytes_read = getline(&line, &nbytes, f)) != -1) {
    int word_len = strlen(line) - 1;
    assert(line[word_len] == '\n');
    line[word_len] = '\0';
    assert(strpbrk(line, " \t\n") == NULL && word_len >= 1);
    if (strcmp(line, "A") && strcmp(line, "C")) // the word is not "A" (begin of sent) or
      wordlist.push_back(std::string(line));  // "C" (unk).
  }
  if (wordlist.size() == 0) {
    fprintf(stderr, "Error: empty wordlist (from %s)\n", wordlist_filename);
    exit(1);
  }
  fprintf(stderr, "interpolate_ngrams: %d words in wordslist\n", (int)wordlist.size());
  delete line;
  fclose(f);
}
