#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <ctype.h>
#include <stdlib.h>

inline void process_line(char *line_in, int N) {
  static char *buf = NULL;
  static int bufsz = 0;
  // make sure buffer is large enough.
  int need_sz = (20 + strlen(line_in))*(N+1); // should be more than enough.
  if (buf == NULL || bufsz < need_sz) {
    delete buf;
    buf = new char[need_sz];
    bufsz = need_sz;
  }
  
  char *cur_buf_ptr = buf;
  
  char *line = line_in;
  static std::vector<int> positions;
  positions.clear();

  if (*line == '\n') {
    line += 1;
  } else {
    assert(!isspace(*line));
    while (*line != '\0') {
      positions.push_back(line - line_in);
      while(!isspace(*line)) line++;
      if (isspace(*line)) line++;
      assert(!isspace(*line) && "get_raw_ngrams doesn't accept extra spaces!");
      // We don't allow >1 space in succession.
      // This program expects very specific input.
    }
  }
  positions.push_back(line - line_in); // position one past terminating "\n".
  assert(line[-1] == '\n'); // We expect the line to be
  // terminated by newline.  The last position is the
  // position of the terminating '\0'
  line[-1] = ' '; // Replace the newline by a space.
  int num_words = positions.size() - 1;
  for (int n = 0; n <= num_words; n++) {
    // First output the history, starting from most recent word.
    for (int m = n-1; m > n-N && m >= 0; m--) {
      int len = positions[m+1] - positions[m] - (m == n-N+1 ? 1 : 0);
      // omit the space if this is the last history we'll write.
      strncpy(cur_buf_ptr, line_in + positions[m], len);
      cur_buf_ptr += len;
    }
    if (n < N-1) {  // Include A (begin-of-sentence) in the history.
      strncpy(cur_buf_ptr, "A\t", 2);
      cur_buf_ptr += 2;
    } else {
      strncpy(cur_buf_ptr, "\t", 1);
      cur_buf_ptr += 1;
    }
    // Now write the predicted word.
    if (n < num_words) {
      int len = positions[n+1]-positions[n]-1;
      strncpy(cur_buf_ptr, line_in + positions[n], len);
      cur_buf_ptr += len;
      strncpy(cur_buf_ptr, "\n", 1);
      cur_buf_ptr += 1;
    } else {
      strncpy(cur_buf_ptr, "B\n", 2); // Predicted word is B (end-of-sentence marker).
      cur_buf_ptr += 2;
    }
  }
  int stdout_fd = 1;
  write(stdout_fd, buf, cur_buf_ptr - buf);
}
  

int main(int argc, char **argv) {
  // This program takes one argument, the value of N
  // (in N-grams.)  t reads the stdin, e.g.
  // a b c
  // d e
  // and (suppose N=3), it outputs the N-grams with the histories first (reversed),
  // then the predicted word, i.e.:
  // A\ta
  // a A\tb
  // b a\tc
  // c b\tB
  // A\td
  // etc.
  // We assume A is the begin-of-sentence symbol and B is the end-of-sentence
  // symbol.  We'll pipe this output into sort and uniq -c.
  // Note: this program will not accept lines with "extra" spaces!
  if (argc != 2) {
    fprintf(stderr, "Usage: get_raw_ngrams N <text\n"
            "Warning: this program will not work if there are extra spaces\n"
            "in the input.  It uses our \"special names\" (A,B,C) for <s>,\n"
            "</s> and <UNK>.\n");
    exit(1);
  }
  int N = atoi(argv[1]);
  if (N == 0) { printf("Invalid value N = %s " , argv[1]); }
  size_t nbytes = 100;
  char *line = (char*) malloc(nbytes + 1);
  int bytes_read;
  while ((bytes_read = getline(&line, &nbytes, stdin)) != -1) {
    process_line(line, N);
  }
  delete line;
}
