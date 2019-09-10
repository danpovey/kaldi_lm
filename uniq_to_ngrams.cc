#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

inline void process_line(char *line_in) {
  while (isspace(*line_in)) line_in++;
  if (!isdigit(*line_in)) {
    fprintf(stderr, "Bad input to uniq_to_ngrams: no digit on line\n");
    exit(1);
  }
  char *digits = line_in;
  while(isdigit(*line_in)) line_in++; // skip over spaces.
  assert(isspace(*line_in));
  *line_in = '\n';
  line_in++;
  int digits_str_len = line_in - digits;
  int stdout_fd = 1;
  int len = strlen(line_in);
  line_in[len-1] = '\t'; // Change the final \n to a tab (the number will
  // come after this).
  write(stdout_fd, line_in, len);
  write(stdout_fd, digits,digits_str_len); // write the digits then "\n".
}
  
int main(int argc, char **argv) {
  // This program takes the output of uniq -c, and changes from the format
  //      2 A b c
  // to
  // A b c 2
  // I.e. it puts the count after the text.
  
  if (argc != 1) {
    fprintf(stderr, "Usage: uniq_to_ngrams <uniq-output >ngrams\n");
    exit(1);
  }
  size_t nbytes = 100;
  char *line = (char*) malloc(nbytes + 1);
  int bytes_read;
  while ((bytes_read = getline(&line, &nbytes, stdin)) != -1) {
    process_line(line);
  }
  delete line;
}
