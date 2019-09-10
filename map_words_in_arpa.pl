#!/usr/bin/perl


# This program takes as its first argumenta word-map, with, on each line:
# normal-word short-word
# e.g.
# AND A
# and as stdin (or 2nd argument) an arpa LM as output 
# by counts_to_arpa.pl, and it maps from short to long
# form of the words.

(@ARGV == 1 || @ARGV == 2) || die "Usage: word-map [arpa-LM-in]\n";

$wordmap = shift @ARGV;
open(W, "<$wordmap") || die "Opening word-map file $wordmap";
while(<W>){
  @A = split(" ", $_);
  @A == 2 || die "Bad line in word-map file: $_";
  $map{$A[1]} = $A[0]; # Apply the map in the reverse sense...
}

while(<>) {
  @A = split(" ", $_);
  if (!($A[0] eq "ngram")) {
    for ($n = 0; $n < @A; $n++) {
      if ($A[$n] =~ m:^[a-zA-Z\200-\376]+$:) {
        # Note: \200 and \376 are the octal for 128 and 254: the
        # non-ASCII range that we also use in our short-form words.
        $long_form = $map{$A[$n]};
        if (!defined $long_form) {
          die "No such short form word: $A[$n]";
        }
        $A[$n] = $long_form;
      }
    }
  }
  print join(" ", @A), "\n";
}

