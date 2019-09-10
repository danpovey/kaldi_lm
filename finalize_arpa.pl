#!/usr/bin/perl

# This program reads in the output of interpolate_ngrams --arpa,
# which is lines like:
# 00 3  [this tells us the n-gram order]
# 01 a #-3.634 [this is the unigram prob]
# 01 a -0.43142 [this is the backoff prob]
# 02 a b #-4.31432
# 02 a b -0.4768 [this is the backoff prob]
# 03 a b c  <-3.26532
#
# and it creates the Arpa-format file.  To do this it has to create the
# header with the number of N-grams etc., and lines like:
# -3.634 a -0.43142
# The "hard part" of this is to ensure that we have explicit 
# probabilities for all N-grams that appear as history states,
# and that all history states have the backoff history state
# present.
# This program assumes the input is sorted.
# It requires a directory for temporary files, call this $tdir.
# First it puts the lines from interpolate_ngrams --arpa, with
# the numbers removed, into $tdir/1, $tdir/2, etc.
# While doing this is creates the files $tdir


$check = 1;  #If true, do more extensive
   # checking of arpa: make sure that if history state
   # exists, backoff history state exists too.

@ARGV == 1 || die "usage: interpolate_ngrams --arpa | finalize_arpa.pl tempdir  > arpa";
$tdir = shift @ARGV;

$line = <>;
if (!defined $line) {
  die "finalize_arpa.pl: empty input.";
}
if ($line !~ m:^00 (\d+)$:) {
  die "finalize_arpa.pl: bad input: line is $line";
}
$N = 0 + $1; # N-gram order, e.g. 3.

$last_order = 0;
$last_ngram = ""; # last N-gram we read in.
$last_ngram_prob = "";  # last n-gram prob we read in.

sub flush_ngram;
sub write_ngram_only;
sub write_ngram_and_backoff;
sub write_backoff_only;
sub get_backoff_prob;
sub get_ngram_prob;

@num_ngrams = ();
for ($n = 1; $n <= $N; $n++) { 
  if ($n <= $N-2) {
    $ngram_probs[$n] = {}; #  New reference to anonymous hash.
    $backoff_probs[$n] = {}; #  New reference to anonymous hash.
  } else { # We'll be storing the N-grams in temporary files...
    # In memory we just store the number of N-grams.
    $num_ngrams[$n] = 0; 
    $pipename = "|gzip -c >$tdir/$$.$n.gz";
    open($n, $pipename) || die "Error opening pipe \"$pipename\"";
  }
}

while(<>) {
  @A = split(" ", $_);
  $n = 0 + shift @A; # N-gram order (1,2,3,...)
  ($n>0 && $n <= $N) || die "Invalid order $n\n";
  
  # If $n <= N-2, then keep it in memory; else we'll write it to a temporary
  # file.
  $prob = pop @A;
  $ngram = join(" ", @A);
  if ($prob =~ s:^\#::) { # This is an N-gram prob, 
    # not a backoff prob.  Note: N-gram probs come before the
    # corresponding backoff probs in the input.
    flush_ngram();
    $last_order = $n;
    $last_ngram = $ngram;
    $last_ngram_prob = $prob;
  } else { # This is a backoff prob.  
    if ($ngram eq $last_ngram) { # We have an N-gram prob for this N-gram.
      # store/write out both this and the backoff prob.  Note: $prob
      # is actually a backoff prob.
      write_ngram_and_backoff($n, $ngram, $last_ngram_prob, $prob);
      $last_ngram = "";
    } else {
      flush_ngram(); # Write out previous N-gram, which is unreleated.
      write_backoff_only($n, $ngram, $prob);
    }
  }
}
flush_ngram();


print "\\data\\\n";
for ($n = 1; $n <= $N; $n++) { # Print out the number of N-grams of each order.
  if ($n <= $N-2) {
    my $hashref = $ngram_probs[$n];
    $this_num_ngrams = keys %$hashref;
  } else {
    $this_num_ngrams = $num_ngrams[$n];
  }
  print "ngram $n=$this_num_ngrams\n";
}
print "\n";
for ($n = 1; $n <= $N; $n++) {
  print "\\$n-grams:\n";
  if ($n <= $N-2) {
    my $hashref = $ngram_probs[$n];
    my $b_hashref = $backoff_probs[$n];
    foreach $key (sort keys %$hashref) {
      my $prob = $$hashref{$key};
      my $backoffprob = $$b_hashref{$key};
      if (defined $backoffprob) {
        print "$prob  $key $backoffprob\n";
      } else {
        print "$prob  $key\n";
      }
    }
    if ($check) { 
      foreach $key (keys %$b_hashref) {
        defined $$hashref{$key} || die "No prob where backoff prob exists, for key $key.";
      }
    }
  } else {
    open($n, "gunzip -c $tdir/$$.$n.gz|") || die "Opening pipe to read $tdir/$$.$n.gz";
    while(<$n>) { print; }
  }
  print "\n";
}
print "\\end\\\n";



sub flush_ngram {
  # This function writes (to memory or disk) whatever's stored in $last_ngram,
  # and sets $last_ngram to ""
  if ($last_ngram ne "") {
    write_ngram_only($last_order, $last_ngram, $last_ngram_prob);
    $last_ngram = "";
  }
}

sub write_ngram_only {
  my ($n, $ngram, $prob) = @_;
  if ($n > $N-2) {  # Write to file; increment number of N-grams of this
    # order.
    $num_ngrams[$n]++;
    print $n "$prob  $ngram\n" || die "Error writing to $n'th pipe";
  } else { # Store in memory.
    my $ngram_hashref = $ngram_probs[$n];
    $$ngram_hashref{$ngram} = $prob;
  }
}

sub write_ngram_and_backoff {
  my ($n, $ngram, $prob, $backoff_prob) = @_;
  if ($n > $N-2) { # Write it to a file.
    $num_ngrams[$n]++;
    print $n "$prob  $ngram $backoff_prob\n";
  } else {
    my $ngram_hashref = $ngram_probs[$n];
    $$ngram_hashref{$ngram} = $prob;
    my $backoff_hashref = $backoff_probs[$n];
    $$backoff_hashref{$ngram} = $backoff_prob;
    if ($n>1 && $check) {
      my @ngram = split(" ", $ngram);
      shift @ngram; # get backoff hist state.
      get_backoff_prob(@ngram); # Just make sure it exists;
      # we don't use this value.
      get_ngram_prob(@ngram); # Make sure N-gram prob also exists.
    }
  }
}

sub write_backoff_only {
  my ($n, $ngram, $backoff_prob) = @_;
  # This is called when only a backoff prob was specified in the input
  # N-grams.
  # This function works out the probability of the N-gram 
  # from data we have in memory (if necessary, it creates
  # new backoff states), and it calls write_ngram_and_backoff
  # with this info.
  $n < $N || die;
  my $ngram_prob = get_ngram_prob(split(" ", $ngram));
  write_ngram_and_backoff($n, $ngram, $ngram_prob, $backoff_prob);
  if ($n>1 && $check) {
    my @ngram = split(" ", $ngram);
    shift @ngram; # get backoff hist state.
    get_backoff_prob(@ngram); # Just make sure it exists;
    # we don't use this value.
    get_ngram_prob(@ngram); # Make sure N-gram prob also exists.
  }
}

sub get_backoff_prob { # Gets backoff prob for this history state,
  # as log-10 value.
  # Sets it to (in log-10) 0.000 if doesn't exist.
  my @hist = @_;
  my $n = @hist;
  $n <= $N-2 || die;
  
  my $backoff_hashref = $backoff_probs[$n];
  defined $backoff_hashref || die "No backoff hash for n=$n";
  my $hist_str = join(" ", @hist);
  if (defined $$backoff_hashref{$hist_str}) { # backoff prob is defined for this hist state.
    return $$backoff_hashref{$hist_str};
  } else { # We'll have to define the backoff prob for this hist state (as 1),
    # and make sure the N-gram prob is defined for his word sequence, so we don't
    # get "orphan" backoff probs that ARPA can't handle.
    get_ngram_prob(@hist); # We don't need the return value: just make sure it's defined.
    # Note: we don't have to check that the backoff prob is defined for the
    # less-specific history state, as get_ngram_prob (which calls this)
    # will also query probabilities for the less-specific history state, which
    # will automatically ensure this.
    $$backoff_hashref{$hist_str} = 0.0;
    return 0.0;
  }
}

sub get_ngram_prob { # Works out the N-gram prob
  # (and stores it if $n<=$N-2), and returns it.
  my @ngram = @_;
  my $n = @ngram;
  my $ngram_str = join(" ", @ngram);
  my $hashref;
  if ($n <= $N-2 && 
      defined ($hashref=$ngram_probs[$n]) # This should always be defined..
      && defined $$hashref{$ngram_str}) {
    return $$hashref{$ngram_str};
  } elsif ($n == 1) { # Can't backoff -> return -99 (should only happen for <s>)
    if ($ngram_str ne "A" && $ngram_str ne "<s>") {
      print STDERR "Warning: n-gram not defined for $ngram_str, setting to -99\n";
    }
    $hashref = $ngram_probs[1];
    my $log_prob = -99;
    $$hashref{$ngram_str} = $log_prob;
    return $log_prob;
  } else { # Go to backoff.
    my @backoff_ngram = @ngram;
    shift @backoff_ngram;
    pop @ngram;
    my $ans = get_backoff_prob(@ngram) + get_ngram_prob(@backoff_ngram);

    get_ngram_prob(@ngram); # Ensure N-gram prob exists for this N-gram.

    if ($n <= $N-2) {
      my $hashref = $ngram_probs[$n];
      $$hashref{$ngram_str} = $ans;
    }
    return $ans;
  }
}
