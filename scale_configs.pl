#!/usr/bin/perl

# This script requires the names of two config files, where 
# the first has lines like
#
# D=0.6 tau=1.0 phi=2.0
#
# and the second is the same except the values 
# of the constants are all 0 or 1;
# and it takes a constant alpha (which should be between -1 and 1).
# This script will produce (on its stdout) a config file that's
# the same as the first one, except the D and tau values where
# the second config file was 1, are scaled by multiplying them by 
# (1 + alpha), and the phi values are transformed by:
#   phi <-- ((phi-1)*alpha ) + 1
# [i.e. we do the scaling on phi-1; this will ensure that
# phi never goes below zero].


@ARGV == 3 || die "usage: add_configs.pl config1 config2 alpha >config";

($config1,$config2,$alpha) = @ARGV;

$alpha =~ m/[0-9\.-eE]/ || die "Expecting numeric third argument";

open(C1, "<$config1") || die "Failed to open config file $config1";
open(C2, "<$config2") || die "Failed to open config file $config2";

# This program is longer than it needs to be.  To do the
# ensure-nonnegative stuff I copy-pasted and modified the
# entire main loop.

while (<C1>) {
  $line1 = $_;
  $line2 = <C2>;
  if (!defined $line2) {
    print $line1;               # Treat all line2 values as zero.
  } else {
    $line1 =~ m:^\s*D=(.+)\s+tau=(.+)\s+phi=(.+)\s*$: || die "bad line $line1";
    $D = $1; $tau = $2; $phi = $3;
    $line2 =~ m:^\s*D=(.+)\s+tau=(.+)\s+phi=(.+)\s*$: || die "bad line $line2";
    $opt_D = $1; $opt_tau = $2; $opt_phi = $3;
    if (($opt_D != 0 && $opt_D != 1) 
        ||($opt_tau != 0 && $opt_tau != 1) 
        ||($opt_phi != 0 && $opt_phi != 1) ) {
      die "Invalid values in $line2";
    }
    if ($opt_D == 1) {
      $D *= (1.0+$alpha);
    }
    if ($opt_tau == 1) {
      $tau *= (1.0+$alpha);
    }
    if ($opt_phi == 1) {
      $phi -= 1.0;
      $phi *= (1.0+$alpha); # do the scaling on phi-1.  This ensures phi never gets <1.
      $phi += 1.0;
    }
    print "D=$D tau=$tau phi=$phi\n";
  }
}

<C2> && die "Too many lines in config file $config2";
