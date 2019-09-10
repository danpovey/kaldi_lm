#!/usr/bin/perl

# This script takes as command-line arguments three values of alpha, which must
# be respectively negative zero and positive, and three perplexities, measured
# at those values of alpha.  It will print on its stdout the optimized value of
# alpha, which it computes based on a quadratic assumption (i.e. it tries to
# approximate the perplexity curve with a quadratic, and find its minimum).
# However, it will never go more negative than twice the negative alpha value,
# or more positive than twice the positive alpha value [this is for safety, so
# we don't go haywire when the curve isn't very quadratic].  Also, if the curve
# is not convex, it returns double the value of alpha that gave the best
# perplexity.
#
# Note: normally we expect the alpha values to be something
# like -0.25, 0, +0.4

# assumed to be measured at alpha=-0.5, alpha=0, and alpha=0.5,
# where alpha measures how far we go along some direction in
# config-space.
# It prints out the optimum value of alpha based on a quadratic
# assumption, but not going outside the range [-1, 1].

@ARGV == 6 || die "Usage: optimize_alpha.pl alpha1 perplexity\@alpha1 alpha2 perplexity\@alpha2 alpha3 perplexity\@alph3";

($alpha1,$p1,$alpha2,$p2,$alpha3,$p3) = @ARGV; 
($alpha2 == 0) || die "Expecting alpha2 to be zero, got $alpha2";
($alpha1 < 0  && $alpha1 > -0.5) || die "Bad value of alpha1: $alpha1";
($alpha3 > 0  && $alpha3 < 0.5) || die "Bad value of alpha3: $alpha3";

($p1>0.0&&$p2>0.0&&$p1>0.0) || die "Bad perplexities $x $y $z.";

# Fit a quadratic, as follows.  if $p1 is f(alpha1), $p2 is f(alpha2) and $p3 is f(alpha3),
# and f is quadratic, i.e.
# f = ax^2 + bx + c
# and alpha2 = 0, then
# c = p2
# Next, to work out alpha2, we have to cancel alpha1, so write down
#  p1 = a alpha1^2 + b alpha1 + c 
#  p3 = a alpha3^2 + b alpha3 + c 
# NExt:
#  p1-c = a alpha1^2 + b alpha1  (1)
#  p3-c = a alpha3^2 + b alpha3  (3)
# Next write down (1)/alpha1 -  (3)/alpha3:
# (p1-c)/alpha1 = a alpha1 + b
# (p3-c)/alpha3 = a alpha3 + b
# (p1-c)/alpha1 - (p3-c)/alpha3 = a (alpha1-alpha3)
# a = ((p1-c)/alpha1 - (p3-c)/alpha3) / (alpha1-alpha3)
#
# Next we can work out b from p1:
# b = (p1 - a alpha1^2 - c) / alpha1

$c = $p2;
$a = (($p1-$c)/$alpha1 - ($p3-$c)/$alpha3) / ($alpha1 - $alpha3);
$b = ($p1 - $a*$alpha1*$alpha1 - $c) / $alpha1;

$err1 = ($a*$alpha1*$alpha1 + $b*$alpha1 + $c) - $p1;
$err2 = ($a*$alpha2*$alpha2 + $b*$alpha2 + $c) - $p2;
$err3 = ($a*$alpha3*$alpha3 + $b*$alpha3 + $c) - $p3;

$err1*$err1 < 0.0001 || die "optimize_alpha.pl: error in code?: error is $err1";
$err2*$err2 < 0.0001 || die "optimize_alpha.pl: error in code?: error is $err2";
$err3*$err3 < 0.0001 || die "optimize_alpha.pl: error in code?: error is $err3";

sub print_objf_change {
  $alpha = shift @_;
  $new_ppl = $alpha*$alpha*$a + $alpha*$b + $c;
  $old_ppl = $c;
  $change = $old_ppl - $new_ppl;
  print STDERR "Projected perplexity change from setting alpha=$alpha is $old_ppl->$new_ppl, reduction of $change\n"
}

if ($a <= 0.0) {
  if ($err1 < $err3) {  $alpha = 2*$alpha1; }
  else { $alpha = 2*$alpha3; }
  printf STDERR "optimize_alpha.pl: objective function is not convex; returning alpha=$alpha\n";
} else {
  $alpha = -0.5*$b/$a;
  if ($alpha < 2*$alpha1) {
    $newalpha=2*$alpha1;
    printf STDERR "optimize_alpha.pl: alpha=$alpha is too negative, limiting it to $newalpha\n";
    $alpha=$newalpha;
  } elsif ($alpha > 2*$alpha3) {
    $newalpha=2*$alpha3;
    printf STDERR "optimize_alpha.pl: alpha=$alpha is too positive, limiting it to $newalpha\n";
    $alpha=$newalpha;
  }
}
print_objf_change($alpha);
print $alpha . "\n";
