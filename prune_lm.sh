#!/bin/bash

# This script prunes an LM that has been trained with
# train_lm.sh.
# It uses an entropy threshold (typically between about 1.0 and 4.0,
# although it could be larger); it differs from the Stolcke-pruning
# type of threshold in that it is multiplied by the number of
# training counts.
# Note:

write_arpa=false
cleanup=false

# Parse options.
for n in `seq 3`; do
  if [ "$1" == "--arpa" ]; then
    write_arpa=true
    shift
  fi
  if [ "$1" == "--cleanup" ]; then
    cleanup=true
    shift
  fi
done

if [ $# != 2 ]; then
  echo "Usage: prune_lm.sh [options] threshold train_subdir"
  echo " e.g. prune_lm.sh 4.0 my/dir/3gram-mincount/"
  echo "Options:"
  echo "  --arpa   : write arpa file as well as N-gram format"
  echo "             Note: output filenames are determined by type of LM."
  echo "  --cleanup : clean up temporary files rpa   : write arpa file as well as N-gram format"
fi

thresh=$1
subdir=$2
dir=$subdir/..

[ ! -f $subdir/ngrams_disc.gz -o ! -f $dir/word_map ] && \
 ( echo Expecting files $subdir/ngrams_disc and $dir/word_map to exist;
   echo E.g. see egs/wsj/s3/local/wsj_train_lm.sh for examples. ) && exit 1;

# Check the path.
! merge_ngrams </dev/null >&/dev/null  && echo You need to have kaldi_lm on your path \
   && exit 1;



if ! gunzip -c $subdir/ngrams_disc_pr$thresh.gz >&/dev/null; then
  echo "Pruning N-grams"
  gunzip -c $subdir/ngrams_disc.gz | \
    prune_ngrams $thresh | sort | merge_ngrams | \
    sort | gzip -c >$subdir/ngrams_disc_pr$thresh.gz
else
  echo "Not creating discounted N-gram file $subdir/ngrams_disc_pr$thresh.gz" 
  echo "since it already exists."
fi

echo "Computing pruned perplexity"

gunzip -c $subdir/ngrams_disc_pr$thresh.gz | \
  interpolate_ngrams $dir/wordlist.mapped 0.5 | sort | \
  sort -m <(gunzip -c $subdir/heldout_ngrams.gz) - | compute_perplexity 2>&1 | \
    tee $subdir/perplexity.pruned.$thresh &

ngrams=`gunzip -c $subdir/ngrams_disc_pr$thresh.gz | grep -v '*' | wc -l`
echo "After pruning, number of N-grams is $ngrams"

if $write_arpa; then
  echo "Building ARPA LM (perplexity computation is in background)"
  mkdir -p $subdir/tmpdir
  gunzip -c $subdir/ngrams_disc_pr$thresh.gz | \
    interpolate_ngrams --arpa $dir/wordlist.mapped 0.5 | \
    sort | finalize_arpa.pl $subdir/tmpdir | \
    map_words_in_arpa.pl $dir/word_map | \
    gzip -c > $subdir/lm_pr$thresh.gz
  echo ARPA output is in $subdir/lm_pr$thresh.gz
fi

wait
echo Done pruning LM with threshold $thresh


