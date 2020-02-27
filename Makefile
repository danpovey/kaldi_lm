
# -std=c++11 is not because it really requires c++11 support,
# it's to work around a gcc 4.8 issue whereby unordered_map
# requires c++11 support.
CXXFLAGS += -g -std=c++11
COMPILER = $(shell $(CXX) -v 2>&1 )
ifeq ($(findstring clang,$(COMPILER)),clang)
    LDFLAGS += -stdlib=libc++
endif

PROGRAMS = get_raw_ngrams uniq_to_ngrams merge_ngrams discount_ngrams interpolate_ngrams \
   compute_perplexity prune_ngrams

all: $(PROGRAMS)

clean:
	rm $(PROGRAMS)

merge_ngrams_online: merge_ngrams_online.cc

prune_ngrams: prune_ngrams.cc

discount_ngrams: discount_ngrams.cc


