# Informative Translation

This codebase implements the informative speaker model over the top of a deep neural translation model. It's the code corresponding to this paper: https://arxiv.org/abs/1902.09514

## To run ##

Clone the repo, using git-lfs for the pretrained models.

Run: ipython scripts/examples.py

This file contains each of the models described in the paper.

## What it does ##

The models in the paper are of the form P(u|w,c) or P(w|u,c), for a source (here English) word or sentence w, a target (here German) word or sentence u, and a sequence of previously generated target language words c (i.e. a partial translation).

In the code, each model m is a class with methods, e.g.:

* m.forward(source_sentence,sequence): returns distribution over next word/sentence (depending on model)
* m.likelihood(source_sentence,sequence,output): returns probability of a given word/sentence being produced by m.forward

The models are build in a compositional way, with two key operations, Unfold and Pragmatic. These are classes which take a model object and themselves constitute a model object. E.g. Unfold(underlying_model=s0_word,beam_width=2) takes a word level model and gives back a sentence level one. Pragmatic takes a speaker and gives a pragmatic speaker (e.g. s0 -> s1). 

S1SENTIP is an unfolded, pragmatified speaker and S1SENTGP is a pragmatified unfolded speaker (unfold and pragmatic are operations which don't commute).

Note: As you'll see if you run examples.py, the more complex models are ludicrously slow. But this isn't (for the most part) a function of my algorithm, so much as the very inefficient way I implemented the wrapper around the PyTorch Fairseq models. My goal was a proof of concept, not a practical system.

## What is where ##

The various models are each implemented with memoization and various bells and whistles in translators/translators.py. The pragmatics (i.e. the RSA model) is in bayesian_agents/bayesian_pragmatics.
