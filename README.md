# Match Accuracy

This is a rewritten form of the original [WMT21 Evaluation Script](https://github.com/mahfuzibnalam/terminology_evaluation/blob/040906f012d25539e0ec8d9c9ab0b61cdfc0a83e/evaluate_term_wmt.py#L87) which had a few non-ideal behaviors.

It builds the basics of the reported scores in [CacadeBeamSearch](https://arxiv.org/pdf/2305.14538)


Explanation on `def maximally_match` and `def maximally_match_with_lemmatized` in [match_accuracy.py](./match_accuracy.py)

## Exact Match Accuracy

Given

* `Input`: one single prediction sentence
* `Terms`: `List` of `List`s

`[[t1_a1, t1_a2], [t2_a1], ...]`

<pre>
term 1 alternative 1      | term 1 alternative 2 || term 2 alternative 1 || ... </

<pre>
appearance 1 appearance 2 | appearance 1         ||

-------------------------------------------------
 One of these appears, we have a match for term 1
</pre>

</pre>


* We create **binary_masks** for every appearance

```
n_terms = len(terms with at least one valid span)
while n_terms > 0
	Step 1: all combinations to pick n_terms out of all terms
	Step 2: for every term combination,
		create all binary mask combinations (cartesian product) for all chosen terms
    Step 3: for every binary mask combination,
		if logic and of all binary mask is all zero → return n_terms
    Step 4: n_terms -= 1
```

Advantages:

* can deal with alternatives
* can deal with same constraint appearing in multiple terms
* implementation with binary masks is efficient
* while runtime search could be exhaustive, in practice we can almost always match all the terms after one or just a few iterations
* Adversarial attack: lots of terms that do not appear in pred: their binary_masks won’t be added and n_terms will be reduced to start at number of terms that actually do have a match in pred: for 100_000 terms that don't appear computes in < 1 sec


## Lematized Match Accuracy

* create mapping between lemmatized and original space
* binary masks of term found in either space also block the range in the other space
* same procedure