import itertools
import logging
import re
from functools import reduce
from operator import and_
from typing import List

"""
Pseudo code:

prediction, lemmatized_prediction, terms

Definitions:
    Exact Match Accuracy (EMA): Number of terms found in prediction / Number of terms
    Lemmatized Match Accuracy (LMA): Number of terms found in prediction or lemmatized_prediction / Number of terms

Critical thing to consider:
    Terms can have alternatives and only one of these alternatives must match, however multiple terms can not match the same part of the prediction.

    Ex: terms [[a, b], [a]]: prediction [a, b] --> EMA = 1
        terms [[a, b], [a]]: prediction [a, a] --> EMA = 1
        terms [[a, b], [a]]: prediction [b, b] --> EMA = 0.5

    For LMA we can match either in prediction or lemmatized_prediction, but we also need to block the matched part in the other domain
"""


def get_match_accuracies(predictions: List, terms: List[List[List]], tar_stanza):
    """
    Computes exact matches for both perfect matches and lemmatized matches

    Exact Match Accuracy: EMA
    Lemmatized Match Accuracy: LMA

    Args:
        terms (list): list of target terms  based on the tag "tgt" attribute in the structure
            Ex:   [[[s1_t1_a1], [s1_t2_a1, s1_t2_a2]], ...]
            Desc: sentence 1, term 1, alternative 1, sentence 1, term 2 with alternatives 1 and 2, ...
        predictions (List): list of predictions
        tar_stanza (stanza.Pipeline): stanza pipeline for lemmatization

    Returns:
        result (Dict):
            exact match correct (int): number of exact matches
            exact match wrong (int): number of non-exact matches
            exact match lemmatized correct (int): number of exact matches after lemmatization
            exact match lemmatized wrong (int): number of non-exact matches after lemmatization
            exact match accuracy (float): exact match accuracy
            lemmatized match accuracy (float): lemmatized match accuracy
    """
    assert len(terms) == len(
        predictions
    ), "Number of terms and predictions should match"
    predictions_long = "\n".join(predictions)

    """
    Note on stanza lemmatization:
    * It is significantly faster to lemmatize one long string than to lemmatize each sentence individually (factor ~3)
    * After lemmatization, every word contains information about start_char and end_char
    * If we lemmatize with one long sentence, these indices are global and not per sentence.
    * Therefore, we renormalize the indices with start_char of the first word
    * Note, that this means we ignore potential spaces at the beginning of the sentence
    * Stanza will cut out strings containing only spaces
    """

    log = logging.getLogger("CBS")
    log.info(f"Lemmatizing {len(predictions)} predictions")
    predictions_stanza = tar_stanza(predictions_long)
    log.info(f"Lemmatization done")

    # Normalize indices to sentence
    predictions_lemmatized = []
    for s in predictions_stanza.sentences:
        sentence_offset = s.words[0].start_char
        # sentence_offset = max([x for x in new_line_indices if x <= start])
        predictions_lemmatized.append(
            {
                "lemmatized_words": [w.lemma for w in s.words],
                "indices": [
                    (w.start_char - sentence_offset, w.end_char - sentence_offset)
                    for w in s.words
                ],
            }
        )

    total_correct_exact = 0
    total_correct_lemmatized = 0
    total_wrong_exact_and_lemmatized = 0

    # as stanza removes strings only containing spaces, we also remove them from predictions and terms
    stanza_removed_indices = [
        i for i, x in enumerate(predictions) if len(x.strip()) == 0
    ]
    total_wrong_exact_and_lemmatized += sum(
        [len(terms[i]) for i in stanza_removed_indices]
    )

    for i in sorted(stanza_removed_indices, reverse=True):
        del predictions[i]
        del terms[i]

    assert len(predictions_lemmatized) == len(predictions), (
        f"Number of lemmatized predictions ({len(predictions_lemmatized)}) "
        f"should match number of predictions ({len(predictions)})"
    )
    assert (
        len(terms) == len(predictions)
    ), f"Number of terms ({len(terms)}) should match number of predictions ({len(predictions)})"

    for pred, pred_l, terms_per_sentence in zip(
        predictions, predictions_lemmatized, terms
    ):
        correct_exact, correct_lemmatized, wrong_exact_and_lemmatized = count_matches(
            pred, pred_l, terms_per_sentence
        )
        total_correct_exact += correct_exact
        total_correct_lemmatized += correct_lemmatized
        total_wrong_exact_and_lemmatized += wrong_exact_and_lemmatized

    n_terms = sum([len(x) for x in terms])

    # calculate EMA
    exact_match_accuracy = total_correct_exact / n_terms

    # calculate LMA
    lemmatized_match_accuracy = total_correct_lemmatized / n_terms

    return_dict = {
        "exact match correct": total_correct_exact,
        "lemmatized match correct": total_correct_lemmatized,
        "wrong exact and lemmatized": total_wrong_exact_and_lemmatized,
        "exact match accuracy": exact_match_accuracy,
        "lemmatized match accuracy": lemmatized_match_accuracy,
    }
    return return_dict


def count_matches(pred, pred_l, terms_per_sentence):
    """
    Returns the number of exact matches, lemmatized matches and wrong matches for a prediction and terms.
    Args:
        pred (str): Prediction
        pred_l (dict): "lemmatized_words": lemmatized sentence, "indices": List of tuples of start and end indices
        terms_per_sentence: List of per term: List of possible alternatives
            Ex: [[term1, term2],[term3]] --> There is one term where two alternatives are accepted and another term
    """
    if len(terms_per_sentence) == 0:
        return 0, 0, 0

    if not isinstance(terms_per_sentence, list):
        raise ValueError("terms_per_sentence should be a list of lists")

    if len(terms_per_sentence) > 0 and not isinstance(terms_per_sentence[0], list):
        raise ValueError("terms_per_sentence should be a list of lists")

    """
    Consider things we don't want right before or after our constraint/terminology
    """
    dont_want = {
        "lower case letters": "a-z",
        "upper case letters": "A-Z",
        "digits": "0-9",
    }
    avoid_pattern = "".join(dont_want.values())

    correct_exact = maximally_match(pred, terms_per_sentence, avoid_pattern)

    """
    correct lemmatized: in the original implementation of WMT21, lemmatized matches are supposed to be
    counted additionally for matches that couldn't be matched exactly. However, this is flawed: with the
    setup of potential alternatives, the interactioon with lemmatized matches is not that simple.

    We count lemmatized matches as the maximum amount of matches that can be found in both pred and pred_l
    with the restriction, that if a subpart of pred_l is used to match, the corresponding part of pred
    can't be used as well and vice versa.

    This means we need a mapping from pred to pred_l and vice versa.
    Thankfully stanza provides start_char and end_char indices for each lemmatized word, allowing for the mapping
    """

    correct_lemmatized = maximally_match_with_lemmatized(
        pred, pred_l, terms_per_sentence, avoid_pattern
    )
    if correct_lemmatized < correct_exact:
        import logging

        log = logging.getLogger("CBS")
        log.warning(
            f"Rare case where lemmatized matches are lower than exact matches, likely due to combined lemmatization of two constraints, leading to a collision in lemmatized space \nfor {pred} and {terms_per_sentence} and {pred_l['lemmatized_words']}\nsetting correct lemmatized to correct exact"
        )
        correct_lemmatized = correct_exact

    wrong_exact_and_lemmatized = len(terms_per_sentence) - correct_lemmatized

    return correct_exact, correct_lemmatized, wrong_exact_and_lemmatized


def maximally_match(pred, terms_per_sentence, avoid_pattern):
    """
    Returns maximally matchable number of terms within pred
    Runs an exhaustive search over all possible combinations of spans and returns the number of terms that can be matched.
    Let n be the number of terms_per_sentence and m the number of alternatives for each term. The time complexity of this function is O(m^n).
    This could incur significant runtime, but in practice a match can be found quickly as the while loop is usually only run once or a few times.

    Args:
        pred (str): prediction string
        terms_per_sentence (List[List[str]]): List of terms, where inner-lists are alternatives
        avoid_pattern (str): pattern to avoid before/after terms
    """

    # Step 1: Find all possible spans/binary masks for each term
    pred_len = len(pred)
    all_spans = {}
    for idx, term in enumerate(terms_per_sentence):
        for alternative in term:
            regex_search = (
                f"(?<![{avoid_pattern}]){re.escape(alternative)}(?![{avoid_pattern}])"
            )
            starts_and_ends = [
                (m.start(), m.end())
                for m in re.finditer(regex_search, pred, re.IGNORECASE)
            ]
            for start, end in starts_and_ends:
                length = end - start
                length_binary_mask = 2**length - 1
                binary_mask = length_binary_mask << (pred_len - end)
                if idx not in all_spans:
                    all_spans[idx] = []
                all_spans[idx].append(binary_mask)

    # terms that never appear in the prediction are unmatcheable
    # therefore start not at len(terms) but at len(all_spans)
    n_terms = len(all_spans)

    # Step 2: Let n be the number of terms that we try to match
    # Try all combinations of n terms, if impossible, try n-1, n-2, ... 1
    while n_terms > 0:
        for combination in itertools.combinations(all_spans.keys(), n_terms):
            # all combinations of taking n_terms out of the total
            # all possible combinations between elements of lists:
            alternatives_combination = itertools.product(
                *[all_spans[c] for c in combination]
            )
            for a_comb in alternatives_combination:
                if len(a_comb) < 2:  # only 1 span, can't collide
                    return n_terms
                if reduce(and_, a_comb) == 0:  # if no collision
                    return n_terms
        n_terms -= 1

    assert n_terms == 0, "n_terms should be 0"
    return n_terms


def maximally_match_with_lemmatized(pred, pred_l, terms, avoid_pattern):
    """
    Returns maximally matchable number of terms within pred or pred_l
    Runs an exhaustive search over all possible combinations of spans and returns the number of terms that can be matched.
    Let n be the number of terms and m the number of alternatives for each term. The time complexity of this function is O(m^n).
    This could incur significant runtime, but in practice a match can be found quickly

    Args:
        pred (str): prediction string
        pred_l (dict): "lemmatized_words": lemmatized sentence, "indices": List of tuples of start and end indices
        terms (List[List[str]]): List of terms, where inner-lists are alternatives
        avoid_pattern (str): pattern to avoid before/after terms
    """

    # Step 1: collect information on what stanza lemmatizer skips
    # a) indices ignore preceding spaces b) indices reduce multiple spaces to one

    n_leading_spaces = len(pred) - len(pred.lstrip())

    extra_spaces = [
        (x.start() + 1, len(x.group()) - 1)
        for x in re.finditer("  ( )*", pred)
        if x.start() != 0
    ]
    # x.start() + 1 : add the index of the first space that is too much

    lemmatized_string = " " * n_leading_spaces
    mapping = []
    pos_counter_lemm = n_leading_spaces
    pos_counter_orig = n_leading_spaces
    lemmatized_same_counter = 0

    # Step 2: build lemmatized sentence and mapping
    for word, (start, end) in zip(pred_l["lemmatized_words"], pred_l["indices"]):
        # very rarely, for chinese, stanza will return a lemmatized_word "None"
        if word is None:
            word = pred[start:end]

        lemmatized_string += word + " "

        # offset not captured in start, end
        delete_index = -1
        for es_indx, (ind, length) in enumerate(extra_spaces):
            # if where we want to continue, there is an extra space, we need to consider it
            # we check after the loop that all extra spaces have been found
            if ind == pos_counter_orig:
                pos_counter_orig += length
                delete_index = es_indx
                break
        if delete_index != -1:
            extra_spaces.pop(delete_index)

        mapping.append(
            {
                "lemmatized_space": (pos_counter_lemm, pos_counter_lemm + len(word)),
                "original_space": (
                    pos_counter_orig,
                    pos_counter_orig + end - start,
                ),
            }
        )

        # statistics
        if (
            word.lower()
            == pred[pos_counter_orig : pos_counter_orig + end - start].lower()
        ):
            lemmatized_same_counter += 1

        pos_counter_lemm += len(word) + 1
        pos_counter_orig += end - start + 1

    assert len(extra_spaces) == 0, f"Extra spaces should be empty: {extra_spaces}"
    # Statistics check
    percentage = lemmatized_same_counter / len(pred_l["lemmatized_words"]) * 100
    if percentage < 20:
        log = logging.getLogger("CBS")
        log.info(
            f"Lemmatized overlap {percentage:.2f}% (<20%): {pred} -> {pred_l['lemmatized_words']}"
        )

    # Step 3: create bidirectional dense mappings
    pred_check = pred
    pred_lemm_check = lemmatized_string
    list_mapping_to_lemmatized = [-1] * len(pred)
    list_mapping_to_original = [-1] * len(lemmatized_string)
    for m in mapping:
        for i in range(m["original_space"][0], m["original_space"][1]):
            list_mapping_to_lemmatized[i] = m[
                "lemmatized_space"
            ]  # for every index in original assign tuple of span in lemmatized
        for i in range(m["lemmatized_space"][0], m["lemmatized_space"][1]):
            list_mapping_to_original[i] = m[
                "original_space"
            ]  # for every index in lemmatized assign tuple of span in original

        # sanity check
        pred_check = (
            pred_check[: m["original_space"][0]]
            + " " * (m["original_space"][1] - m["original_space"][0])
            + pred_check[m["original_space"][1] :]
        )
        pred_lemm_check = (
            pred_lemm_check[: m["lemmatized_space"][0]]
            + " " * (m["lemmatized_space"][1] - m["lemmatized_space"][0])
            + pred_lemm_check[m["lemmatized_space"][1] :]
        )

    assert (
        pred_check.strip() == ""
    ), f"Mapping should have removed all characters: {pred_check}\n for {pred}"
    assert (
        pred_lemm_check.strip() == ""
    ), f"Mapping should have removed all characters: {pred_lemm_check}\n for {pred}"

    # Step 4: Find all possible spans for each term
    pred_len = len(pred)
    pred_lem_len = len(lemmatized_string)
    all_spans = {}

    for idx, t in enumerate(terms):
        for alternative in t:
            regex_search = (
                f"(?<![{avoid_pattern}]){re.escape(alternative)}(?![{avoid_pattern}])"
            )

            starts_and_ends = [
                (m.start(), m.end())
                for m in re.finditer(regex_search, pred, re.IGNORECASE)
            ]
            starts_and_ends_lemm = [
                (m.start(), m.end())
                for m in re.finditer(regex_search, lemmatized_string, re.IGNORECASE)
            ]

            for start, end in starts_and_ends:
                lemmatized_mark = (
                    set()
                )  # set of spans in lemmatized that need to be locked
                for term_pos_idx in range(start, end):
                    if list_mapping_to_lemmatized[term_pos_idx] != -1:
                        lemmatized_mark.add(list_mapping_to_lemmatized[term_pos_idx])

                binary_mask_lemmatized = 0
                for lemmatized_span in list(lemmatized_mark):
                    length = lemmatized_span[1] - lemmatized_span[0]
                    length_binary_mask = 2**length - 1
                    binary_mask = length_binary_mask << (
                        pred_lem_len - lemmatized_span[1]
                    )
                    binary_mask_lemmatized |= binary_mask

                length = end - start
                length_binary_mask = 2**length - 1
                binary_mask = length_binary_mask << (pred_len - end)
                if idx not in all_spans:
                    all_spans[idx] = set()

                # combine binary masks in orig space and lemm space
                moved_binary_mask_lemmatized = binary_mask_lemmatized << pred_len + 1
                assert (
                    moved_binary_mask_lemmatized & binary_mask == 0
                ), "Binary masks should not collide"
                total_mask = moved_binary_mask_lemmatized | binary_mask
                all_spans[idx].add(total_mask)

            # do the reverse for matches in lemmatized
            for start, end in starts_and_ends_lemm:
                original_mark = set()  # set of spans in original that need to be locked
                for term_pos_idx in range(start, end):
                    if list_mapping_to_original[term_pos_idx] != -1:
                        original_mark.add(list_mapping_to_original[term_pos_idx])

                binary_mask_original = 0
                for original_span in list(original_mark):
                    length = original_span[1] - original_span[0]
                    length_binary_mask = 2**length - 1
                    binary_mask = length_binary_mask << (pred_len - original_span[1])
                    binary_mask_original |= binary_mask

                length = end - start
                length_binary_mask = 2**length - 1
                binary_mask = length_binary_mask << (pred_lem_len - end)
                if idx not in all_spans:
                    all_spans[idx] = set()

                # combine binary masks in orig space and lemm space
                moved_binary_mask_lemmatized = binary_mask << pred_len + 1
                assert (
                    moved_binary_mask_lemmatized & binary_mask_original == 0
                ), "Binary masks should not collide"
                total_mask = moved_binary_mask_lemmatized | binary_mask_original
                all_spans[idx].add(total_mask)

    # Step 5: Let n be the number of terms that we try to match
    # Try all combinations of n terms, if impossible, try n-1, n-2, ... 1
    n_terms = len(all_spans)
    while n_terms > 0:
        for combination in itertools.combinations(all_spans.keys(), n_terms):
            # all combinations of terms sorted from largest to smallest
            alternatives_combination = itertools.product(
                *[all_spans[c] for c in combination]
            )
            for a_comb in alternatives_combination:
                if len(a_comb) < 2:  # only 1 span, can't collide
                    return n_terms
                if reduce(and_, a_comb) == 0:  # if no collision
                    return n_terms
        n_terms -= 1

    assert n_terms == 0, "n_terms should be 0"
    return n_terms