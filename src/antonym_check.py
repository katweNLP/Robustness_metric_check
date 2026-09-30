"""
Antonym-aware verb similarity for RHI*.

Two-layer check to prevent antonym pairs from passing semantic similarity:
  Layer 1: WordNet antonym lookup (fast, no model, covers ~60% of cases)
  Layer 2: NLI entailment check (accurate, uses cross-encoder, covers rest)

Usage:
    from src.antonym_check import is_antonym_pair, semantic_valid_polarity_aware

    # Quick check
    is_antonym_pair("arrested", "released")  # True

    # Full similarity with polarity awareness
    semantic_valid_polarity_aware(triple1, triple2, verb_embs)  # False if antonyms
"""

import nltk
from functools import lru_cache

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet as wn

_nli_model = None


@lru_cache(maxsize=2048)
def _get_wordnet_antonyms(word):
    antonyms = set()
    for syn in wn.synsets(word, pos=wn.VERB):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name().replace('_', ' '))
    return frozenset(antonyms)


@lru_cache(maxsize=2048)
def _get_wordnet_lemmas(word):
    lemmas = set()
    for syn in wn.synsets(word, pos=wn.VERB):
        for l in syn.lemmas():
            lemmas.add(l.name().replace('_', ' '))
    return frozenset(lemmas)


def _wordnet_antonym_check(verb1, verb2):
    v1 = verb1.lower().strip()
    v2 = verb2.lower().strip()

    ants1 = _get_wordnet_antonyms(v1)
    lemmas2 = _get_wordnet_lemmas(v2)
    if ants1 & lemmas2:
        return True

    ants2 = _get_wordnet_antonyms(v2)
    lemmas1 = _get_wordnet_lemmas(v1)
    if ants2 & lemmas1:
        return True

    return False


def _nli_contradiction_check(verb1, verb2, subject="Someone", obj="something"):
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    import numpy as np

    premise = f"{subject} {verb1} {obj}."
    hypothesis = f"{subject} {verb2} {obj}."

    scores = _nli_model.predict([(premise, hypothesis)])
    scores = np.atleast_1d(np.squeeze(np.array(scores)))

    if scores.shape[-1] == 3:
        probs = np.exp(scores) / np.exp(scores).sum()
        contradiction_prob = float(probs[0])
        entailment_prob = float(probs[1])
        return contradiction_prob > 0.5 and entailment_prob < 0.3

    return False


def is_antonym_pair(verb1, verb2, use_nli=True, subject=None, obj=None):
    if _wordnet_antonym_check(verb1, verb2):
        return True

    if use_nli:
        s = subject or "Someone"
        o = obj or "something"
        return _nli_contradiction_check(verb1, verb2, s, o)

    return False


def semantic_valid_polarity_aware(tup1, tup2, verb_embs=None, cosine_threshold=0.3, use_nli=True):
    s1, v1, o1 = tup1[:3]
    s2, v2, o2 = tup2[:3]

    if s1.lower() == s2.lower() and v1.lower() == v2.lower() and o1.lower() == o2.lower():
        return True

    if is_antonym_pair(v1, v2, use_nli=use_nli, subject=s1, obj=o1):
        return False

    if verb_embs is not None and v1 in verb_embs and v2 in verb_embs:
        from sentence_transformers import util
        sim = float(util.cos_sim(verb_embs[v1], verb_embs[v2]))
        return sim >= cosine_threshold

    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
    e1 = model.encode(v1)
    e2 = model.encode(v2)
    sim = float(util.cos_sim(e1, e2))
    return sim >= cosine_threshold
