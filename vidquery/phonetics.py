"""Soft phonetic matching for spoken terms that ASR commonly mishears.

Whisper ``tiny`` transcribes technical vocabulary inconsistently ("binary
search tree" -> "Vandri search tree", "Neo4j" -> "4.js").  Exact token matching
then misses the moment entirely.  This module provides a deliberately small,
dependency-free fallback that is consulted only for query terms that did *not*
match exactly:

* ``phonetic_key`` reduces a word to a consonant-class skeleton (voicing and
  nasal place are merged because those are the confusions ASR makes most);
* ``phonetic_similarity`` is a normalised edit similarity between two keys;
* ``KNOWN_ASR_VARIANTS`` lists curated mis-transcriptions of common technical
  terms that no generic key can recover (digits, split words).

Matches are reported with their own ``phonetic_transcript`` provenance so the
reliability layer can weight them separately from exact transcript hits.  The
feature is off unless ``ENABLE_PHONETIC_MATCH=true``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_PHONETIC_THRESHOLD = 0.75
MINIMUM_TERM_LENGTH = 5
MINIMUM_KEY_LENGTH = 3

_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

# Ordered digraph rewrites applied before single-letter classes.
_DIGRAPHS = (
    ("tch", "X"),
    ("sch", "SK"),
    ("ph", "P"),
    ("gh", ""),
    ("ck", "K"),
    ("sh", "X"),
    ("ch", "X"),
    ("th", "0"),
    ("qu", "KW"),
    ("dg", "J"),
    ("wr", "R"),
    ("kn", "N"),
)
_SOFT_C = re.compile(r"c(?=[eiy])")
_SOFT_G = re.compile(r"g(?=[eiy])")
_CLASSES = {
    "b": "P",
    "p": "P",
    "f": "P",
    "v": "P",
    "d": "T",
    "t": "T",
    "c": "K",
    "g": "K",
    "k": "K",
    "q": "K",
    "s": "S",
    "z": "S",
    "x": "KS",
    "j": "J",
    "m": "N",
    "n": "N",
    "l": "L",
    "r": "R",
}

# Canonical technical term -> token sequences ASR is known to produce for it.
# Keys and variants are lower-case word sequences after tokenisation.
KNOWN_ASR_VARIANTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "neo4j": (("4", "js"), ("4js",), ("neo", "4j"), ("neo", "for", "j"), ("neo", "4", "j"),
              ("new", "4j"), ("neo", "forge")),
    "kubernetes": (("cooper", "netes"), ("kuber", "netes"), ("cuber", "nettis"),
                   ("kubernetis",)),
    "postgresql": (("postgres",), ("post", "gres"), ("postgres", "sql")),
    "sql": (("sequel",),),
    "nginx": (("engine", "x"),),
    "github": (("git", "hub"),),
    "pytorch": (("pie", "torch"), ("py", "torch")),
    "numpy": (("num", "pie"), ("numb", "pie")),
    "django": (("jango",),),
    "cuda": (("kuda",), ("cooda",)),
    "json": (("jason",),),
    "yaml": (("yamel",), ("yammel",)),
    "gpu": (("g", "p", "u"),),
    "cpu": (("c", "p", "u"),),
    "api": (("a", "p", "i"),),
    "sqlite": (("sequel", "light"), ("sql", "light"), ("sequelite",)),
    "docker": (("doctor",),),
}


def _expand_digits(word: str) -> str:
    return "".join(_DIGIT_WORDS.get(character, character) for character in word)


def phonetic_key(word: str) -> str:
    """Return a consonant-class skeleton such as ``binary`` -> ``PNR``."""

    text = _expand_digits(re.sub(r"[^a-z0-9]", "", word.lower()))
    if not text:
        return ""
    text = _SOFT_C.sub("s", text)
    text = _SOFT_G.sub("j", text)
    for source, target in _DIGRAPHS:
        text = text.replace(source, target)
    key: list[str] = []
    for index, character in enumerate(text):
        if character.isupper() or character == "0":
            code = character
        elif character in "aeiou":
            code = "A" if index == 0 else ""
        elif character in "hwy":
            code = ""
        else:
            code = _CLASSES.get(character, "")
        for symbol in code:
            if not key or key[-1] != symbol:
                key.append(symbol)
    return "".join(key)


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def phonetic_similarity(left: str, right: str) -> float:
    """Normalised edit similarity of two phonetic keys, in [0, 1]."""

    first, second = phonetic_key(left), phonetic_key(right)
    if not first or not second:
        return 0.0
    return 1.0 - _edit_distance(first, second) / max(len(first), len(second))


@dataclass(frozen=True, slots=True)
class PhoneticMatch:
    term: str
    surface_tokens: tuple[str, ...]
    similarity: float
    method: str  # "known_variant" or "phonetic_key"


def _known_variant_match(term: str, tokens: list[str]) -> PhoneticMatch | None:
    for variant in KNOWN_ASR_VARIANTS.get(term, ()):
        width = len(variant)
        for start in range(len(tokens) - width + 1):
            if tuple(tokens[start : start + width]) == variant:
                return PhoneticMatch(term, variant, 1.0, "known_variant")
    return None


def find_phonetic_matches(
    terms: list[str],
    tokens: list[str],
    *,
    threshold: float = DEFAULT_PHONETIC_THRESHOLD,
) -> dict[str, PhoneticMatch]:
    """Match query ``terms`` that are absent from ``tokens`` by sound.

    Only terms with no exact token hit are considered.  Generic key matching
    requires the term to be at least ``MINIMUM_TERM_LENGTH`` characters, the
    keys to share their first sound class, and similarity >= ``threshold``.
    Single tokens and concatenations of two adjacent tokens are compared, so
    split words ("vandri" / "van dree") can still match.
    """

    if not 0 < threshold <= 1:
        raise ValueError("phonetic threshold must be in (0, 1]")
    present = set(tokens)
    matches: dict[str, PhoneticMatch] = {}
    for term in dict.fromkeys(item.lower() for item in terms if item):
        if term in present:
            continue
        known = _known_variant_match(term, tokens)
        if known is not None:
            matches[term] = known
            continue
        if len(term) < MINIMUM_TERM_LENGTH:
            continue
        term_key = phonetic_key(term)
        if len(term_key) < MINIMUM_KEY_LENGTH:
            continue
        best: PhoneticMatch | None = None
        candidates: list[tuple[str, ...]] = [(token,) for token in tokens]
        candidates.extend(
            (tokens[index], tokens[index + 1]) for index in range(len(tokens) - 1)
        )
        for candidate in candidates:
            joined = "".join(candidate)
            candidate_key = phonetic_key(joined)
            if len(candidate_key) < MINIMUM_KEY_LENGTH or candidate_key[0] != term_key[0]:
                continue
            similarity = phonetic_similarity(term, joined)
            if similarity >= threshold and (best is None or similarity > best.similarity):
                best = PhoneticMatch(term, candidate, round(similarity, 4), "phonetic_key")
        if best is not None:
            matches[term] = best
    return matches
