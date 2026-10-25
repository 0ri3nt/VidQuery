from __future__ import annotations

import pytest

from vidquery.action_lexicon import ACTION_TEMPLATES, match_actions, verb_forms
from vidquery.ava_labels import AVA_V22_ACTIONS
from vidquery.hypotheses import generate_hypotheses
from vidquery.search import StructuredQueryParser

# One supported phrasing per AVA label, using the label's own wording.
SUPPORTED_FORMS = {
    1: "someone bending at the waist",
    2: "a baby crawling",
    3: "a person crouching",
    4: "people dancing",
    5: "someone falling down",
    6: "a person getting up",
    7: "someone jumping",
    8: "a person lying down",
    9: "someone doing martial arts",
    10: "a person running",
    11: "a person sitting",
    12: "a person standing",
    13: "someone swimming",
    14: "a person walking",
    15: "someone answering the phone",
    16: "a person brushing their teeth",
    17: "a person carrying a box",
    18: "someone catching a ball",
    19: "someone chopping vegetables",
    20: "a person climbing a mountain",
    21: "people clinking glasses",
    22: "someone closing the door",
    23: "a person cooking",
    24: "someone cutting bread",
    25: "a person digging",
    26: "someone putting on clothes",
    27: "a person drinking",
    28: "someone driving a car",
    29: "a person eating",
    30: "someone entering the room",
    31: "a person exiting the building",
    32: "someone extracting a nail",
    33: "a person fishing",
    34: "someone hitting an object",
    35: "someone kicking a ball",
    36: "a person picking up a cup",
    37: "someone listening to music",
    38: "someone opening a window",
    39: "a person painting",
    40: "people playing a board game",
    41: "someone playing a musical instrument",
    42: "a person playing with pets",
    43: "someone pointing at the screen",
    44: "a person pressing a button",
    45: "someone pulling a cart",
    46: "a person pushing a cart",
    47: "someone putting the bag down",
    48: "a person reading",
    49: "someone riding a horse",
    50: "a person rowing a boat",
    51: "someone sailing a boat",
    52: "someone shooting",
    53: "a person shoveling",
    54: "someone smoking",
    55: "a person stirring",
    56: "someone taking a photo",
    57: "a person texting on a phone",
    58: "someone throwing a ball",
    59: "a person touching an object",
    60: "someone turning a screwdriver",
    61: "a person watching tv",
    62: "someone working on a computer",
    63: "a person writing",
    64: "people fighting",
    65: "someone giving a cup to another person",
    66: "a person grabbing someone",
    67: "someone clapping",
    68: "two people shaking hands",
    69: "a person waving",
    70: "two people hugging",
    71: "someone kicking a person",
    72: "two people kissing",
    73: "someone lifting another person",
    74: "a person listening to someone",
    75: "an adult playing with kids",
    76: "someone pushing another person",
    77: "a person singing to someone",
    78: "someone taking a cup from her",
    79: "two people talking",
    80: "someone watching another person",
}


def _actions(query: str) -> list[str]:
    return StructuredQueryParser().parse(query).actions


def test_every_label_has_templates_and_a_supported_form():
    assert set(ACTION_TEMPLATES) == set(AVA_V22_ACTIONS) == set(SUPPORTED_FORMS)


@pytest.mark.parametrize("label_id", sorted(SUPPORTED_FORMS))
def test_each_ava_label_resolves_from_its_supported_form(label_id):
    assert _actions(SUPPORTED_FORMS[label_id]) == [AVA_V22_ACTIONS[label_id]]


@pytest.mark.parametrize(
    ("query", "label"),
    [
        ("someone kneeling down", "crouch/kneel"),
        ("a person lying down", "lie/sleep"),
        ("a person running", "run/jog"),
        ("someone answering the phone", "answer phone"),
        ("someone working on a computer", "work on computer"),
        ("someone bending over", "bend/bow"),
        ("a person getting up", "get up"),
        ("someone taking a photo", "take photo"),
        ("a person grabbing someone", "grab person"),
        ("someone lifting another person", "lift person"),
    ],
)
def test_known_parser_failures_resolve(query, label):
    plan = StructuredQueryParser().parse(query)
    assert plan.actions == [label]
    assert plan.intent == "action_search"
    assert plan.spoken_terms == []


@pytest.mark.parametrize(
    "query",
    ["she kneels", "he knelt", "they kneel", "a man kneeling", "people kneeling down"],
)
def test_verb_inflections_and_plural_subjects(query):
    assert _actions(query) == ["crouch/kneel"]


def test_irregular_and_doubled_verb_forms():
    assert {"ran", "running", "runs"} <= set(verb_forms("run"))
    assert {"lying", "lain"} <= set(verb_forms("lie"))
    assert "shot" not in verb_forms("shoot") and "written" not in verb_forms("write")


def test_object_and_person_fillers_choose_the_paired_label():
    assert _actions("someone kicking something") == ["kick object"]
    assert _actions("someone kicking a person") == ["kick person"]
    assert _actions("a person watching something") == ["watch"]
    assert _actions("a person watching another person") == ["watch person"]
    assert _actions("someone listening to music") == ["listen"]
    # Bare forms keep the legacy person-variant mapping.
    assert _actions("someone watching") == ["watch person"]
    assert _actions("someone listening") == ["listen to"]


def test_filler_words_never_become_spoken_terms():
    for query in (
        "a person holding something",
        "someone touching an object",
        "someone doing martial arts",
        "a person pushing something",
    ):
        plan = StructuredQueryParser().parse(query)
        assert plan.spoken_terms == [], query
        assert plan.actions, query


def test_unknown_actions_are_not_forced_into_a_label():
    plan = StructuredQueryParser().parse("someone juggling")
    assert plan.actions == []
    assert plan.spoken_terms == ["juggling"]


def test_multiple_actions_and_multimodal_queries_keep_other_evidence():
    assert _actions("a person sitting and reading") == ["sit", "read"]
    plan = StructuredQueryParser().parse("a person reading a book")
    assert plan.actions == ["read"] and "book" in plan.visual_entities
    # The phone in "answer phone" is part of the action, not a separate requirement.
    phone_plan = StructuredQueryParser().parse("someone answering the phone")
    assert "cell phone" not in phone_plan.visual_entities
    speech = StructuredQueryParser().parse("when does he talk about running")
    assert speech.actions == [] and "running" in speech.spoken_terms


def test_relation_queries_are_unchanged():
    plan = StructuredQueryParser().parse("person talking to another person")
    assert plan.intent == "relationship_search"
    assert [(item.subject, item.predicate, item.object) for item in plan.relationship_tuples] == [
        ("person", "talk_to", "person")
    ]


def test_bare_ambiguous_action_still_offers_hypotheses():
    plan = StructuredQueryParser().parse("find drive")
    assert plan.actions == ["drive"]
    assert len(generate_hypotheses("find drive", plan)) > 1


@pytest.mark.parametrize(
    "query",
    [
        "run the code",
        "when does she run the program",
        "what is the running time of the algorithm",
        "throw an exception",
        "catch the error",
        "the pointer points to the next node",
        "walk through the algorithm",
        "jump to the next slide",
        "push the changes to github",
        "open a pull request",
        "save it to google drive",
        "extract features from the image",
        "give an example of recursion",
        "understand recursion",
        "the main point of the lecture",
        "the text on the slide",
        "radio waves",
        "person close to the laptop",
        "the open door",
        "whose turn is it",
        "he tells a lie",
        "a wrist watch",
        "read the file",
        "the lecture touches on graphs",
        "open source software",
        "cache hit rate",
    ],
)
def test_lecture_idioms_and_similar_words_are_not_actions(query):
    assert _actions(query) == [], query


def test_matches_report_spans_and_templates():
    found = match_actions("someone answering the phone")
    assert len(found) == 1
    assert found[0].label == "answer phone"
    assert "someone answering the phone"[found[0].start : found[0].end] == "answering the phone"
