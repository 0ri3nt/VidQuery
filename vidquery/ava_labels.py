"""The complete official AVA v2.2 action-label map.

``AVA_V22_LABELS`` preserves the exact names and label types published in the
official pbtxt file. ``AVA_V22_ACTIONS`` remains the compact, search-friendly
name map used by the existing VidQuery contracts.

Source: https://research.google.com/ava/download/ava_action_list_v2.2.pbtxt
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AVAActionLabel:
    label_id: int
    official_name: str
    label_type: str
    canonical_name: str

AVA_V22_ACTIONS = {
    1: "bend/bow",
    2: "crawl",
    3: "crouch/kneel",
    4: "dance",
    5: "fall down",
    6: "get up",
    7: "jump/leap",
    8: "lie/sleep",
    9: "martial art",
    10: "run/jog",
    11: "sit",
    12: "stand",
    13: "swim",
    14: "walk",
    15: "answer phone",
    16: "brush teeth",
    17: "carry/hold",
    18: "catch",
    19: "chop",
    20: "climb",
    21: "clink glass",
    22: "close",
    23: "cook",
    24: "cut",
    25: "dig",
    26: "dress/put on clothing",
    27: "drink",
    28: "drive",
    29: "eat",
    30: "enter",
    31: "exit",
    32: "extract",
    33: "fishing",
    34: "hit object",
    35: "kick object",
    36: "lift/pick up",
    37: "listen",
    38: "open",
    39: "paint",
    40: "play board game",
    41: "play musical instrument",
    42: "play with pets",
    43: "point to",
    44: "press",
    45: "pull",
    46: "push",
    47: "put down",
    48: "read",
    49: "ride",
    50: "row boat",
    51: "sail boat",
    52: "shoot",
    53: "shovel",
    54: "smoke",
    55: "stir",
    56: "take photo",
    57: "text on phone",
    58: "throw",
    59: "touch object",
    60: "turn",
    61: "watch",
    62: "work on computer",
    63: "write",
    64: "fight/hit person",
    65: "give/serve",
    66: "grab person",
    67: "hand clap",
    68: "hand shake",
    69: "hand wave",
    70: "hug",
    71: "kick person",
    72: "kiss",
    73: "lift person",
    74: "listen to",
    75: "play with kids",
    76: "push person",
    77: "sing to",
    78: "take from person",
    79: "talk to",
    80: "watch person",
}


_OFFICIAL_NAMES = (
    "bend/bow (at the waist)",
    "crawl",
    "crouch/kneel",
    "dance",
    "fall down",
    "get up",
    "jump/leap",
    "lie/sleep",
    "martial art",
    "run/jog",
    "sit",
    "stand",
    "swim",
    "walk",
    "answer phone",
    "brush teeth",
    "carry/hold (an object)",
    "catch (an object)",
    "chop",
    "climb (e.g., a mountain)",
    "clink glass",
    "close (e.g., a door, a box)",
    "cook",
    "cut",
    "dig",
    "dress/put on clothing",
    "drink",
    "drive (e.g., a car, a truck)",
    "eat",
    "enter",
    "exit",
    "extract",
    "fishing",
    "hit (an object)",
    "kick (an object)",
    "lift/pick up",
    "listen (e.g., to music)",
    "open (e.g., a window, a car door)",
    "paint",
    "play board game",
    "play musical instrument",
    "play with pets",
    "point to (an object)",
    "press",
    "pull (an object)",
    "push (an object)",
    "put down",
    "read",
    "ride (e.g., a bike, a car, a horse)",
    "row boat",
    "sail boat",
    "shoot",
    "shovel",
    "smoke",
    "stir",
    "take a photo",
    "text on/look at a cellphone",
    "throw",
    "touch (an object)",
    "turn (e.g., a screwdriver)",
    "watch (e.g., TV)",
    "work on a computer",
    "write",
    "fight/hit (a person)",
    "give/serve (an object) to (a person)",
    "grab (a person)",
    "hand clap",
    "hand shake",
    "hand wave",
    "hug (a person)",
    "kick (a person)",
    "kiss (a person)",
    "lift (a person)",
    "listen to (a person)",
    "play with kids",
    "push (another person)",
    "sing to (e.g., self, a person, a group)",
    "take (an object) from (a person)",
    "talk to (e.g., self, a person, a group)",
    "watch (a person)",
)

_LABEL_TYPES = (
    *("PERSON_MOVEMENT" for _ in range(14)),
    *("OBJECT_MANIPULATION" for _ in range(49)),
    *("PERSON_INTERACTION" for _ in range(17)),
)

AVA_V22_LABELS = tuple(
    AVAActionLabel(
        label_id=label_id,
        official_name=_OFFICIAL_NAMES[label_id - 1],
        label_type=_LABEL_TYPES[label_id - 1],
        canonical_name=AVA_V22_ACTIONS[label_id],
    )
    for label_id in range(1, 81)
)


def load_ava_v22_label_map() -> tuple[AVAActionLabel, ...]:
    """Return the validated, ordered official AVA v2.2 label map."""

    if len(AVA_V22_LABELS) != 80:
        raise RuntimeError("AVA v2.2 label map must contain exactly 80 labels")
    if [label.label_id for label in AVA_V22_LABELS] != list(range(1, 81)):
        raise RuntimeError("AVA v2.2 action IDs must be contiguous from 1 through 80")
    return AVA_V22_LABELS
