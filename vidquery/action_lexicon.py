"""Explicit natural-language lexicon for the 80 AVA v2.2 action labels.

Construction rule (keep it when editing): templates are derived only from
AVA's own label wording and its official parenthetical descriptions (for
example ``watch (e.g., TV)`` or ``text on/look at a cellphone``), plus
mechanical English morphology: verb inflections, plural nouns, optional
articles/possessives, phrasal particles, and object-vs-person fillers that
choose between paired labels (``push`` vs ``push person``).  Free synonyms
that do not appear in the label wording are intentionally *not* added, so a
held-out paraphrase benchmark measures real generalisation.  The single
exception is marked where it is defined: it came from a calibration-split
failure, never from held-out queries.  A bare base form that is mostly a noun
or adjective ("open", "close", "turn") counts only directly after a person
("someone open").  Bare "listen" / "watch" resolve to the person variants
(``listen to`` / ``watch person``), as the legacy aliases did.

Template syntax (space-separated tokens):

* ``verb+``  - any inflection of the verb (``kneel+`` matches kneel, kneels,
  kneeling, kneeled, knelt);
* ``noun~``  - the noun or its plural;
* ``(a|b)``  - one of the literal alternatives;
* ``{P}``    - a person reference (someone, another person, them, ...);
* ``{O}``    - an object filler (something, an object, it, ...);
* ``{PHONE}`` / ``{COMPUTER}`` - the cellphone / computer entity words;
* ``*``      - up to three arbitrary words.

Optional articles and possessives are allowed between any two tokens.
``VETOES`` reject matches whose context makes the word non-physical, mostly
lecture idioms ("run the code", "throw an exception", "points to the next
node").  Matching prefers the longest span, so ``lifting another person``
resolves to ``lift person`` rather than ``lift/pick up``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .ava_labels import AVA_V22_ACTIONS

_DETERMINERS = (
    "a|an|the|his|her|their|its|my|your|our|some|this|that|these|those|"
    "another|other|each|one|[a-z]+'s"
)
_GAP = rf"(?:\s+(?:{_DETERMINERS}))*"
_PERSON_WORDS = (
    "each other|one another|another person|other person|someone else|"
    "persons|person|people|someone|somebody|anyone|man|men|woman|women|guy|guys|"
    "him|her|them"
)
_OBJECT_WORDS = "something|anything|objects|object|things|thing|stuff|it"
_PHONE_WORDS = (
    "cell phones|cell phone|cellphones|cellphone|mobile phone|mobiles|mobile|phones|phone"
)
_COMPUTER_WORDS = "computers|computer|laptops|laptop"
_PARTICLES = "down|up|over|around|back|out|away|in"

_IRREGULAR: dict[str, tuple[str, ...]] = {
    "lie": ("lay", "lain", "lying", "lies", "lied"),
    "sleep": ("slept",),
    "run": ("ran",),
    "sit": ("sat",),
    "stand": ("stood",),
    "swim": ("swam", "swum"),
    "fall": ("fell", "fallen"),
    "get": ("got", "gotten"),
    "eat": ("ate", "eaten"),
    "drink": ("drank", "drunk"),
    "drive": ("drove", "driven"),
    "ride": ("rode", "ridden"),
    "write": ("wrote",),
    "take": ("took", "taken"),
    "give": ("gave", "given"),
    "throw": ("threw", "thrown"),
    "catch": ("caught",),
    "hold": ("held",),
    "dig": ("dug",),
    "sing": ("sang", "sung"),
    "fight": ("fought",),
    "bend": ("bent",),
    "kneel": ("knelt",),
    "leap": ("leapt",),
    "shake": ("shook", "shaken"),
    "put": (),
    "cut": (),
    "hit": (),
    "read": (),
    "shoot": (),
}
# Monosyllables that double their final consonant before -ing / -ed.
_DOUBLING = frozenset(
    {"run", "sit", "swim", "get", "put", "cut", "dig", "chop", "stir", "hug", "grab",
     "jog", "hit", "clap", "set", "shut", "row"}
)
# Inflections that are mostly nouns/adjectives in queries and must not match bare.
_EXCLUDED_FORMS = frozenset({"written", "shot", "drunk", "lay", "lies", "lied"})


def verb_forms(base: str) -> tuple[str, ...]:
    """All inflected forms of ``base`` used for matching."""

    forms = {base}
    if base.endswith(("s", "sh", "ch", "x", "z", "o")):
        forms.add(base + "es")
    elif base.endswith("y") and base[-2:-1] not in "aeiou":
        forms.add(base[:-1] + "ies")
    else:
        forms.add(base + "s")
    if base.endswith("ie"):
        ing = base[:-2] + "ying"
    elif base.endswith("e") and not base.endswith(("ee", "ye", "oe")):
        ing = base[:-1] + "ing"
    elif base in _DOUBLING:
        ing = base + base[-1] + "ing"
    else:
        ing = base + "ing"
    forms.add(ing)
    if base.endswith("e"):
        past = base + "d"
    elif base.endswith("y") and base[-2:-1] not in "aeiou":
        past = base[:-1] + "ied"
    elif base in _DOUBLING:
        past = base + base[-1] + "ed"
    else:
        past = base + "ed"
    if base not in _IRREGULAR or base in {"lie"}:
        forms.add(past)
    forms.update(_IRREGULAR.get(base, ()))
    if base == "lie":
        forms.discard("lied")
    return tuple(sorted(form for form in forms if form not in _EXCLUDED_FORMS))


# Label id -> templates.  Each template comment names the AVA wording it comes from.
ACTION_TEMPLATES: dict[int, tuple[str, ...]] = {
    1: ("bend+", "bow+"),  # bend/bow (at the waist)
    2: ("crawl+",),
    3: ("crouch+", "kneel+"),  # crouch/kneel
    4: ("dance+",),
    5: ("fall+ down",),  # fall down
    6: ("get+ up",),  # get up
    7: ("jump+", "leap+"),  # jump/leap
    8: ("lie+", "lying", "sleep+"),  # lie/sleep
    9: ("martial art~", "do+ martial art~"),  # martial art
    10: ("run+", "jog+"),  # run/jog
    11: ("sit+",),
    12: ("stand+",),
    13: ("swim+",),
    14: ("walk+",),
    15: ("answer+ {PHONE}",),  # answer phone
    16: ("brush+ (teeth|tooth)",),  # brush teeth
    17: ("carry+", "hold+"),  # carry/hold (an object)
    18: ("catch+",),  # catch (an object)
    19: ("chop+",),
    20: ("climb+",),  # climb (e.g., a mountain)
    21: ("clink+ (glass|glasses)",),  # clink glass
    22: ("closes", "closing", "closed", "close+ (door|doors|box|boxes|window|windows)",
         "{P} close"),
    23: ("cook+",),
    24: ("cut+",),
    25: ("dig+",),
    26: ("dress+", "put+ on (clothing|clothes)"),  # dress/put on clothing
    27: ("drink+",),
    28: ("drive+",),  # drive (e.g., a car, a truck)
    29: ("eat+",),
    30: ("enter+",),
    31: ("exit+",),
    32: ("extract+",),
    33: ("fishing", "fish+ for"),  # fishing
    34: ("hit+", "hit+ {O}"),  # hit (an object)
    35: ("kick+", "kick+ {O}"),  # kick (an object)
    36: ("lift+", "pick+ up", "pick+ * up"),  # lift/pick up
    37: ("listen+ to music", "listen+ to {O}"),  # listen (e.g., to music)
    38: ("opens", "opening", "opened", "open+ (door|doors|window|windows|box|boxes)",
         "{P} open"),
    39: ("paint+",),
    40: ("play+ board game~",),  # play board game
    41: ("play+ musical instrument~", "play+ instrument~"),  # play musical instrument
    42: ("play+ with (pet|pets)",),  # play with pets
    43: ("pointing", "pointed", "points", "point+ (to|at)"),  # point to (an object)
    44: ("press+",),  # press
    45: ("pull+",),  # pull (an object)
    46: ("push+", "push+ {O}"),  # push (an object)
    47: ("put+ down", "put+ * down"),  # put down
    48: ("read+",),
    49: ("ride+",),  # ride (e.g., a bike, a car, a horse)
    50: ("rowing", "rowed", "row+ boat~"),  # row boat
    51: ("sail+", "sail+ boat~", "sailboat~"),  # sail boat
    52: ("shoot+",),
    53: ("shovel+",),
    54: ("smoke+",),
    55: ("stir+",),
    56: ("take+ photo~",),  # take a photo
    # text on/look at a cellphone
    57: ("texting", "texted", "text+ on {PHONE}", "look+ at {PHONE}"),
    58: ("throw+",),
    59: ("touch+", "touch+ {O}"),  # touch (an object)
    60: ("turn+ screwdriver~", "{P} turn"),  # turn (e.g., a screwdriver)
    61: ("watch+ (tv|television)", "watch+ {O}"),  # watch (e.g., TV)
    62: ("work+ on {COMPUTER}",),  # work on a computer
    63: ("write+",),
    64: ("fight+", "hit+ {P}"),  # fight/hit (a person)
    # give/serve (an object) to (a person).  "handing ... to" is the one
    # calibration-derived synonym (from a calibration-split failure).
    65: ("give+", "serve+", "(handing|handed) * to {P}", "hand+ {O} to {P}"),
    66: ("grab+", "grab+ {P}"),  # grab (a person)
    67: ("clap+", "hand clap~"),  # hand clap
    68: ("shake+ hands", "hand shake~", "handshake~"),  # hand shake
    69: ("waving", "waved", "wave+ (hand|hands)", "hand wave~"),  # hand wave
    70: ("hug+",),  # hug (a person)
    71: ("kick+ {P}",),  # kick (a person)
    72: ("kiss+",),  # kiss (a person)
    73: ("lift+ {P}",),  # lift (a person)
    74: ("listen+", "listen+ to", "listen+ to {P}"),  # listen to (a person)
    75: ("play+ with (kid|kids)",),  # play with kids
    76: ("push+ {P}",),  # push (another person)
    77: ("sing+", "sing+ to", "sing+ to {P}"),  # sing to (e.g., self, a person, a group)
    78: ("take+ * from {P}",),  # take (an object) from (a person)
    79: ("talk+ to", "talk+ to {P}", "talking"),  # talk to (e.g., self, a person, a group)
    80: ("watch+", "watch+ {P}"),  # watch (a person)
}

_DEVICE_WORDS = r"(?:code|program|programs|script|scripts|tests?|query|queries|commands?|models?|"
_DEVICE_WORDS += r"experiments?|server|servers|algorithm|simulation|functions?|loop|loops|app|"
_DEVICE_WORDS += r"application|jobs?|file|files|terminal|browser|tab|editor|notebook|link|project|"
_DEVICE_WORDS += r"repo|repository|database|data|input|memory|bytes|line|lines)\b"
_DET = r"(?:(?:the|a|an|this|that|my|our|your|these|those)\s+)?"

# Context patterns that cancel a match overlapping them.
VETOES: dict[int, tuple[str, ...]] = {
    10: (rf"\brun(?:s|ning)?\s+{_DET}{_DEVICE_WORDS}", r"\brun(?:ning)?\s+time\b",
         r"\blong run\b"),
    14: (r"\bwalk(?:s|ing|ed)?\s+(?:us\s+|you\s+|them\s+)?through\b",),
    7: (r"\bjump(?:s|ing|ed)?\s+(?:to|ahead|back|into)\b",),
    8: (r"\b(?:tell|tells|told|telling)\s+(?:a\s+)?lies?\b", r"\blie(?:s)?\s+about\b",
        r"\blying\s+about\b"),
    46: (rf"\bpush(?:es|ing|ed)?\s+{_DET}(?:code|changes|commits?|branch|updates?|repo|"
         r"repository)\b", r"\bgit\s+push\b"),
    45: (r"\bpull(?:s|ing|ed)?\s+requests?\b",
         rf"\bpull(?:s|ing|ed)?\s+{_DET}(?:code|changes|branch|repo|data)\b"),
    28: (r"\b(?:hard|google|usb|flash|disk|external|thumb|network)\s+drives?\b",),
    32: (rf"\bextract(?:s|ing|ed)?\s+{_DET}(?:features?|data|information|text|frames?|"
         r"keywords?|values?)\b",),
    25: (r"\bdig(?:s|ging)?\s+into\b",),
    43: (r"\bpoint(?:s|ing|ed)?\s+to\s+(?:the\s+|a\s+)?(?:next|previous|null|nodes?|address|"
         r"memory|head|tail|location|value)\b", r"\bpointer\b"),
    34: (r"\b(?:cache|page)\s+hits?\b", r"\bhit\s+(?:rate|ratio)\b"),
    58: (rf"\bthrow(?:s|ing|n)?\s+{_DET}(?:exceptions?|errors?)\b",),
    18: (rf"\bcatch(?:es|ing)?\s+{_DET}(?:exceptions?|errors?)\b",),
    65: (rf"\bgive(?:s|n)?\s+{_DET}(?:examples?|overview|lecture|talk|presentation|demo|hints?|"
         r"answer|summary|introduction)\b", rf"\bgiving\s+{_DET}(?:examples?|overview|lecture|"
         r"talk|presentation|demo|hints?|answer|summary|introduction)\b",
         rf"\bserv(?:e|es|ing|ed)\s+{_DET}(?:requests?|models?|pages?|traffic|files?)\b"),
    48: (rf"\bread(?:s|ing)?\s+(?:from\s+)?{_DET}(?:file|files|input|data|memory|bytes|line|"
         r"lines|database)\b",),
    63: (rf"\bwrit(?:e|es|ing)\s+(?:to|into)\s+{_DET}(?:file|files|disk|memory|database)\b",),
    38: (rf"\bopen(?:s|ing|ed)?\s+{_DET}{_DEVICE_WORDS}", r"\bopen\s+source\b",
         r"\b(?:the|a|an|its|their|his|her)\s+(?:open|opened)\b"),
    22: (rf"\bclos(?:e|es|ing|ed)\s+{_DET}{_DEVICE_WORDS}", r"\bclose\s+(?:to|by)\b",
         r"\b(?:the|a|an|its|their|his|her)\s+(?:closed|close)\b"),
    30: (rf"\benter(?:s|ing|ed)?\s+{_DET}(?:commands?|password|values?|data|text|numbers?|"
         r"input)\b",),
    31: (rf"\bexit(?:s|ing|ed)?\s+{_DET}(?:program|loop|function|application|app|code)\b",
         r"\b(?:the|an|emergency)\s+exit\b"),
    59: (r"\btouch(?:es|ing|ed)?\s+(?:on|upon)\b", r"\bin\s+touch\b"),
    61: (r"\b(?:a|the|his|her|my|your|wrist)\s+watch(?:es)?\b",),
    80: (r"\b(?:a|the|his|her|my|your|wrist)\s+watch(?:es)?\b",),
    44: (r"\bpress\s+conference\b", r"\bthe\s+press\b"),
    12: (r"\bstand(?:s|ing)?\s+for\b", r"\bunderstand\b"),
}


def _expand_token(token: str) -> str:
    if token == "*":
        return r"(?:\s+[a-z0-9']+){0,3}?"
    placeholders = {
        "{P}": _PERSON_WORDS,
        "{O}": _OBJECT_WORDS,
        "{PHONE}": _PHONE_WORDS,
        "{COMPUTER}": _COMPUTER_WORDS,
    }
    if token in placeholders:
        return rf"(?:{placeholders[token]})"
    if token.startswith("(") and token.endswith(")"):
        return rf"(?:{token[1:-1]})"
    if token.endswith("+"):
        return "(?:" + "|".join(re.escape(form) for form in verb_forms(token[:-1])) + ")"
    if token.endswith("~"):
        word = token[:-1]
        return rf"(?:{re.escape(word)}|{re.escape(word)}s|{re.escape(word)}es)"
    return re.escape(token)


def _template_regex(template: str) -> str:
    tokens = template.split()
    parts: list[str] = []
    for index, token in enumerate(tokens):
        piece = _expand_token(token)
        if index == 0:
            parts.append(piece)
        elif token == "*":
            parts.append(piece)
        else:
            parts.append(_GAP + r"\s+" + piece)
    # A trailing phrasal particle is consumed so it never becomes a spoken term.
    body = "".join(parts) + rf"(?:\s+(?:{_PARTICLES}))?"
    return rf"(?<![a-z0-9_']){body}(?![a-z0-9_])"


@dataclass(frozen=True, slots=True)
class ActionMatch:
    label_id: int
    label: str
    start: int
    end: int
    template: str

    @property
    def text_span(self) -> tuple[int, int]:
        return self.start, self.end


@lru_cache(maxsize=1)
def _compiled() -> tuple[tuple[int, str, re.Pattern[str], int], ...]:
    compiled: list[tuple[int, str, re.Pattern[str], int]] = []
    for label_id, templates in ACTION_TEMPLATES.items():
        for template in templates:
            literal = sum(1 for token in template.split() if token != "*")
            compiled.append((label_id, template, re.compile(_template_regex(template)), literal))
    return tuple(compiled)


@lru_cache(maxsize=1)
def _compiled_vetoes() -> dict[int, tuple[re.Pattern[str], ...]]:
    return {
        label_id: tuple(re.compile(pattern) for pattern in patterns)
        for label_id, patterns in VETOES.items()
    }


def _vetoed(label_id: int, text: str, start: int, end: int) -> bool:
    for pattern in _compiled_vetoes().get(label_id, ()):
        for found in pattern.finditer(text):
            if found.start() < end and found.end() > start:
                return True
    return False


_LABEL_IDS = {label: label_id for label_id, label in AVA_V22_ACTIONS.items()}


def is_vetoed(label: str, text: str, start: int, end: int) -> bool:
    """Whether a mention of ``label`` at ``text[start:end]`` sits in a veto context."""

    label_id = _LABEL_IDS.get(label)
    return label_id is not None and _vetoed(label_id, text, start, end)


def match_actions(text: str) -> list[ActionMatch]:
    """Non-overlapping AVA action mentions in normalised (lower-case) ``text``."""

    candidates: list[tuple[int, int, int, int, str]] = []
    for label_id, template, pattern, literal in _compiled():
        for found in pattern.finditer(text):
            if found.end() <= found.start():
                continue
            if _vetoed(label_id, text, found.start(), found.end()):
                continue
            candidates.append((found.start(), found.end(), literal, label_id, template))
    # Longest span first; then the more specific template; then the lower label id.
    candidates.sort(key=lambda item: (-(item[1] - item[0]), -item[2], item[3], item[0]))
    selected: list[ActionMatch] = []
    for start, end, _literal, label_id, template in candidates:
        if any(start < item.end and end > item.start for item in selected):
            continue
        selected.append(
            ActionMatch(label_id, AVA_V22_ACTIONS[label_id], start, end, template)
        )
    return sorted(selected, key=lambda item: item.start)


# Bare legacy aliases that are too ambiguous to force an action on their own.
AMBIGUOUS_BARE_ALIASES = frozenset(
    {"turn", "close", "open", "point", "text on phone", "row", "exit", "enter", "press",
     "shoot", "cut", "read", "write", "hit object", "touch object", "extract"}
)
