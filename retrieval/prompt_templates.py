"""Legacy module retained to make the security decision explicit.

VidQuery does not ask a language model to generate executable Cypher.  See
``vidquery.search.StructuredQueryParser`` and ``SafeCypherCompiler``.
"""

SYSTEM_PROMPT = "DISABLED: executable Cypher is compiled from an allowlisted QueryPlan."
FEW_SHOT_EXAMPLES: list[dict[str, str]] = []


def build_prompt(query: str) -> list[dict[str, str]]:
    raise RuntimeError(
        "Raw LLM-to-Cypher generation is disabled; use StructuredQueryParser"
    )
