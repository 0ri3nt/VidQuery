# Constrained Groq query planner

VidQuery applies local aliases, morphology, plurals, speech cues, and OCR cues
before considering Groq. Clear queries never consume planner quota. Ambiguous
queries follow this fixed boundary:

```text
natural language -> deterministic parser -> ambiguity check
  -> validated Groq plan when needed -> VidQuery retrieval
  -> existing grounded Groq answer over retrieved evidence
```

The planner receives only the supported AVA action, VidOR predicate, entity,
and alias vocabularies plus a strict JSON schema. It never receives video
evidence, cannot access SQLite or Neo4j, cannot create or execute database
queries, and cannot rank results. Unsupported labels, invalid speakers, invalid
JSON, low confidence, timeouts, and quota errors fall back to the deterministic
plan. `find drive` remains explicitly ambiguous and searches both `action:
drive` and `spoken: drive`.

Simple English action inflections are normalized locally (for example,
`kissing`/`kissed`/`kisses` to `kiss`). If meaningful terms remain unclassified
and the user did not explicitly request spoken or visible text, parser
confidence is treated as low and the constrained planner is invoked. Explicit
cues remain authoritative: `where did they say kissing?` is a transcript search
and does not ask Groq to reinterpret it as an observed action.

Configuration uses the existing environment-only `GROQ_API_KEY`:

```text
ENABLE_QUERY_PLANNER=true
QUERY_PLANNER_MODEL=openai/gpt-oss-20b
QUERY_PLANNER_MIN_CONFIDENCE=0.70
QUERY_PLANNER_TIMEOUT_SECONDS=10
```

Normal tests use an injected mock transport and consume no Groq quota:

```powershell
python -m pytest tests/test_query_planner.py -q
```
