# Groq grounded RAG answer layer

VidQuery retrieves and ranks evidence first. The optional Groq layer receives at
most `RAG_TOP_K` already-ranked segments and synthesizes an answer; it cannot
query SQLite or Neo4j, generate Cypher, change ranking, or hide raw results.

## Configuration

Set the API key only in the runtime environment. Never commit its value:

```powershell
$env:GROQ_API_KEY="..."
$env:ENABLE_RAG_GENERATION="true"
$env:RAG_PROVIDER="groq"
$env:RAG_MODEL="openai/gpt-oss-20b"
$env:RAG_TOP_K="5"
$env:RAG_MIN_EVIDENCE_SCORE="0.20"
python -m vidquery serve --host 127.0.0.1 --port 8000
```

`RAG_MODEL` is configurable because Groq model availability changes. The adapter
uses Groq's OpenAI-compatible chat-completions endpoint with strict JSON Schema
output. Only transcript, speakers, OCR, entities, actions, relationships,
retrieval score, video identity, and exact start/end timestamps are sent.

The response contract is:

```json
{
  "answer": "...",
  "supported": true,
  "evidence_ids": ["E1", "E2"]
}
```

VidQuery validates every returned ID against the evidence supplied in that same
request. Valid IDs are converted to exact video/segment/start/end citations.
Unknown IDs, supported answers without citations, or malformed provider output
are rejected. If no result meets `RAG_MIN_EVIDENCE_SCORE`, VidQuery returns an
unsupported answer without calling Groq.

Quota, timeout, network, schema, and provider failures set a `rag_status` but do
not fail `/api/search`; ranked retrieval remains visible and usable. The UI shows
the answer above the original cards, and each citation opens the existing player
at its exact start time.

## Verification

Normal tests use deterministic injected transports and consume no Groq quota:

```powershell
python -m pytest -q tests/test_rag.py
```

After setting `GROQ_API_KEY`, capture exactly one sanitized live proof:

```powershell
python -m scripts.capture_groq_rag_proof `
  --query "Where is architecture discussed?" `
  --video-name final_demo.mp4
```

The output is `evaluation/results/groq-rag-live-proof.json`. It contains no API
key and verifies that raw results remain present and every citation interval
matches a returned ranked segment.

## Live proof

On 13 August 2026, the normal `/api/search` path completed a real Groq request
using `openai/gpt-oss-20b` for `Where is architecture discussed?`, scoped to
`final_demo.mp4`. Groq returned `supported=true` with `E1` and `E2`. VidQuery
validated both IDs and resolved them to exact ranked intervals 00:00–00:05 and
00:15–00:20. Raw results remained present. The sanitized artifact is
`evaluation/results/groq-rag-live-proof.json`; its secret scan passed.
