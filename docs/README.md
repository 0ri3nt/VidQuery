# VidQuery documentation

Start with the [project README](../README.md), then [SETUP.md](SETUP.md).

## Using and running the system

| Document | Contents |
|---|---|
| [SETUP.md](SETUP.md) | install, credentials, model cache layout, health check, troubleshooting |
| [API.md](API.md) | HTTP endpoints, search request and response contract, health states |
| [FINAL_DEMO_GUIDE.md](FINAL_DEMO_GUIDE.md) | controlled demo script and expected results |
| [UI_QA.md](UI_QA.md) | browser and deployment QA |
| [RETRIEVAL_DIAGNOSTICS.md](RETRIEVAL_DIAGNOSTICS.md) | `diagnose-query`, failure taxonomy, custom-video case study |

## Design

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | pipeline, component boundaries, trust boundaries |
| [REVIEW2.md](REVIEW2.md) | timestamp localization, reliability fusion, query hypotheses, challenge benchmark |
| [DATA_FORMATS.md](DATA_FORMATS.md) | canonical segment and artifact formats |
| [GRAPH_SCHEMA.md](GRAPH_SCHEMA.md) | Neo4j graph schema |
| [GROQ_QUERY_PLANNER.md](GROQ_QUERY_PLANNER.md) | constrained Groq query planning |
| [GROQ_RAG.md](GROQ_RAG.md) | grounded answer synthesis |

## Models

| Document | Contents |
|---|---|
| [GNN_STATUS.md](GNN_STATUS.md) | metrics and safe claim boundaries for both GNNs |
| [AVA80_GNN.md](AVA80_GNN.md) | 80-label AVA person-action GNN |
| [ACTION_MODEL.md](ACTION_MODEL.md) | legacy reduced action model |
| [VIDOR_RELATION_FINAL_PASS.md](VIDOR_RELATION_FINAL_PASS.md) | active VidOR pair-visual relation GNN |
| [MODEL_LIMITATIONS.md](MODEL_LIMITATIONS.md) | what each model and layer cannot do |

## Evidence and evaluation

| Document | Contents |
|---|---|
| [FINAL_CAPSTONE_EVIDENCE.md](FINAL_CAPSTONE_EVIDENCE.md) | claims that are safe, and unsafe, to make |
| [EVALUATION_FINAL.md](EVALUATION_FINAL.md) | 62-query independent retrieval benchmark |
| [NEO4J_VERIFICATION.md](NEO4J_VERIFICATION.md) | live Neo4j verification |
| [PIPELINE_REPRODUCTION.md](PIPELINE_REPRODUCTION.md) | end-to-end reproduction record |

## Historical records

Kept for traceability; each is superseded by a document above.

| Document | Superseded by |
|---|---|
| [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) | baseline audit before the application existed |
| [EVALUATION.md](EVALUATION.md) | [EVALUATION_FINAL.md](EVALUATION_FINAL.md) |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | [FINAL_DEMO_GUIDE.md](FINAL_DEMO_GUIDE.md) |
| [VIDOR_RELATION_GNN.md](VIDOR_RELATION_GNN.md) | [VIDOR_RELATION_FINAL_PASS.md](VIDOR_RELATION_FINAL_PASS.md) |
| [LIBRARY_REFRESH_VERIFICATION.md](LIBRARY_REFRESH_VERIFICATION.md) | [SETUP.md](SETUP.md) |
