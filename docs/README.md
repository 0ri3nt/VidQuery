# VidQuery documentation

Start with the [project README](../README.md), then [SETUP.md](SETUP.md).

## Using and running the system

| Document | Contents |
|---|---|
| [SETUP.md](SETUP.md) | install, credentials, model cache layout, health check, troubleshooting |
| [RUN_AND_TEST.md](RUN_AND_TEST.md) | running, examiner-style testing, demo-day checklist, reporting issues |
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
| [EVALUATION_END_TO_END.md](EVALUATION_END_TO_END.md) | model-only end-to-end action benchmark, learned trust, and abstention ablations |
| [EVALUATION_CHARADES_STA.md](EVALUATION_CHARADES_STA.md) | Charades-STA + NA-VMR temporal localization and answerability benchmark (separate from AVA) |
| [PROGRESS_2026-10-25.md](PROGRESS_2026-10-25.md) | work log, measured results, and approval gates for pending expansions |
| [EXPERIMENTS_2026-10-26.md](EXPERIMENTS_2026-10-26.md) | deterministic Whisper, AVA 24->94 and VidOR 400->1,900 training-video experiments with paired CIs |
| [AUDIT_2026-10.md](AUDIT_2026-10.md) | codebase, dataset-coverage, hardware, feasibility, and cost audit |
| [NEO4J_VERIFICATION.md](NEO4J_VERIFICATION.md) | live Neo4j verification |
| [PIPELINE_REPRODUCTION.md](archive/PIPELINE_REPRODUCTION.md) | end-to-end reproduction record |

## Historical records

Kept for traceability; each is superseded by a document above. Files under
`archive/` are indexed in [archive/README.md](archive/README.md).

| Document | Superseded by |
|---|---|
| [IMPLEMENTATION_AUDIT.md](archive/IMPLEMENTATION_AUDIT.md) | baseline audit before the application existed |
| [EVALUATION.md](archive/EVALUATION.md) | [EVALUATION_FINAL.md](EVALUATION_FINAL.md) |
| [DEMO_GUIDE.md](archive/DEMO_GUIDE.md) | [FINAL_DEMO_GUIDE.md](FINAL_DEMO_GUIDE.md) |
| [TEAMMATE_IMPROVEMENT_PLAN.md](archive/TEAMMATE_IMPROVEMENT_PLAN.md) | [PROGRESS_2026-10-25.md](PROGRESS_2026-10-25.md), [EXPERIMENTS_2026-10-26.md](EXPERIMENTS_2026-10-26.md) |
| [IMPLEMENTATION_PROGRESS_SLIDES.md](archive/IMPLEMENTATION_PROGRESS_SLIDES.md) | presentation draft in `presentation/` |
| [VIDOR_RELATION_GNN.md](VIDOR_RELATION_GNN.md) | [VIDOR_RELATION_FINAL_PASS.md](VIDOR_RELATION_FINAL_PASS.md) |
| [LIBRARY_REFRESH_VERIFICATION.md](LIBRARY_REFRESH_VERIFICATION.md) | [SETUP.md](SETUP.md) |
