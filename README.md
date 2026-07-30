# 🇫🇷 French Residency Assistant

An AI-powered RAG (Retrieval-Augmented Generation) application that answers questions about French residency documents, permits, and administrative processes — grounded in real official sources.

**Live API**: https://french-residency-api.whiteglacier-3fe4620b.francecentral.azurecontainerapps.io/docs

---

## What it does

Users ask natural-language questions about French residency (carte de séjour, carte de résident, civic exam, OFII medical visit) and receive accurate, cited answers grounded in real administrative documents — not hallucinated general knowledge.

---

## Architecture

User Question
↓
Input Guardrail (on-topic check)
↓
Hybrid Search: BM25 + Vector Similarity (Chroma)
↓
Reciprocal Rank Fusion (RRF)
↓
Cross-Encoder Reranking
↓
LLM Generation (GPT-4o-mini) with citation
↓
Output Guardrail (grounding check)
↓
Cached Answer

---

## Technical Stack

| Layer | Technology |
|---|---|
| API | FastAPI + JWT authentication |
| RAG Pipeline | Hybrid search (BM25 + vector), RRF, cross-encoder reranking |
| Vector Database | ChromaDB (embedded) |
| Embeddings | OpenAI text-embedding-3-small with contextual retrieval |
| LLM | GPT-4o-mini |
| Containerization | Docker + Docker Compose |
| Deployment | Azure Container Apps + Azure Container Registry |
| CI/CD | GitHub Actions (auto-test on every push) |
| Testing | pytest (7 tests, mocked for CI) |
| Observability | Latency + cost logging per request |
| Caching | Exact-match query cache (in-memory) |
| Guardrails | Input topic filter + output grounding check |

---

## RAG Pipeline Details

- **Contextual retrieval**: each chunk is enriched with an AI-generated context summary before embedding, improving retrieval accuracy for short/ambiguous fragments
- **Hybrid search**: combines BM25 keyword matching with vector similarity — catches exact terms (form numbers, article references) that pure semantic search misses
- **Reciprocal Rank Fusion**: merges BM25 and vector rankings into one combined score
- **Cross-encoder reranking**: re-scores top retrieved chunks with a more precise model before generation
- **Embedding cache**: avoids re-embedding unchanged document chunks on ingestion reruns

---

## Document Corpus

Four real French administrative documents:
- Carte de résident (10-year permit) requirements
- Carte de séjour pluriannuelle renewal process
- OFII medical visit procedure
- Civic exam (examen civique) overview

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/dsdatta/french-residency-assistant.git
cd french-residency-assistant

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Ingest documents (first time only)
python ingest.py

# Run the API
uvicorn api:app --reload

# Or run with Docker Compose
docker compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/token` | Get JWT token (username: sameer, password: secret) |
| POST | `/query` | Ask a residency question (requires auth) |

---

## Running Tests

```bash
pytest tests/ -v
```

CI runs automatically on every push via GitHub Actions.

---

## Design Decisions

**Why RAG over fine-tuning?** The assistant needs current, citable, updatable facts from specific documents — RAG handles this naturally. Fine-tuning would bake facts into weights unreliably and lose citation capability.

**Why hybrid search?** Pure vector search sometimes misses exact terms like form numbers or article references. BM25 catches these; combining both gives the best of both retrieval approaches.

**Why in-memory query caching?** Repeated identical questions (common in a FAQ-style assistant) are served instantly without re-running the 6-10 second pipeline. A production deployment would use Redis for persistent cross-instance caching.

---

## What I'd Add in Production

- Redis for persistent, cross-instance query caching
- Semantic caching for paraphrase variations
- A proper user database replacing the hardcoded credentials
- Full evaluation framework (Langfuse) for systematic answer quality scoring
- Angular frontend (in progress)