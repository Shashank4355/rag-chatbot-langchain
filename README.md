# RAG chatbot: naive vs fixed (LangChain)

A help-center chatbot over 11 articles for **Kitebox**, a made-up file-sharing product. It runs two pipelines over the same documents so you can see what each fix changes:

- **Naive**: how most first versions are wired. 1,000-character chunks, dense top-4, "use the context to answer", retrieval on the latest message only, no checks.
- **Fixed**: heading-aware chunks with breadcrumbs, tables rewritten row by row, hybrid BM25 + dense retrieval with an exact match on error codes, a relevance gate before the model, follow-ups rewritten into standalone questions, a cite-or-refuse prompt, and a check that every citation and every number in the answer is backed by a retrieved excerpt (one retry, then refuse).

Live demo: _see the link in the proposal_. The **Chat** tab has a Side by side mode; click any answer to see what was retrieved, by which retriever, and which checks it passed. The **Eval results** tab shows all 33 questions.

## Results (33 questions, `eval/questions.json`)

| | Naive | Fixed |
|---|---|---|
| Answers correct | 18 / 33 | 28 / 33 |
| Out-of-scope questions refused | 0 / 5 | 4 / 5 |
| In-scope questions wrongly refused | 0 / 28 | 1 / 28 |

These numbers are from **extractive mode** (no LLM key: the answer is the best-matching sentences of the retrieved excerpts), so they measure retrieval and the guardrails only. With a key set, run `python -m eval.run_eval` again; it rewrites `eval/results.json`, which the Eval tab reads.

Question types: facts, table lookups, error codes (KB-412, SSO-403), false premises ("SCIM on Business"), facts changed by a later release note, multi-part, follow-ups ("And Business?") and questions the docs don't answer (CEO, Salesforce, KB-777).

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                                # optional: add an OpenAI-compatible key
uvicorn api.index:app --reload                      # http://localhost:8000
pytest -q                                           # 15 tests
python -m eval.run_eval                             # both pipelines, all questions
```

The index (`index/`) is built from `kb/*.md` and committed. Delete it and it is rebuilt on start.

## Layout

```
rag/kb.py           loading + the two chunkers (naive, fixed), table serialisation
rag/embeddings.py   bge-small-en-v1.5 (int8 ONNX) as a LangChain Embeddings class
rag/retrieval.py    DenseRetriever, HybridRetriever (BM25 + dense + RRF + exact codes)
rag/pipeline.py     prompts, LCEL chains, relevance gate, grounding checks, extractive fallback
api/index.py        FastAPI: /api/chat, /api/meta, /api/eval, /api/kb/{file}, rate limit
public/             the chat UI, eval view, knowledge-base viewer
eval/               questions, runner, last results
tests/              pytest
```

## Deploy (Vercel)

Import the repo; no build settings needed. `api/index.py` runs as a Python function, `public/` is served statically. Add `OPENAI_API_KEY` (and `OPENAI_BASE_URL` / `OPENAI_MODEL` for Gemini, Groq or Claude's OpenAI-compatible endpoint) under Project Settings > Environment Variables to switch from extractive answers to LLM answers.

## Notes

- Retrieval uses an in-memory index because the knowledge base is 46 chunks. Both retrievers are LangChain `BaseRetriever`s, so swapping in Pinecone, pgvector, Chroma or Qdrant changes the retriever, not the chain.
- The same embedding model embeds documents (at build) and questions (at query). A mismatch there is one of the most common reasons a bot's retrieval quietly degrades.
- The grounding checks catch invented numbers and dangling citations. They don't prove an answer is right; the eval does that, question by question.
