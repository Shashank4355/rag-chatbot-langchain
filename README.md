# RAG chatbot: naive vs fixed (LangChain)

A help-center chatbot over 11 articles for **Kitebox**, a made-up file-sharing product. It runs two pipelines over the same documents so you can see what each fix changes:

- **Naive**: how most first versions are wired. 1,000-character chunks, dense top-4, "use the context to answer", retrieval on the latest message only, no checks.
- **Fixed**: heading-aware chunks with breadcrumbs, tables rewritten row by row, hybrid BM25 + dense retrieval with an exact match on error codes, a relevance gate before the model, follow-ups rewritten into standalone questions, a cite-or-refuse prompt, and a check that every citation and every number in the answer is backed by a retrieved excerpt (one retry, then refuse).

Live demo: https://rag-chatbot-langchain-mu.vercel.app The **Chat** tab has a Side by side mode; click any answer to see what was retrieved, by which retriever, and which checks it passed. The **Eval results** tab shows all 33 questions.

## Results (33 questions, `eval/questions.json`)

| | Naive | Fixed |
|---|---|---|
| **With Gemini Flash** (`eval/results_llm.json`) | | |
| Answers correct | 33 / 33 | 33 / 33 |
| Answers that cite their source and pass the checks | 0 / 28 | 28 / 28 |
| Out-of-scope questions refused | 5 / 5 | 5 / 5 |
| **Retrieval only, no LLM** (`eval/results_extractive.json`) | | |
| Answers correct | 18 / 33 | 28 / 33 |
| Out-of-scope questions refused | 0 / 5 | 4 / 5 |
| In-scope questions wrongly refused | 0 / 28 | 1 / 28 |

On 11 short, clean articles a strong model makes up for weak retrieval, so with Gemini both pipelines get every answer right. The difference there is traceability: every fixed answer cites its excerpt and passed the checks; no naive answer cites anything. With the model taken out, the answer is just the best-matching retrieved sentences, which shows what the chunking, hybrid search and gate change. Bigger or messier documents shrink the model's cushion, which is why the eval should be run on a client's own questions first.

`python -m eval.run_eval` writes `results_llm.json` when a key is set and `results_extractive.json` when not; the Eval tab shows both.

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

## Models: free tier first

With only `GEMINI_API_KEY` set, the bot uses Google AI Studio's free tier through its OpenAI-compatible endpoint, as a LangChain fallback chain: `gemini-flash-latest` → `gemini-flash-lite-latest` → `gemini-3.1-flash-lite`. Retries are off, so a 429 on one model moves to the next at once. A daily cap (`DAILY_LLM_CAP`) and a per-visitor limit keep a public demo inside the free quota; past either, answers fall back to extractive mode with a note instead of an error. Any OpenAI-compatible key (`OPENAI_API_KEY`, comma-separated `OPENAI_MODEL` for fallbacks) takes precedence when set.

## Deploy (Vercel)

Import the repo; no build settings needed. `api/index.py` runs as a Python function, `public/` is served statically. Add `GEMINI_API_KEY` under Project Settings > Environment Variables and redeploy.

## Notes

- Retrieval uses an in-memory index because the knowledge base is 46 chunks. Both retrievers are LangChain `BaseRetriever`s, so swapping in Pinecone, pgvector, Chroma or Qdrant changes the retriever, not the chain.
- The same embedding model embeds documents (at build) and questions (at query). A mismatch there is one of the most common reasons a bot's retrieval quietly degrades.
- The grounding checks catch invented numbers and dangling citations. They don't prove an answer is right; the eval does that, question by question.
