"""The two pipelines the demo compares, built from LangChain pieces.

naive  = 1,000-char chunks, dense top-4, a permissive prompt, retrieval on the
         latest message only. The way most first versions are wired.
fixed  = heading-aware chunks, hybrid retrieval, a relevance gate that refuses
         before the model is called, follow-ups rewritten into standalone
         questions, a cite-or-refuse prompt, and a check that every citation and
         every number in the answer is backed by a retrieved excerpt.

With no LLM key configured, both pipelines fall back to extractive answers
(sentences picked from the retrieved excerpts), so retrieval and the guardrails
can still be tried.
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .embeddings import MODEL_NAME, BgeSmallEmbeddings
from .kb import load_kb, split_fixed, split_naive
from .llm import LLMUnavailable, make_llm
from .retrieval import DenseRetriever, HybridRetriever, VectorIndex, codes_in, normalise_codes, tokens

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
REFUSAL = "I couldn't find that in the Kitebox help center."
# Below this cosine similarity the best excerpt is treated as unrelated and the
# model is never called. Tuned on eval/questions.json (see README).
GATE = float(os.getenv("RELEVANCE_FLOOR", "0.55"))
# Question words that say nothing about the topic, so their absence from the knowledge base means nothing.
GENERIC = set("kitebox much many long get got need want tell know there their they them about also any some way does did "
              "doe should would could make made use using used help please work explain mean happen happens happened "
              "much cost costs price thing things like".split())

FIXED_SYSTEM = f"""You answer questions about Kitebox using only the numbered help-center excerpts you are given.

Rules:
1. Use only facts stated in the excerpts. Never use outside knowledge.
2. If the excerpts don't answer the question, reply with exactly: {REFUSAL}
3. End every sentence with the number of the excerpt it comes from, like [2].
4. If the question assumes something the excerpts contradict (a feature on the wrong plan, a limit that doesn't exist), say so first.
5. If two excerpts give different values, use the one with the later "updated" date and say what changed.
6. Copy plan names, numbers and limits exactly. At most 4 short sentences, or a short list."""

NAIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for Kitebox."),
    MessagesPlaceholder("history"),
    ("human", "Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"),
])
FIXED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", FIXED_SYSTEM),
    ("human", "Excerpts:\n\n{context}\n\nQuestion: {question}"),
])
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest message as one standalone question about Kitebox, using the conversation "
               "for context. Keep error codes, plan names and numbers exactly as written. Return only the question."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
RETRY_NOTE = ("Your previous answer failed a check: {problem}. Answer again using only the excerpts, with an [n] "
              "citation on every sentence. If the excerpts don't contain the answer, reply with exactly: " + REFUSAL)


# ---------- index ----------

def build_index(embedder: BgeSmallEmbeddings, out_dir: Path = INDEX_DIR) -> None:
    docs = load_kb()
    out_dir.mkdir(exist_ok=True)
    for name, chunks in (("naive", split_naive(docs)), ("fixed", split_fixed(docs))):
        vecs = embedder.embed([c.page_content for c in chunks])
        np.save(out_dir / f"{name}.npy", vecs)
        (out_dir / f"{name}.json").write_text(json.dumps(
            [{"text": c.page_content, "meta": c.metadata} for c in chunks], indent=1), encoding="utf-8")


def load_index(name: str) -> VectorIndex:
    rows = json.loads((INDEX_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return VectorIndex([Document(page_content=r["text"], metadata=r["meta"]) for r in rows], np.load(INDEX_DIR / f"{name}.npy"))


# ---------- checks ----------

NUM = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUM.findall(text)}


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower().rstrip(".") in answer.lower()


def check_answer(answer: str, excerpts: list[str], question: str, require_citations: bool = True) -> list[dict]:
    """Deterministic grounding checks. They catch invented figures and dangling citations, not every error."""
    if is_refusal(answer):
        return [{"name": "Refused instead of guessing", "ok": True, "detail": "No claim to check."}]
    checks = []
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    valid = [c for c in cited if 1 <= c <= len(excerpts)]
    if require_citations:
        bad = [c for c in cited if c not in valid]
        ok = bool(valid) and not bad
        checks.append({"name": "Citations point to retrieved excerpts", "ok": ok,
                       "detail": "No citations." if not cited else f"Unknown excerpt {bad}." if bad else f"Cites {valid}."})
        pool = " ".join(excerpts[c - 1] for c in valid)
    else:
        pool = " ".join(excerpts)
    body = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", answer)
    missing = sorted(numbers(body) - numbers(pool + " " + question), key=len)
    where = "the cited excerpts" if require_citations else "any retrieved excerpt"
    checks.append({"name": "Every number is in " + where, "ok": not missing,
                   "detail": f"Not found: {', '.join(missing)}." if missing else f"{len(numbers(body))} number(s) checked."})
    return checks


# ---------- extractive fallback ----------

def sentences(text: str) -> list[str]:
    parts = []
    for line in text.split("\n"):
        line = line.strip().lstrip("-0123456789. ").strip() if re.match(r"^(\d+\.|-)\s", line.strip()) else line.strip()
        parts += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if len(s.strip()) > 15]
    return parts


def extractive(question: str, docs: list[Document], embedder, n: int = 2, top_docs: int = 3, crumb: bool = True) -> str:
    cands = []
    for k, d in enumerate(docs[:top_docs], 1):
        body = d.page_content.split("\n", 1)[-1] if crumb else d.page_content  # drop the breadcrumb line
        cands += [(k, s) for s in sentences(body) if not s.startswith(("---", "title:", "updated:", "#", "|"))]
    if not cands:
        return REFUSAL
    sims = embedder.embed([s for _, s in cands]) @ embedder.query_vector(question)
    wanted = codes_in(question)
    # Prefer sentences from the excerpt that names the asked-for error code, then from higher-ranked excerpts.
    score = [float(s) + (0.5 if wanted & codes_in(docs[k - 1].page_content) else 0) - 0.03 * (k - 1)
             for s, (k, _) in zip(sims, cands)]
    best = sorted(sorted(range(len(cands)), key=lambda i: -score[i])[:n])
    return " ".join(f"{cands[i][1]} [{cands[i][0]}]" for i in best)


# ---------- pipelines ----------

def _history(history: list[dict]) -> list:
    return [HumanMessage(h["content"]) if h["role"] == "user" else AIMessage(h["content"]) for h in history[-6:]]


def _context(docs: list[Document], numbered: bool) -> list[str]:
    if not numbered:
        return [d.page_content for d in docs]
    return [f"[{i}] {d.metadata['title']} > {d.metadata['section']} (updated {d.metadata['updated']})\n"
            + d.page_content.split("\n", 1)[-1] for i, d in enumerate(docs, 1)]


def _sources(docs: list[Document], answer: str) -> list[dict]:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    keep = ("title", "section", "source", "anchor", "updated", "dense", "dense_rank", "bm25_rank", "found_by", "exact_code", "rrf")
    return [{"n": i, **{k: d.metadata.get(k) for k in keep}, "cited": i in cited, "text": d.page_content}
            for i, d in enumerate(docs, 1)]


class RagService:
    def __init__(self):
        self.embedder = BgeSmallEmbeddings()
        if not (INDEX_DIR / "fixed.npy").exists():
            build_index(self.embedder)
        self.fixed_index, self.naive_index = load_index("fixed"), load_index("naive")
        self.hybrid = HybridRetriever(index=self.fixed_index, embedder=self.embedder)
        self.vocab = {p for d in self.fixed_index.docs for t in tokens(d.page_content) for p in [t, *re.split(r"[-.]", t)]}
        self.dense = DenseRetriever(index=self.naive_index, embedder=self.embedder)
        self.llm = make_llm()

    @property
    def model(self) -> str:
        return self.llm.name if self.llm else "none (extractive answers)"

    def meta(self) -> dict:
        return {"llm": self.llm is not None, "model": self.model, "models": self.llm.models if self.llm else [],
                "embeddings": MODEL_NAME, "relevance_floor": GATE,
                "chunks": {"fixed": len(self.fixed_index.docs), "naive": len(self.naive_index.docs)},
                "docs": [{"source": d.metadata["source"], "title": d.metadata["title"], "updated": d.metadata["updated"]} for d in load_kb()]}

    def ask(self, message: str, history: list[dict] | None = None, pipeline: str = "fixed") -> dict:
        t0 = time.perf_counter()
        self._used, self._notice = None, None
        out = self._fixed(message, history or []) if pipeline == "fixed" else self._naive(message, history or [])
        out.update(pipeline=pipeline, model=self._used or "extractive", mode="llm" if self._used else "extractive",
                   notice=self._notice)
        out["timings"]["total_ms"] = round((time.perf_counter() - t0) * 1000)
        return out

    def _naive(self, message: str, history: list[dict]) -> dict:
        t = time.perf_counter()
        docs = self.dense.invoke(message)  # the latest message only: follow-ups lose their subject
        tr = round((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        answer = self._llm(NAIVE_PROMPT, {"history": _history(history), "context": "\n\n".join(_context(docs, False)), "question": message})
        if answer is None:
            answer = extractive(message, docs, self.embedder, n=2, top_docs=1, crumb=False)
            answer = re.sub(r"\s*\[\d+\]", "", answer)
        tg = round((time.perf_counter() - t) * 1000)
        checks = check_answer(answer, _context(docs, False), message, require_citations=False)
        for c in checks:
            c["detail"] += " (Diagnostic only: this pipeline doesn't act on it.)"
        return {"answer": answer.strip(), "standalone_question": message, "refused": is_refusal(answer), "gate": None,
                "sources": _sources(docs, ""), "checks": checks, "attempts": 1, "timings": {"retrieve_ms": tr, "generate_ms": tg}}

    def _llm(self, prompt, inputs: dict) -> str | None:
        """One model call through the fallback chain. None means: answer without a model this time."""
        if not self.llm:
            return None
        try:
            text, used = self.llm.invoke(prompt, inputs)
            self._used = used
            return text
        except LLMUnavailable as e:
            self._notice = f"Showing an extractive answer: {e}."
            return None

    def unknown_names(self, text: str) -> list[str]:
        """Capitalised words and acronyms (not sentence-initial) that appear nowhere in the knowledge base."""
        out = []
        for m in re.finditer(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-zA-Z]+|[A-Z]{2,})\b", text.strip()):
            word = m.group(1)
            if word.lower() not in GENERIC and not set(tokens(word)) & self.vocab:
                out.append(word)
        return out

    @staticmethod
    def _fuse(a: list[Document], b: list[Document], k: int = 5) -> list[Document]:
        score: dict[str, float] = {}
        keep: dict[str, Document] = {}
        for lst in (a, b):
            for r, d in enumerate(lst):
                score[d.metadata["id"]] = score.get(d.metadata["id"], 0) + 1 / (60 + r + 1)
                keep.setdefault(d.metadata["id"], d)
        return [keep[i] for i in sorted(score, key=lambda i: -score[i])[:k]]

    def condense(self, message: str, history: list[dict]) -> str:
        if not history:
            return message
        rewritten = self._llm(CONDENSE_PROMPT, {"history": _history(history), "question": message})
        if rewritten:
            return rewritten
        # Without a model: carry the previous question's subject into a short follow-up.
        last = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
        return f"{last} {message}" if len(message.split()) <= 8 else message

    def _fixed(self, message: str, history: list[dict]) -> dict:
        t = time.perf_counter()
        standalone = normalise_codes(self.condense(message, history))
        docs = self.hybrid.invoke(standalone)
        if history and not self.llm and standalone != message:
            # No model to rewrite the follow-up: fuse results for the combined and the bare message,
            # so "What about SCIM?" isn't drowned out by the previous question's subject.
            docs = self._fuse(docs, self.hybrid.invoke(normalise_codes(message)))
        tr = round((time.perf_counter() - t) * 1000)
        best = max((d.metadata["dense"] for d in docs), default=0.0)
        asked_codes = codes_in(standalone)
        found_codes = set().union(*(codes_in(d.page_content) for d in docs)) if docs else set()
        unknown = sorted(asked_codes - found_codes)
        # Names and acronyms the knowledge base never mentions ("Salesforce", "CEO").
        foreign = self.unknown_names(message)
        gate = {"best_similarity": round(best, 3), "floor": GATE, "exact_code_match": any(d.metadata.get("exact_code") for d in docs),
                "names_not_in_kb": foreign}
        base = {"standalone_question": standalone, "gate": gate, "timings": {"retrieve_ms": tr, "generate_ms": 0}, "attempts": 0}

        # With a model, the cite-or-refuse prompt judges these; without one, an unknown name refuses.
        off_topic = bool(foreign) and not self.llm
        if unknown or off_topic or (best < GATE and not gate["exact_code_match"]):
            why = (f"No article mentions {', '.join(unknown)}." if unknown else
                   f"No article mentions {', '.join(foreign)}." if off_topic else
                   f"Best excerpt similarity {best:.2f} is under the {GATE:.2f} floor.")
            gate["passed"] = False
            return {**base, "answer": REFUSAL, "refused": True, "sources": _sources(docs, ""),
                    "checks": [{"name": "Relevance gate", "ok": True, "detail": why + " The model was not called."}]}
        gate["passed"] = True

        ctx = _context(docs, True)
        t = time.perf_counter()
        inputs = {"context": "\n\n".join(ctx), "question": standalone}
        answer = self._llm(FIXED_PROMPT, inputs)
        if answer is None:
            answer = extractive(standalone, docs, self.embedder)
            checks, attempts = check_answer(answer, ctx, standalone), 0
        else:
            attempts = 1
            checks = check_answer(answer, ctx, standalone)
            if not all(c["ok"] for c in checks):
                problem = "; ".join(f"{c['name']}: {c['detail']}" for c in checks if not c["ok"])
                retry = ChatPromptTemplate.from_messages([*FIXED_PROMPT.messages, ("ai", "{previous}"), ("human", RETRY_NOTE)])
                answer = self._llm(retry, {**inputs, "previous": answer, "problem": problem}) or REFUSAL
                attempts = 2
                checks = check_answer(answer, ctx, standalone)
                if not all(c["ok"] for c in checks):
                    checks.append({"name": "Unverified answer withheld", "ok": True,
                                   "detail": "Two attempts failed the checks, so the bot refused rather than show it."})
                    answer = REFUSAL
        base["timings"]["generate_ms"] = round((time.perf_counter() - t) * 1000)
        return {**base, "answer": answer, "refused": is_refusal(answer), "sources": _sources(docs, answer),
                "checks": checks, "attempts": attempts}
