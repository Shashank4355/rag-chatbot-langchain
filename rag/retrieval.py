"""Retrievers. Both are LangChain BaseRetrievers, so they drop into any chain.

DenseRetriever: cosine top-k over the vectors, nothing else (the naive setup).
HybridRetriever: dense + BM25, fused with reciprocal rank fusion, plus an
exact-match boost for error codes. Embeddings place "KB-412" and "KB-204"
almost on top of each other; BM25 and the exact match don't.
"""
import re

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

STOP = set("a an and are as at be by can do does for from has have how i if in is it its me my of on or so that the this to was what when where which who why will with you your".split())
# Error-code families in this knowledge base. A client's bot gets its own list.
CODE = re.compile(r"\b(KB|SSO)[\s-]?(\d{3})\b", re.I)


def normalise_codes(text: str) -> str:
    """'kb412', 'kb 412' and 'KB-412' all become 'KB-412'."""
    return CODE.sub(lambda m: f"{m.group(1).upper()}-{m.group(2)}", text)


def codes_in(text: str) -> set[str]:
    return {f"{a.upper()}-{b}" for a, b in CODE.findall(text)}


def tokens(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", normalise_codes(text).lower()):
        if t in STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class VectorIndex:
    def __init__(self, docs: list[Document], vectors: np.ndarray):
        self.docs, self.vectors = docs, vectors

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        sims = self.vectors @ qvec
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


def _with(doc: Document, **extra) -> Document:
    return Document(page_content=doc.page_content, metadata={**doc.metadata, **extra})


class DenseRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: VectorIndex
    embedder: object
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        hits = self.index.search(self.embedder.query_vector(query), self.k)
        return [_with(self.index.docs[i], dense=round(s, 3), dense_rank=r + 1, found_by="dense") for r, (i, s) in enumerate(hits)]


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: VectorIndex
    embedder: object
    bm25: BM25Okapi | None = None
    k: int = 5
    depth: int = 20
    rrf_k: int = 60

    def model_post_init(self, _ctx) -> None:
        if self.bm25 is None:
            self.bm25 = BM25Okapi([tokens(d.page_content) for d in self.index.docs])

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        query = normalise_codes(query)
        qvec = self.embedder.query_vector(query)
        all_sims = self.index.vectors @ qvec
        dense = [int(i) for i in np.argsort(-all_sims)[: self.depth]]
        bm = self.bm25.get_scores(tokens(query))
        sparse = [int(i) for i in np.argsort(-bm)[: self.depth] if bm[i] > 0]
        fused: dict[int, float] = {}
        for ranks in (dense, sparse):
            for r, i in enumerate(ranks):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
        wanted = codes_in(query)
        exact = {i for i in fused if wanted and wanted & codes_in(self.index.docs[i].page_content)}
        order = sorted(fused, key=lambda i: (i not in exact, -fused[i]))[: self.k]
        out = []
        for i in order:
            by = [n for n, lst in (("dense", dense), ("bm25", sparse)) if i in lst]
            out.append(_with(self.index.docs[i], dense=round(float(all_sims[i]), 3),
                             dense_rank=dense.index(i) + 1 if i in dense else None,
                             bm25_rank=sparse.index(i) + 1 if i in sparse else None,
                             rrf=round(fused[i], 4), exact_code=i in exact, found_by="+".join(by) or "exact"))
        return out
