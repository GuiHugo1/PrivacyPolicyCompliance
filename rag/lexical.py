"""Dependency-free Okapi BM25 lexical index over the same chunks as the vector store.

This is the second leg of hybrid retrieval (see ``rag.retriever.retrieve(...,
hybrid=True)``): dense cosine search and this BM25 pass are computed
independently over the same corpus and merged with reciprocal rank fusion
(``rag.fusion``). The point is to catch near-verbatim ("easy") queries that
share exact statutory vocabulary with their gold chunk even when the dense
embedding happens to land slightly off, without weakening the dense signal
that paraphrased ("hard") queries rely on.

Kept free of any third-party dependency (plain ``math``/``re``/
``collections``) so it's cheap to unit test and doesn't add a new package to
the ``rag`` dependency group just for this.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-run tokenization -- good enough for BM25 term
    matching over statutory/policy English without pulling in a tokenizer."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """In-memory Okapi BM25 index over a fixed corpus of (id, text) pairs."""

    ids: list[str]
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    _doc_freqs: list[Counter] = field(default_factory=list, repr=False)
    _doc_lengths: list[int] = field(default_factory=list, repr=False)
    _avg_doc_length: float = field(default=0.0, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls, ids: list[str], texts: list[str], k1: float = DEFAULT_K1, b: float = DEFAULT_B
    ) -> BM25Index:
        index = cls(ids=list(ids), k1=k1, b=b)
        doc_tokens = [tokenize(t) for t in texts]
        index._doc_lengths = [len(toks) for toks in doc_tokens]
        n_docs = len(doc_tokens)
        index._avg_doc_length = (sum(index._doc_lengths) / n_docs) if n_docs else 0.0

        index._doc_freqs = [Counter(toks) for toks in doc_tokens]
        df: Counter[str] = Counter()
        for freqs in index._doc_freqs:
            df.update(freqs.keys())
        # BM25 IDF (Robertson-Sparck Jones with the +1 smoothing term so a
        # term appearing in every document still gets a small positive
        # weight rather than going negative).
        index._idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }
        return index

    @classmethod
    def from_collection(cls, collection, where: dict | None = None) -> BM25Index:
        """Build an index from every document currently in a Chroma
        collection (optionally scoped to a ``where`` filter, matching the
        same filter semantics as ``rag.retriever.retrieve``)."""
        data = collection.get(where=where, include=["documents"])
        ids = data.get("ids", []) or []
        texts = data.get("documents", []) or []
        return cls.build(ids, texts)

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        """Return the top-n (id, bm25_score) pairs for ``query``, descending
        by score. Documents with zero term overlap are excluded rather than
        padded in with a meaningless zero score."""
        query_terms = tokenize(query)
        if not query_terms or not self.ids:
            return []

        scores = [0.0] * len(self.ids)
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self._doc_freqs):
                freq = freqs.get(term)
                if not freq:
                    continue
                doc_len = self._doc_lengths[i]
                denom = freq + self.k1 * (
                    1 - self.b + self.b * doc_len / (self._avg_doc_length or 1)
                )
                scores[i] += idf * (freq * (self.k1 + 1)) / denom

        ranked = sorted(zip(self.ids, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [(id_, score) for id_, score in ranked if score > 0][:n]
