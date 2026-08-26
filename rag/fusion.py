"""Score-fusion utilities for combining multiple ranked-id lists into one.

Used to merge the dense-cosine and BM25 ranked lists in hybrid retrieval
(``rag.retriever.retrieve(..., hybrid=True)``), but kept generic -- it only
operates on plain id lists -- so it's trivial to unit test and reusable if a
third retrieval signal is ever added.
"""

from __future__ import annotations

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, float]:
    """Merge several rank-ordered id lists into one fused score per id.

    Standard reciprocal rank fusion: ``score(id) = sum_over_lists(weight /
    (rrf_k + rank))``, where ``rank`` is 1-based. ``rrf_k`` dampens the
    influence of any single list's very top ranks so one retrieval signal
    can't dominate purely by placing an item 1st -- the default of 60 is the
    standard value from the original RRF paper (Cormack et al., 2009).

    An id that appears in only some of the lists still gets scored (from the
    lists it does appear in); it is not penalized beyond simply not
    accumulating a contribution from the lists it's absent from.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must be the same length as ranked_lists")

    fused: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for rank, id_ in enumerate(ranked, start=1):
            fused[id_] = fused.get(id_, 0.0) + weight / (rrf_k + rank)
    return fused
