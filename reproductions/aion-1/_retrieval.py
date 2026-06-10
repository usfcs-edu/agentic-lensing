"""
Semantic retrieval harness for AION-1 retrieval tasks (paper tasks 7-9:
GZ-DECaLS spirals/mergers, HSC strong lenses).

Protocol (matches the paper): embed the whole corpus with the frozen encoder,
L2-normalise, and for each query object rank the rest of the corpus by cosine
similarity. Relevance of a retrieved neighbour = whether it belongs to the same
rare positive class as the query. Report mean nDCG@10 over all queries. Cosine
scoring is chunked on the GPU so a ~300k-object corpus fits in memory.
"""

from __future__ import annotations

import numpy as np

from _metrics import mean_ndcg_at_k


def retrieval_ndcg(embeddings: np.ndarray, is_positive: np.ndarray, *, k: int = 10,
                   query_indices=None, device: str = "cuda", chunk: int = 4096) -> dict:
    """Compute mean nDCG@k for class-consistent retrieval.

    embeddings : (N, D) float array (mean-pooled frozen-encoder features).
    is_positive: (N,) bool/int -- membership in the rare positive class.
    query_indices: which rows to use as queries (default: all positives).
    Returns {"ndcg@k", "n_queries", "corpus", "n_positive"}.
    """
    import torch

    E = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device)
    E = torch.nn.functional.normalize(E, dim=1)
    pos = np.asarray(is_positive).astype(bool)
    N = E.shape[0]
    if query_indices is None:
        query_indices = np.where(pos)[0]
    query_indices = np.asarray(query_indices)

    pos_t = torch.as_tensor(pos.astype(np.float32), device=device)
    ranked_rel = []
    for s in range(0, len(query_indices), chunk):
        qi = query_indices[s : s + chunk]
        q = E[torch.as_tensor(qi, device=device)]            # (b, D)
        sims = q @ E.T                                        # (b, N)
        # exclude the query itself from its own ranking
        sims[torch.arange(len(qi), device=device), torch.as_tensor(qi, device=device)] = -1e9
        topk = torch.topk(sims, k=k, dim=1).indices           # (b, k)
        rel = pos_t[topk]                                     # (b, k) binary relevance
        ranked_rel.extend(rel.cpu().numpy())
    return {
        "ndcg@%d" % k: mean_ndcg_at_k(ranked_rel, k),
        "n_queries": int(len(query_indices)),
        "corpus": int(N),
        "n_positive": int(pos.sum()),
    }
