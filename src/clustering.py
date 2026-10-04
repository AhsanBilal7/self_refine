"""Semantic clustering of reasoning traces (sampling-redundancy analysis).

Implements the clustering of the paper's appendix "Sampling Redundancy
Metrics": each initial rollout is embedded with a sentence encoder
(``all-MiniLM-L6-v2`` by default), embeddings are unit-normalised, and
agglomerative clustering with average linkage and cosine distance is cut at
``1 - tau_sim`` (``tau_sim = 0.85``). ``redundancy_metrics`` turns the cluster
labels into the realised cluster count R(N), R(N)/N, the dominant-cluster
mass f_max(N) and the entropy H(N).

The clustering is analysis-only and does not affect the method's predictions.
If ``sentence-transformers`` is unavailable, a mean-pooled Hugging Face
encoder is used, then TF-IDF.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Clustering back-ends
# ──────────────────────────────────────────────────────────────────────────────

def _cluster_agglomerative(
    embeddings: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine distance."""
    from sklearn.cluster import AgglomerativeClustering  # type: ignore
    n = len(embeddings)
    if n == 1:
        return np.array([0])
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    return clustering.fit_predict(embeddings).astype(int)


def _cluster_threshold_greedy(
    embeddings: np.ndarray,
    similarity_threshold: float,
) -> np.ndarray:
    """Single-pass clustering: join the most similar centroid above threshold."""
    n = len(embeddings)
    labels = -np.ones(n, dtype=int)
    centroids: List[np.ndarray] = []
    cluster_sizes: List[int] = []

    for i in range(n):
        if not centroids:
            labels[i] = 0
            centroids.append(embeddings[i].copy())
            cluster_sizes.append(1)
            continue
        sims = np.array([embeddings[i] @ c for c in centroids])
        best = int(np.argmax(sims))
        if sims[best] >= similarity_threshold:
            labels[i] = best
            sz = cluster_sizes[best]
            centroids[best] = (centroids[best] * sz + embeddings[i]) / (sz + 1)
            centroids[best] /= np.linalg.norm(centroids[best]) + 1e-12
            cluster_sizes[best] += 1
        else:
            new_id = len(centroids)
            labels[i] = new_id
            centroids.append(embeddings[i].copy())
            cluster_sizes.append(1)

    return labels


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF fallback  (stateless, no persistent model needed)
# ──────────────────────────────────────────────────────────────────────────────

def _embed_tfidf(texts: List[str]) -> np.ndarray:
    """
    TF-IDF bag-of-words embeddings with full robustness against:
      - empty strings
      - strings that consist entirely of stop words
      - a corpus where every document is empty after tokenisation
        (→ 'empty vocabulary' ValueError from sklearn)

    Strategy
    --------
    1. Replace every blank/whitespace-only text with a unique sentinel
       so sklearn always has something to tokenise.
    2. Try TfidfVectorizer with stop_word filtering first.
    3. If the vocabulary is still empty (all tokens were stop words),
       retry without stop-word removal so at least *something* remains.
    4. If that also fails (all texts were truly empty after cleaning),
       return an identity-like matrix of uniform unit vectors.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    # Step 1 — replace empty / whitespace-only texts with a sentinel
    # Use index-based sentinels so each empty doc gets a unique token
    # (prevents them from all collapsing to the same cluster).
    cleaned: List[str] = []
    for i, t in enumerate(texts):
        s = t.strip()
        cleaned.append(s if s else f"__empty_doc_{i}__")

    n = len(cleaned)

    def _vectorize(stop_words):
        vec = TfidfVectorizer(max_features=4096, sublinear_tf=True,
                              stop_words=stop_words)
        X = vec.fit_transform(cleaned).toarray().astype(np.float32)
        return X

    # Step 2 — try with English stop words
    try:
        X = _vectorize("english")
        if X.shape[1] > 0:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return X / norms
    except ValueError:
        pass

    # Step 3 — retry without stop-word removal
    try:
        X = _vectorize(None)
        if X.shape[1] > 0:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return X / norms
    except ValueError:
        pass

    # Step 4 — last resort: one-hot unit vectors, so every document is its
    # own cluster.
    logger.warning(
        "TF-IDF: all documents empty after cleaning; "
        "returning identity-like unit embeddings."
    )
    d = max(n, 1)
    X = np.eye(n, d, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ──────────────────────────────────────────────────────────────────────────────
# ClusterManager
# ──────────────────────────────────────────────────────────────────────────────

class ClusterManager:
    """Cluster the traces of one problem into reasoning directions.

    The embedding model is loaded once in ``__init__`` and reused by every
    ``cluster()`` call.

    Parameters
    ----------
    embedding_model      : sentence-transformers id, or ``'tfidf'``.
    similarity_threshold : ``tau_sim``; clusters are cut at ``1 - tau_sim``.
    clustering_method    : ``'agglomerative'`` (paper) or ``'greedy'``.
    use_answer_only      : embed only the ``\\boxed{}`` answer, not the trace.
    batch_size           : encoder batch size.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        clustering_method: str = "agglomerative",
        use_answer_only: bool = False,
        batch_size: int = 32,
    ):
        self.embedding_model      = embedding_model
        self.similarity_threshold = similarity_threshold
        self.distance_threshold   = 1.0 - similarity_threshold
        self.clustering_method    = clustering_method
        self.use_answer_only      = use_answer_only
        self.batch_size           = batch_size   # used by HF AutoModel _embed path

        self._st_model   = None   # SentenceTransformer instance
        self._hf_model   = None   # HF AutoModel instance
        self._hf_tok     = None   # HF AutoTokenizer instance
        self._hf_device  = None
        self._embedder   = self._load_embedder()  # sets the fields above

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_embedder(self) -> str:
        """Load the encoder; returns 'sentence_transformers' | 'hf_transformer' | 'tfidf'."""
        if self.embedding_model == "tfidf":
            logger.info("ClusterManager: using TF-IDF embedder (no model load).")
            return "tfidf"

        # Try sentence-transformers first (best quality / easiest API)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info(
                f"ClusterManager: loading SentenceTransformer "
                f"'{self.embedding_model}'."
            )
            self._st_model = SentenceTransformer(self.embedding_model)
            logger.info("ClusterManager: SentenceTransformer ready.")
            return "sentence_transformers"
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; trying HF AutoModel fallback."
            )
        except Exception as e:
            logger.warning(f"SentenceTransformer load failed ({e}); trying HF fallback.")

        # Try HF AutoModel fallback
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            logger.info(
                f"ClusterManager: loading HF AutoModel "
                f"'{self.embedding_model}'."
            )
            self._hf_tok    = AutoTokenizer.from_pretrained(self.embedding_model)
            self._hf_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._hf_model  = AutoModel.from_pretrained(
                self.embedding_model
            ).to(self._hf_device).eval()
            logger.info("ClusterManager: HF AutoModel ready.")
            return "hf_transformer"
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"HF AutoModel load failed ({e}); using TF-IDF fallback.")

        logger.warning(
            "ClusterManager: no neural embedder available — using TF-IDF fallback."
        )
        return "tfidf"

    # ── embedding  (uses the already-loaded model) ────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using the pre-loaded model. Never reloads anything.

        Robustness: empty or whitespace-only strings are replaced with a
        unique sentinel before embedding so downstream vectorisers never
        receive an empty-vocabulary corpus.  The sentinel is index-based so
        each blank document gets a distinct token, keeping them maximally
        separate rather than all mapping to the same embedding.
        """
        # Sanitise: replace blank texts with unique sentinels
        safe_texts: List[str] = [
            t if t.strip() else f"__empty_{i}__"
            for i, t in enumerate(texts)
        ]

        if self._embedder == "sentence_transformers":
            try:
                return self._st_model.encode(
                    safe_texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer.encode failed ({e}); "
                    "falling back to TF-IDF."
                )
                return _embed_tfidf(safe_texts)

        if self._embedder == "hf_transformer":
            import torch
            all_embs = []
            try:
                for i in range(0, len(safe_texts), self.batch_size):
                    batch = safe_texts[i : i + self.batch_size]
                    enc = self._hf_tok(
                        batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt"
                    ).to(self._hf_device)
                    with torch.no_grad():
                        out = self._hf_model(**enc)
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                    emb  = torch.nn.functional.normalize(emb, dim=-1)
                    all_embs.append(emb.cpu().numpy())
                return np.vstack(all_embs)
            except Exception as e:
                logger.warning(
                    f"HF AutoModel.encode failed ({e}); falling back to TF-IDF."
                )
                return _embed_tfidf(safe_texts)

        # TF-IDF — stateless, no model to cache
        return _embed_tfidf(safe_texts)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_texts_to_embed(self, traces: List[str]) -> List[str]:
        if not self.use_answer_only:
            return traces
        from .verifier import extract_boxed_answer
        return [extract_boxed_answer(t) or t for t in traces]

    # ── public API ────────────────────────────────────────────────────────────

    def cluster(self, traces: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster the traces of one problem into reasoning directions.

        Returns
        -------
        labels        : np.ndarray shape (N,)  — integer cluster id per trace
        cluster_sizes : np.ndarray shape (N,)  — size of each trace's cluster
        """
        n = len(traces)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int), np.array([1], dtype=int)

        texts      = self._get_texts_to_embed(traces)
        embeddings = self._embed(texts)

        if self.clustering_method == "agglomerative":
            try:
                labels = _cluster_agglomerative(embeddings, self.distance_threshold)
            except Exception as e:
                logger.warning(f"Agglomerative failed ({e}); using greedy.")
                labels = _cluster_threshold_greedy(embeddings, self.similarity_threshold)
        else:
            labels = _cluster_threshold_greedy(embeddings, self.similarity_threshold)

        unique, counts = np.unique(labels, return_counts=True)
        size_map       = dict(zip(unique.tolist(), counts.tolist()))
        cluster_sizes  = np.array([size_map[int(l)] for l in labels], dtype=int)

        return labels, cluster_sizes


def redundancy_metrics(labels: np.ndarray) -> Dict[str, float]:
    """Sampling-redundancy statistics for the cluster labels of N rollouts.

    Returns ``R`` (realised clusters), ``R_over_N``, ``f_max`` (largest
    cluster mass), ``H`` (entropy in bits) and ``H_norm`` (``H / log2 N``).
    """
    labels = np.asarray(labels)
    N = len(labels)
    if N == 0:
        return dict(R=0, R_over_N=0.0, f_max=1.0, H=0.0, H_norm=0.0, N=0)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / N
    H = float(-np.sum(p * np.log2(p + 1e-12)))
    return dict(
        R=int(len(counts)),
        R_over_N=float(len(counts) / N),
        f_max=float(counts.max() / N),
        H=H,
        H_norm=float(H / np.log2(N)) if N > 1 else 0.0,
        N=N,
    )
