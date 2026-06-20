from typing import Optional

from learning.vector_store import VectorStore
from learning.embedder import LocalEmbedder
from learning.example_builder import FewShotBuilder
from learning.config import RAG_TOP_K, RAG_MIN_PROFILE_CLIPS, RAG_RERANK_WEIGHTS


class RAGInjector:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: LocalEmbedder,
        example_builder: FewShotBuilder,
    ):
        self.vs = vector_store
        self.embedder = embedder
        self.builder = example_builder
        self._cache = {}

    def get_few_shot(
        self,
        transcript: str,
        profile_id: str,
        fallback_category: Optional[str] = None,
        top_k: int = None,
    ) -> str:
        k = top_k or RAG_TOP_K

        # Embed the current transcript
        query_embedding = self.embedder.embed_query(transcript[:3000])

        # Query by profile
        results = self.vs.query_by_profile(query_embedding, profile_id, k=k)

        # Fallback to category if not enough results
        if len(results) < RAG_MIN_PROFILE_CLIPS and fallback_category:
            cat_results = self.vs.query_by_category(query_embedding, fallback_category, k=k)
            results = self._merge_and_rerank(results, cat_results)

        if not results:
            return ""

        # Rerank results
        results = self._rerank(results, query_embedding)

        return self.builder.build_examples(results, transcript)

    def _rerank(self, results: list, query_embedding: list) -> list:
        w_sim, w_views, w_eng = RAG_RERANK_WEIGHTS

        max_views = 1
        max_eng = 0.001
        for r in results:
            meta = r.get("metadata", {})
            max_views = max(max_views, int(meta.get("views", 0)))
            max_eng = max(max_eng, float(meta.get("engagement_rate", 0)))

        scored = []
        for r in results:
            meta = r.get("metadata", {})
            similarity = r.get("similarity", 0.5)
            norm_views = int(meta.get("views", 0)) / max_views
            norm_eng = float(meta.get("engagement_rate", 0)) / max_eng

            score = (
                w_sim * (1 - similarity)  # distance → similarity
                + w_views * norm_views
                + w_eng * norm_eng
            )
            scored.append((score, r))

        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored]

    def _merge_and_rerank(self, profile_results: list, category_results: list) -> list:
        seen = set()
        merged = []

        for r in profile_results:
            cid = r["clip_id"]
            if cid not in seen:
                seen.add(cid)
                merged.append(r)

        for r in category_results:
            cid = r["clip_id"]
            if cid not in seen:
                seen.add(cid)
                merged.append(r)

        return merged
