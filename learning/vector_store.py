import os
import json
from typing import List, Optional, Dict
import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, persist_dir: str = None):
        from learning.config import LEARNING_DATA_DIR

        self.persist_dir = persist_dir or LEARNING_DATA_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="clip_feedback",
            metadata={"hnsw:space": "cosine"},
        )

    def add_clip(self, clip_id: str, embedding: List[float], metadata: Dict):
        self.collection.add(
            ids=[clip_id],
            embeddings=[embedding],
            metadatas=[self._sanitize_metadata(metadata)],
        )

    def update_clip(self, clip_id: str, embedding: List[float] = None, metadata: Dict = None):
        if embedding and metadata:
            self.collection.update(
                ids=[clip_id],
                embeddings=[embedding],
                metadatas=[self._sanitize_metadata(metadata)],
            )
        elif embedding:
            self.collection.update(ids=[clip_id], embeddings=[embedding])
        elif metadata:
            self.collection.update(ids=[clip_id], metadatas=[self._sanitize_metadata(metadata)])

    def delete_clip(self, clip_id: str):
        self.collection.delete(ids=[clip_id])

    def query_by_profile(
        self, embedding: List[float], profile_id: str, k: int = 10
    ) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"profile_id": profile_id},
        )
        return self._format_results(results)

    def query_by_category(
        self, embedding: List[float], category: str, k: int = 10
    ) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"base_category": category},
        )
        return self._format_results(results)

    def get_profile_clips(self, profile_id: str) -> List[Dict]:
        results = self.collection.get(where={"profile_id": profile_id})
        return self._format_get_results(results)

    def get_profile_stats(self, profile_id: str) -> Dict:
        clips = self.get_profile_clips(profile_id)
        if not clips:
            return {"clip_count": 0, "avg_engagement": 0.0, "top_patterns": []}

        rates = []
        pattern_counts = {}
        for c in clips:
            meta = c.get("metadata", {})
            rate = float(meta.get("engagement_rate", 0))
            rates.append(rate)
            pat = meta.get("viral_pattern", "unknown")
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])[:5]

        return {
            "clip_count": len(clips),
            "avg_engagement": round(sum(rates) / len(rates), 4) if rates else 0.0,
            "top_patterns": [{"pattern": p, "count": c} for p, c in sorted_patterns],
        }

    def count_by_profile(self, profile_id: str) -> int:
        return self.collection.count(where={"profile_id": profile_id})

    def rebuild_index(self):
        all_data = self.collection.get()
        print(f"  [VectorStore] Collection has {len(all_data.get('ids', []))} entries")

    def _sanitize_metadata(self, meta: Dict) -> Dict:
        sanitized = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif v is None:
                continue
            else:
                sanitized[k] = str(v)
        return sanitized

    def _format_results(self, results) -> List[Dict]:
        formatted = []
        if not results or not results.get("ids"):
            return formatted

        for i, clip_id in enumerate(results["ids"][0]):
            entry = {
                "clip_id": clip_id,
                "similarity": float(results["distances"][0][i]) if results.get("distances") else 0.0,
                "metadata": {},
            }
            if results.get("metadatas") and len(results["metadatas"][0]) > i:
                entry["metadata"] = results["metadatas"][0][i]
            formatted.append(entry)
        return formatted

    def _format_get_results(self, results) -> List[Dict]:
        formatted = []
        if not results or not results.get("ids"):
            return formatted

        for i, clip_id in enumerate(results["ids"]):
            entry = {
                "clip_id": clip_id,
                "metadata": results["metadatas"][i] if results.get("metadatas") else {},
            }
            formatted.append(entry)
        return formatted
