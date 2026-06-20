import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from typing import List, Optional
import numpy as np


class LocalEmbedder:
    def __init__(self, model_name: str = None, use_ollama: bool = None, ollama_host: str = None):
        from learning.config import EMBEDDING_MODEL, USE_OLLAMA, OLLAMA_HOST

        self.model_name = model_name or EMBEDDING_MODEL
        self.use_ollama = use_ollama if use_ollama is not None else USE_OLLAMA
        self.ollama_host = ollama_host or OLLAMA_HOST
        self._model = None
        self._dimension = 384

        if self.use_ollama:
            import ollama
            self._ollama_client = ollama.Client(host=self.ollama_host)
            self._pull_if_needed("nomic-embed-text")
            self._dimension = 768

    def _pull_if_needed(self, model: str):
        try:
            import ollama
            self._ollama_client.list()
        except Exception as e:
            print(f"  [Embedder] Ollama not available: {e}")
            print("  [Embedder] Falling back to sentence-transformers")
            self.use_ollama = False

    def _get_or_load_model(self):
        if self.use_ollama:
            return None
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  [Embedder] Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            try:
                self._dimension = self._model.get_embedding_dimension()
            except AttributeError:
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.use_ollama:
            results = []
            for text in texts:
                resp = self._ollama_client.embeddings(
                    model="nomic-embed-text", prompt=text
                )
                results.append(resp["embedding"])
            return results
        else:
            model = self._get_or_load_model()
            embeddings = model.encode(texts, show_progress_bar=False)
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension
