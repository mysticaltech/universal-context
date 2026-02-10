"""Embedding providers for semantic search.

Supports local (EmbeddingGemma via ONNX) and cloud (OpenAI/OpenRouter) providers.
Local is the default — no API keys required.

Provider resolution:
  - "local"      → EmbeddingGemma-300M via ONNX Runtime (default, no API key)
  - "openai"     → requires OPENAI_API_KEY
  - "openrouter" → requires OPENROUTER_API_KEY
  - "auto"       → local first, then OpenAI, then OpenRouter
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import UCConfig

logger = logging.getLogger(__name__)

# Known dimensions per model (for auto-detection without probing)
_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class EmbedProvider(ABC):
    """Embedding provider protocol with query/document distinction."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension for this provider."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""

    @abstractmethod
    async def embed_document(self, text: str) -> list[float]:
        """Embed a document for storage."""


class LocalEmbedProvider(EmbedProvider):
    """EmbeddingGemma-300M via ONNX Runtime — no API keys needed.

    Downloads the model on first use (~600MB, cached in ~/.cache/huggingface/).
    Uses task-specific prefixes for optimal asymmetric retrieval.
    """

    DEFAULT_MODEL = "onnx-community/embeddinggemma-300m-ONNX"

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self._model_id = model_id
        self._session: object | None = None
        self._tokenizer: object | None = None

    @property
    def dim(self) -> int:
        return 768

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        import os
        from pathlib import Path

        # Suppress misleading "PyTorch was not found" warning from transformers.
        # We use ONNX Runtime for inference — PyTorch is never needed.
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        logger.info("Loading local embedding model: %s", self._model_id)

        # local_dir copies real files (not symlinks) — required because
        # ONNX Runtime validates that model.onnx_data is co-located with
        # model.onnx, and HuggingFace's default cache uses symlinks to a
        # content-addressed blob store where they resolve to different dirs.
        cache_dir = Path.home() / ".cache" / "uc-models" / "embeddinggemma"
        model_dir = snapshot_download(
            self._model_id,
            local_dir=str(cache_dir),
            allow_patterns=["onnx/*", "tokenizer*", "special_tokens*"],
        )
        model_path = os.path.join(model_dir, "onnx", "model.onnx")

        self._session = ort.InferenceSession(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info("Local embedding model loaded (dim=%d)", self.dim)

    async def embed_query(self, text: str) -> list[float]:
        return await self._embed(f"task: search result | query: {text}")

    async def embed_document(self, text: str) -> list[float]:
        return await self._embed(f"title: none | text: {text}")

    async def _embed(self, text: str) -> list[float]:
        self._ensure_loaded()

        def _run() -> list[float]:
            inputs = self._tokenizer(
                [text], padding=True, truncation=True,
                max_length=2048, return_tensors="np",
            )
            _, sentence_embedding = self._session.run(None, dict(inputs))
            return sentence_embedding[0].tolist()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run)


class OpenAIEmbedProvider(EmbedProvider):
    """OpenAI API embeddings (direct or via OpenRouter)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        from openai import AsyncOpenAI

        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            **({"base_url": base_url} if base_url else {}),
        )
        self._dim = _MODEL_DIMS.get(model, 0)

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_query(self, text: str) -> list[float]:
        return await self._embed(text)

    async def embed_document(self, text: str) -> list[float]:
        return await self._embed(text)

    async def _embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(model=self._model, input=text)
        vec = resp.data[0].embedding
        # Auto-detect dimension on first call if unknown model
        if self._dim == 0:
            self._dim = len(vec)
        return vec


async def create_embed_provider(config: UCConfig) -> EmbedProvider | None:
    """Create an embedding provider based on config.

    Returns None only if use_llm is disabled.
    Default provider is 'local' (EmbeddingGemma, no API keys).

    Provider resolution:
      - "local"      → EmbeddingGemma-300M via ONNX (always available)
      - "openai"     → requires OPENAI_API_KEY
      - "openrouter" → requires OPENROUTER_API_KEY
      - "auto"       → local (no key needed)
    """
    if not config.use_llm:
        logger.debug("LLM/embeddings disabled by config")
        return None

    provider = config.embed_provider
    model = config.embed_model

    if provider == "local":
        return LocalEmbedProvider()

    if provider == "openai":
        api_key = config.get_api_key("openai")
        if api_key:
            return OpenAIEmbedProvider(api_key, model)
        logger.error("embed_provider='openai' but no OpenAI API key configured.")
        return None

    if provider == "openrouter":
        api_key = config.get_api_key("openrouter")
        if api_key:
            return OpenAIEmbedProvider(
                api_key, model, base_url="https://openrouter.ai/api/v1",
            )
        logger.error("embed_provider='openrouter' but no OpenRouter API key configured.")
        return None

    # "auto" — local is always available, no API keys needed
    return LocalEmbedProvider()


# --- Backwards compatibility ---


async def create_embed_fn(config: UCConfig):
    """Legacy wrapper — returns (embed_document_fn, dim) or (None, 0)."""
    provider = await create_embed_provider(config)
    if provider is None:
        return None
    return provider
