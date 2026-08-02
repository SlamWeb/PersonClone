"""Embedding adapters for index building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


_BGE_M3_MODEL_CLASS: Any | None = None


def prepare_bge_m3_runtime() -> Any:
    """Import FlagEmbedding on the caller thread before background model loading."""

    global _BGE_M3_MODEL_CLASS
    if _BGE_M3_MODEL_CLASS is not None:
        return _BGE_M3_MODEL_CLASS
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError(
            "BGE-M3 indexing requires optional dependency `FlagEmbedding`. "
            "Install with: pip install -e \".[index]\""
        ) from exc
    _BGE_M3_MODEL_CLASS = BGEM3FlagModel
    return _BGE_M3_MODEL_CLASS


@dataclass(slots=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]


@dataclass(slots=True)
class TextEmbedding:
    dense: list[float]
    sparse: SparseEmbedding


class TextEncoder(Protocol):
    def encode_texts(self, texts: list[str], *, batch_size: int = 12) -> list[TextEmbedding]: ...


class BgeM3Encoder:
    """Thin wrapper around FlagEmbedding's BGE-M3 dense+sparse output."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str | None = None,
        use_fp16: bool = True,
        max_length: int = 8192,
    ) -> None:
        model_class = prepare_bge_m3_runtime()
        kwargs: dict[str, Any] = {"use_fp16": use_fp16}
        if device and device != "auto":
            kwargs["devices"] = device
        self.model = model_class(model_name, **kwargs)
        self.max_length = max_length

    def encode_texts(self, texts: list[str], *, batch_size: int = 12) -> list[TextEmbedding]:
        if not texts:
            return []
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"]
        sparse_weights = output["lexical_weights"]
        return [
            TextEmbedding(
                dense=_dense_to_list(dense),
                sparse=_lexical_weights_to_sparse(sparse),
            )
            for dense, sparse in zip(dense_vectors, sparse_weights, strict=True)
        ]


def _dense_to_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _lexical_weights_to_sparse(value: dict[Any, Any]) -> SparseEmbedding:
    pairs = sorted((int(index), float(weight)) for index, weight in value.items() if float(weight) != 0.0)
    return SparseEmbedding(
        indices=[index for index, _ in pairs],
        values=[weight for _, weight in pairs],
    )
