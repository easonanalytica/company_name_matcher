import numpy as np
import pytest
from pathlib import Path
from typing import List, Tuple
from company_name_matcher import VectorStore
from numpy.typing import NDArray


# ---------
# Fixtures 
# ----------

@pytest.fixture
def embedding_dim() -> int:
    return 8


@pytest.fixture
def company_names() -> List[str]:
    return [
        "Acme Corp",
        "Globex",
        "Initech",
        "Umbrella",
        "Hooli",
    ]


@pytest.fixture
def small_embeddings(embedding_dim: int) -> NDArray[np.floating]:
    rng = np.random.default_rng(seed=42)
    return rng.random((5, embedding_dim))


@pytest.fixture
def vector_store(
    small_embeddings: NDArray[np.floating],
    company_names: List[str],
) -> VectorStore:
    return VectorStore(small_embeddings, company_names)


# -------------------------
# Initialization tests
# -------------------------

def test_embeddings_are_normalized(vector_store: VectorStore) -> None:
    norms: NDArray[np.floating] = np.linalg.norm(vector_store.embeddings, axis=1)
    assert np.allclose(norms, 1.0)


def test_items_are_preserved(vector_store: VectorStore, company_names: List[str]) -> None:
    assert vector_store.items == company_names


# -------------------------
# Exact search tests
# -------------------------

def test_exact_search_returns_top_k(vector_store: VectorStore) -> None:
    query: NDArray[np.floating] = vector_store.embeddings[0]

    results: List[Tuple[str, float]] = vector_store.search(query, k=3)

    assert len(results) == 3
    assert results[0][0] == "Acme Corp"
    assert results[0][1] > 0.99


def test_exact_search_k_larger_than_items(vector_store: VectorStore) -> None:
    query: NDArray[np.floating] = vector_store.embeddings[0]

    results = vector_store.search(query, k=100)

    assert len(results) == len(vector_store.items)


# -------------------------
# KMeans index tests
# -------------------------

def test_build_index_creates_clusters(vector_store: VectorStore) -> None:
    vector_store.build_index(n_clusters=2)

    assert vector_store.kmeans is not None
    assert vector_store.clusters is not None
    assert len(vector_store.clusters) == len(vector_store.items)


def test_build_index_with_single_item() -> None:
    embeddings: NDArray[np.floating] = np.random.rand(1, 8)
    store = VectorStore(embeddings, ["Solo Corp"])

    store.build_index(n_clusters=2)

    assert store.kmeans is None
    assert store.clusters is not None
    assert len(store.clusters) == 1


# -------------------------
# Persistence tests
# -------------------------

def test_save_and_load_index(
    vector_store: VectorStore,
    tmp_path: Path,
    embedding_dim: int,
) -> None:
    vector_store.build_index(n_clusters=2)
    vector_store.save_index(str(tmp_path))

    dummy_embeddings: NDArray[np.floating] = np.zeros((1, embedding_dim))
    new_store = VectorStore(dummy_embeddings, ["dummy"])
    new_store.load_index(str(tmp_path))

    assert new_store.items == vector_store.items
    assert np.allclose(new_store.embeddings, vector_store.embeddings)
    assert new_store.kmeans is not None
    assert new_store.clusters is not None


# -------------------------
# Add items tests
# -------------------------

def test_add_items_updates_store(vector_store: VectorStore) -> None:
    vector_store.build_index(n_clusters=2)

    new_embeddings: NDArray[np.floating] = np.random.rand(2, vector_store.embeddings.shape[1])
    new_items: List[str] = ["Stark Industries", "Wayne Enterprises"]

    vector_store.add_items(new_embeddings, new_items)

    assert len(vector_store.items) == 7
    assert "Stark Industries" in vector_store.items
    assert vector_store.embeddings.shape[0] == 7
    assert vector_store.clusters is not None
    assert len(vector_store.clusters) == 7


# -------------------------
# Edge case tests
# -------------------------

def test_empty_store_search() -> None:
    store = VectorStore(np.empty((0, 8)), [])
    results = store.search(np.random.rand(8))

    assert results == []
