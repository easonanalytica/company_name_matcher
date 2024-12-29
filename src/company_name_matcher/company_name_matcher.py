import logging
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
from .vector_store import VectorStore
import os

logger = logging.getLogger(__name__)

class CompanyNameMatcher:
    def __init__(
        self,
        model_path: str = "models/fine_tuned_model",
        preprocess_fn: callable = None
    ):

        self.embedder = SentenceTransformer(model_path)
        self.vector_store = None
        # Use custom preprocessing function if provided, otherwise use default
        self.preprocess_fn = preprocess_fn if preprocess_fn is not None else self._default_preprocess

    def _default_preprocess(self, name: str) -> str:
        """Default preprocessing: add special tokens to the company name."""
        return name.strip().lower()

    def _preprocess_company_name(self, name: str) -> str:
        """Preprocess company name using the configured preprocessing function."""
        return self.preprocess_fn(name)

    def get_embedding(self, company_name: str) -> np.ndarray:
        """get the embedding for a single company name."""
        preprocessed_name = self._preprocess_company_name(company_name)
        return self.embedder.encode([preprocessed_name])[0]

    def get_embeddings(self, company_names: List[str]) -> np.ndarray:
        """get embeddings for a list of company names."""
        preprocessed_names = [self._preprocess_company_name(name) for name in company_names]
        return self.embedder.encode(preprocessed_names)

    def compare_companies(self, company_a: str, company_b: str) -> float:
        """compare two company names and return a similarity score."""
        embedding_a = self.get_embedding(company_a)
        embedding_b = self.get_embedding(company_b)
        return self._cosine_similarity(embedding_a, embedding_b)[0][0]

    def build_index(self, company_list: List[str], n_clusters: int = 100, save_dir: str = None):
        """
        Build search index for the company list

        Args:
            company_list: List of company names to index
            n_clusters: Number of clusters for KMeans
            save_dir: Optional directory path to save the index files
                     Will create 'embeddings.h5' and 'kmeans_model.joblib' in this directory
        """
        embeddings = self.get_embeddings(company_list)
        self.vector_store = VectorStore(embeddings, company_list)

        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        self.vector_store.build_index(n_clusters, save_dir)

    def load_index(self, load_dir: str):
        """
        Load a previously saved search index

        Args:
            load_dir: Directory path containing the index files
                     ('embeddings.h5' and 'kmeans_model.joblib')
        """
        self.vector_store = VectorStore(np.array([[0]]), ["dummy"])  # Initialize with dummy data
        self.vector_store.load_index(load_dir)

    def find_matches(
        self,
        target_company: str,
        threshold: float = 0.9,
        k: int = 5,
        use_approx: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Find matches for a target company using the built/loaded index.

        Args:
            target_company: Company name to match
            threshold: Minimum similarity score (0-1)
            k: Number of top matches to return
            use_approx: Whether to use approximate k-means search

        Raises:
            ValueError: If no index has been built or loaded
        """
        if self.vector_store is None:
            raise ValueError("No index available. Call build_index or load_index first.")

        target_embedding = self.get_embedding(target_company)

        if use_approx:
            # Get more candidates than k since we'll filter by threshold
            matches = self.vector_store.search(target_embedding, k=max(k * 2, 20), use_approx=True)
            # Filter by threshold and take top k
            matches = [(company, similarity)
                      for company, similarity in matches
                      if similarity >= threshold]
            matches = matches[:k]
        else:
            # Use exact search with the stored embeddings
            similarities = self._cosine_similarity(target_embedding.reshape(1, -1), self.vector_store.embeddings)
            similarities = similarities.flatten()

            # Get all matches above threshold
            matches = [(company, similarity)
                      for company, similarity in zip(self.vector_store.items, similarities)
                      if similarity >= threshold]
            matches = sorted(matches, key=lambda x: x[1], reverse=True)[:k]

        return matches

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between two vectors or between a vector and a matrix."""
        logger.debug(f"Input shapes: a={a.shape}, b={b.shape}")

        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)

        logger.debug(f"Reshaped input shapes: a={a.shape}, b={b.shape}")

        # compute the dot product
        dot_product = np.dot(a, b.T)

        # compute the L2 norm
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)

        # compute the cosine similarity
        result = dot_product / (norm_a[:, np.newaxis] * norm_b)

        logger.debug(f"Result shape: {result.shape}")

        return result

    def expand_index(self, new_company_list: List[str], save_dir: str = None):
        """
        Add new companies to the existing index

        Args:
            new_company_list: List of new company names to add to the index
            save_dir: Optional directory path to save the updated index

        Raises:
            ValueError: If no index has been built or loaded
        """
        if self.vector_store is None:
            raise ValueError("No index available. Call build_index or load_index first.")

        new_embeddings = self.get_embeddings(new_company_list)
        self.vector_store.add_items(new_embeddings, new_company_list, save_dir)
