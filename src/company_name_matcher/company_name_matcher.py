import logging
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Union
import numpy as np
from .vector_store import VectorStore
import os
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

logger = logging.getLogger(__name__)

class CompanyNameMatcher:
    def __init__(
        self,
        model_path: str = "models/fine_tuned_model",
        preprocess_fn: callable = None,
        stopwords: List[str] = None,
        use_cache: bool = True,
        cache_size: int = 1000
    ):
        self.embedder = SentenceTransformer(model_path)
        self.vector_store = None
        self.stopwords = stopwords or ["inc", "corp", "corporation", "llc", "ltd", "limited", "company"]
        # Use custom preprocessing function if provided, otherwise use default
        self.preprocess_fn = preprocess_fn if preprocess_fn is not None else self._default_preprocess
        
        # Embedding cache
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.embedding_cache = {}

    def _default_preprocess(self, name: str) -> str:
        """Default preprocessing: lowercase, remove special chars and optional stopwords."""
        name = name.strip().lower()
        # Remove special characters
        name = re.sub(r'[^a-z0-9\s]', '', name)
        # Optionally remove stopwords
        if self.stopwords:
            words = name.split()
            words = [word for word in words if word not in self.stopwords]
            name = ' '.join(words)
        return name

    def _preprocess_company_name(self, name: str) -> str:
        """Preprocess company name using the configured preprocessing function."""
        return self.preprocess_fn(name)

    def get_embedding(self, company_name: str) -> np.ndarray:
        """Get the embedding for a single company name with caching."""
        preprocessed_name = self._preprocess_company_name(company_name)
        
        # Check cache first if enabled
        if self.use_cache and preprocessed_name in self.embedding_cache:
            return self.embedding_cache[preprocessed_name]
        
        embedding = self.embedder.encode([preprocessed_name])[0]
        
        # Update cache if enabled
        if self.use_cache:
            # Simple LRU-like behavior: remove oldest item if at capacity
            if len(self.embedding_cache) >= self.cache_size:
                # Remove a random item (simple approach)
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            self.embedding_cache[preprocessed_name] = embedding
        
        return embedding

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
        target_company: Union[str, List[str]],
        threshold: float = 0.9,
        k: int = 5,
        use_approx: bool = False,
        batch_size: int = 32,
        n_jobs: int = 1,
        n_probe_clusters: int = 1
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """
        Find matches for one or multiple target companies.

        Args:
            target_company: Single company name or list of company names to match
            threshold: Minimum similarity score (0-1)
            k: Number of top matches to return per company
            use_approx: Whether to use approximate k-means search
            batch_size: Number of companies to process in each batch (when target_company is a list)
            n_jobs: Number of parallel jobs to run (1 means no parallelization)
                   Set to -1 to use all available CPU cores
            n_probe_clusters: Number of closest clusters to search when using approximate search

        Returns:
            For a single company: List of (company, similarity) tuples
            For multiple companies: List of lists of (company, similarity) tuples
        """
        if self.vector_store is None:
            raise ValueError("No index available. Call build_index or load_index first.")

        # Handle single company case
        if isinstance(target_company, str):
            target_embedding = self.get_embedding(target_company)
            return self._find_matches_single(target_embedding, threshold, k, use_approx, n_probe_clusters)
        
        # Handle multiple companies case
        if n_jobs == 1:
            # Sequential processing
            return self._batch_find_matches_sequential(
                target_company, threshold, k, use_approx, batch_size, n_probe_clusters
            )
        else:
            # Parallel processing
            return self._batch_find_matches_parallel(
                target_company, threshold, k, use_approx, batch_size, n_jobs, n_probe_clusters
            )

    def _find_matches_single(
        self, 
        target_embedding: np.ndarray, 
        threshold: float, 
        k: int, 
        use_approx: bool,
        n_probe_clusters: int = 1
    ) -> List[Tuple[str, float]]:
        """Find matches for a single embedding."""
        if use_approx:
            # Get more candidates than k since we'll filter by threshold
            matches = self.vector_store.search(
                target_embedding, 
                k=max(k * 2, 20), 
                use_approx=True,
                n_probe_clusters=n_probe_clusters
            )
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

    def _batch_find_matches_sequential(
        self,
        target_companies: List[str],
        threshold: float,
        k: int,
        use_approx: bool,
        batch_size: int,
        n_probe_clusters: int
    ) -> List[List[Tuple[str, float]]]:
        """Process multiple companies in batches sequentially."""
        results = []
        
        # Process in batches
        for i in range(0, len(target_companies), batch_size):
            batch = target_companies[i:i+batch_size]
            batch_embeddings = self.get_embeddings(batch)
            
            batch_results = []
            for embedding in batch_embeddings:
                matches = self._find_matches_single(embedding, threshold, k, use_approx, n_probe_clusters)
                batch_results.append(matches)
                
            results.extend(batch_results)
            
        return results

    def _batch_find_matches_parallel(
        self,
        target_companies: List[str],
        threshold: float,
        k: int,
        use_approx: bool,
        batch_size: int,
        n_jobs: int,
        n_probe_clusters: int
    ) -> List[List[Tuple[str, float]]]:
        """Process multiple companies in parallel."""
        # Determine number of workers
        if n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count()
        n_jobs = min(n_jobs, multiprocessing.cpu_count())
        
        # Create batches
        batches = []
        for i in range(0, len(target_companies), batch_size):
            batches.append(target_companies[i:i+batch_size])
        
        # Define the worker function
        def process_batch(batch):
            batch_embeddings = self.get_embeddings(batch)
            batch_results = []
            for embedding in batch_embeddings:
                matches = self._find_matches_single(embedding, threshold, k, use_approx, n_probe_clusters)
                batch_results.append(matches)
            return batch_results
        
        # Process batches in parallel
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            batch_results = list(executor.map(process_batch, batches))
            
        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)
            
        return results

    # For backward compatibility
    def batch_find_matches(
        self,
        target_companies: List[str],
        threshold: float = 0.9,
        k: int = 5,
        use_approx: bool = False,
        batch_size: int = 32,
        n_jobs: int = 1
    ) -> List[List[Tuple[str, float]]]:
        """
        Find matches for multiple target companies (alias for find_matches with a list).
        
        Args:
            target_companies: List of company names to match
            threshold: Minimum similarity score (0-1)
            k: Number of top matches to return per company
            use_approx: Whether to use approximate k-means search
            batch_size: Number of companies to process in each batch
            n_jobs: Number of parallel jobs to run (1 means no parallelization)
                   Set to -1 to use all available CPU cores
            
        Returns:
            List of match results for each target company
        """
        return self.find_matches(
            target_companies, threshold, k, use_approx, batch_size, n_jobs
        )

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
