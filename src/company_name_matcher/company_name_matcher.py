import logging
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Union, Callable, Optional, Dict, cast
import numpy as np
from .vector_store import VectorStore
import os
import re
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CompanyNameMatcher:
    """
    Company name matching using embedding-based similarity search.

    This class provides methods to embed company names, build and search
    an index for similar names, and handle batch or parallel searches.
    It also supports caching embeddings and custom preprocessing of company names.

    Parameters
    ----------
    model_path : str, default="easonanalytica/cnm-multilingual-small-v2"
        Path or identifier for the SentenceTransformer model to use.
    preprocess_fn : Callable[[str], str], optional
        Optional custom preprocessing function for company names.
        If not provided, a default preprocessing is used (lowercasing, removing special characters, stopwords).
    stopwords : List[str], default=["inc", "corp", "corporation", "llc", "ltd", "limited", "company"]
        Words to remove from company names during preprocessing (used by default preprocessing).
    use_cache : bool, default=True
        Whether to cache embeddings for faster repeated lookup.
    cache_size : int, default=1000
        Maximum number of embeddings to store in the cache.

    Examples
    --------
    >>> from company_name_matcher import CompanyNameMatcher
    >>> matcher = CompanyNameMatcher()
    >>> company_list = ["Acme Corp", "Globex Corporation", "Initech LLC"]
    >>> matcher.build_index(company_list, n_clusters=2)
    >>> matcher.find_matches("Acme", threshold=0.8, k=3)
    [('Acme Corp', 1.0000001192092896)]
    """

    def __init__(
        self,
        model_path: str = "easonanalytica/cnm-multilingual-small-v2",
        preprocess_fn: Optional[Callable[[str], str]] = None,
        stopwords: List[str] = [
            "inc",
            "corp",
            "corporation",
            "llc",
            "ltd",
            "limited",
            "company",
        ],
        use_cache: bool = True,
        cache_size: int = 1000,
    ):
        self.embedder = SentenceTransformer(model_path)
        self.vector_store = None
        self.stopwords = stopwords
        # Use custom preprocessing function if provided, otherwise use default
        self.preprocess_fn = preprocess_fn if preprocess_fn is not None else self._default_preprocess

        # Embedding cache
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.embedding_cache: Dict[str, NDArray[np.floating]] = {}

    def _default_preprocess(self, name: str) -> str:
        """Default preprocessing: lowercase, remove special chars and optional stopwords."""
        name = name.strip().lower()
        # Remove special characters
        name = re.sub(r"[^a-z0-9\s]", "", name)
        # Optionally remove stopwords
        if self.stopwords:
            words = name.split()
            words = [word for word in words if word not in self.stopwords]
            name = " ".join(words)
        return name

    def _preprocess_company_name(self, name: str) -> str:
        """Preprocess company name using the configured preprocessing function."""
        return self.preprocess_fn(name)

    def get_embedding(self, company_name: str) -> NDArray[np.floating]:
        """
        Get the embedding vector for a single company name.

        This method preprocesses the company name, optionally uses caching,
        and returns a numerical embedding suitable for similarity comparisons.

        Parameters
        ----------
        company_name : str
            The company name to embed.

        Returns
        -------
        NDArray[np.floating]
            A 1D numpy array representing the embedding of the company name.

        Examples
        --------
        >>> from company_name_matcher import CompanyNameMatcher
        >>> matcher = CompanyNameMatcher()
        >>> matcher.build_index(["Acme Corp", "Globex Corp", "Initech LLC"])
        >>> embedding = matcher.get_embedding("Acme Corp")
        >>> embedding.shape
        (384,)  # assuming model outputs 384-dim embeddings
        """
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

    def get_embeddings(self, company_names: List[str]) -> NDArray[np.floating]:
        """
        Get embedding vectors for a list of company names.

        This method preprocesses all names, optionally uses caching,
        and returns a 2D array of embeddings.

        Parameters
        ----------
        company_names : List[str]
            List of company names to embed.

        Returns
        -------
        NDArray[np.floating]
            A 2D numpy array of shape (len(company_names), embedding_dim),
            where each row corresponds to the embedding of a company name.

        Examples
        --------
        >>> from company_name_matcher import CompanyNameMatcher
        >>> matcher = CompanyNameMatcher()
        >>> matcher.build_index(["Acme Corp", "Globex Corp", "Initech LLC"])
        >>> embeddings = matcher.get_embeddings(["Acme Corp", "Initech LLC"])
        >>> embeddings.shape
        (2, 384)
        """
        preprocessed_names = [self._preprocess_company_name(name) for name in company_names]
        return self.embedder.encode(preprocessed_names)

    def compare_companies(self, company_a: str, company_b: str) -> float:
        """
        Compare two company names and return a similarity score between 0 and 1.

        The score is based on cosine similarity of the embeddings. Preprocessing
        is applied to both names, and caching may be used if enabled.

        Parameters
        ----------
        company_a : str
            The first company name.
        company_b : str
            The second company name.

        Returns
        -------
        float
            Cosine similarity score between the two company names (0-1).

        Examples
        --------
        >>> from company_name_matcher import CompanyNameMatcher
        >>> matcher = CompanyNameMatcher()
        >>> matcher.build_index(["Acme Corp", "Globex Corp", "Initech LLC"])
        >>> matcher.compare_companies("Acme Corp", "Acme Inc")
        1.0
        >>> matcher.compare_companies("Acme Corp", "Initech LLC")
        0.3694563
        """
        embedding_a = self.get_embedding(company_a)
        embedding_b = self.get_embedding(company_b)
        return float(self._cosine_similarity(embedding_a, embedding_b)[0][0])

    def build_index(
        self,
        company_list: List[str],
        n_clusters: int = 100,
        save_dir: Optional[str] = None,
    ):
        """
        Build a search index for a list of company names.

        Parameters
        ----------
        company_list : List[str]
            List of company names to index.
        n_clusters : int, default=100
            Number of clusters to create in the underlying KMeans index.
        save_dir : str, optional
            Directory path to save the index files ('embeddings.h5' and 'kmeans_model.joblib').

        Examples
        --------
        >>> from company_name_matcher import CompanyNameMatcher
        >>> matcher = CompanyNameMatcher()
        >>> companies = ["Acme Corp", "Globex Corp", "Initech LLC"]
        >>> matcher.build_index(companies, n_clusters=2)
        """
        embeddings = self.get_embeddings(company_list)
        self.vector_store = VectorStore(embeddings, company_list)

        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        self.vector_store.build_index(n_clusters, save_dir)

    def load_index(self, load_dir: str):
        """
        Load a previously saved search index.

        Parameters
        ----------
        load_dir : str
            Directory path containing the index files ('embeddings.h5' and 'kmeans_model.joblib').

        Examples
        --------
        >>> from company_name_matcher import CompanyNameMatcher
        >>> matcher = CompanyNameMatcher()
        >>> matcher.load_index('company_index/')
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
        n_probe_clusters: int = 1,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """
        Find matching companies for a single company name or a list of names.

        Parameters
        ----------
        target_company : str or List[str]
            Company name or list of company names to match.
        threshold : float, default=0.9
            Minimum similarity score (0-1) to consider a match.
        k : int, default=5
            Number of top matches to return per company.
        use_approx : bool, default=False
            Whether to use approximate search with KMeans clusters.
        batch_size : int, default=32
            Number of companies to process in each batch (when passing a list).
        n_jobs : int, default=1
            Number of parallel jobs to run (-1 uses all available CPU cores).
        n_probe_clusters : int, default=1
            Number of closest clusters to probe when using approximate search.

        Returns
        -------
        If a single company is provided:
            List[Tuple[str, float]] -- top matching company names and their similarity scores.
        If a list of companies is provided:
            List[List[Tuple[str, float]]] -- list of matches for each input company.

        Examples
        --------
        >>> from company_name_matcher import CompanyNameMatcher
        >>> matcher = CompanyNameMatcher()
        >>> matcher.build_index(["Acme Corp", "Globex Corp", "Initech LLC"])
        >>> matcher.find_matches("Acme", threshold=0.8, k=3)
        [('Acme Corp', 1.0000001192092896)]
        >>> matcher.find_matches(["Acme", "Initech"], threshold=0.8)
        [[('Acme Corp', 1.0000001192092896)], [('Initech LLC', 1.0)]]
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
                target_company,
                threshold,
                k,
                use_approx,
                batch_size,
                n_jobs,
                n_probe_clusters,
            )

    def _find_matches_single(
        self,
        target_embedding: NDArray[np.floating],
        threshold: float,
        k: int,
        use_approx: bool,
        n_probe_clusters: int = 1,
    ) -> List[Tuple[str, float]]:
        """Find matches for a single embedding."""
        if use_approx:
            # Get more candidates than k since we'll filter by threshold
            assert self.vector_store is not None, "vector_store is not initialised yet"
            matches = self.vector_store.search(
                target_embedding,
                k=max(k * 2, 20),
                use_approx=True,
                n_probe_clusters=n_probe_clusters,
            )
            # Filter by threshold and take top k
            matches = [(company, similarity) for company, similarity in matches if similarity >= threshold]
            matches = matches[:k]
        else:
            # Use exact search with the stored embeddings
            assert self.vector_store is not None, "vector_store is not initialised yet"
            similarities = self._cosine_similarity(target_embedding.reshape(1, -1), self.vector_store.embeddings)
            similarities = similarities.flatten().tolist()

            # Get all matches above threshold
            matches = [
                (company, similarity)
                for company, similarity in zip(self.vector_store.items, similarities)
                if similarity >= threshold
            ]
            matches = sorted(matches, key=lambda x: x[1], reverse=True)[:k]

        return matches

    def _batch_find_matches_sequential(
        self,
        target_companies: List[str],
        threshold: float,
        k: int,
        use_approx: bool,
        batch_size: int,
        n_probe_clusters: int,
    ) -> List[List[Tuple[str, float]]]:
        """Process multiple companies in batches sequentially."""
        results: List[List[Tuple[str, float]]] = []

        # Process in batches
        for i in range(0, len(target_companies), batch_size):
            batch = target_companies[i : i + batch_size]
            batch_embeddings = self.get_embeddings(batch)

            batch_results: List[List[Tuple[str, float]]] = []
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
        n_probe_clusters: int,
    ) -> List[List[Tuple[str, float]]]:
        """Process multiple companies in parallel."""
        # Determine number of workers
        if n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count()
        n_jobs = min(n_jobs, multiprocessing.cpu_count())

        # Create batches
        batches: List[List[str]] = []
        for i in range(0, len(target_companies), batch_size):
            batches.append(target_companies[i : i + batch_size])

        # Define the worker function
        def process_batch(batch: List[str]) -> List[List[Tuple[str, float]]]:
            batch_embeddings = self.get_embeddings(batch)
            batch_results: List[List[Tuple[str, float]]] = []
            for embedding in batch_embeddings:
                matches = self._find_matches_single(embedding, threshold, k, use_approx, n_probe_clusters)
                batch_results.append(matches)
            return batch_results

        # Process batches in parallel
        results: List[List[Tuple[str, float]]] = []
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
        n_jobs: int = 1,
    ) -> List[List[Tuple[str, float]]]:
        """
        Alias for `find_matches` when matching a list of companies.

        Parameters
        ----------
        target_companies : List[str]
            List of company names to match.
        threshold : float, default=0.9
            Minimum similarity score (0-1) to consider a match.
        k : int, default=5
            Number of top matches to return per company.
        use_approx : bool, default=False
            Whether to use approximate search with KMeans clusters.
        batch_size : int, default=32
            Number of companies to process in each batch.
        n_jobs : int, default=1
            Number of parallel jobs to run (-1 uses all available CPU cores).

        Returns
        -------
        List[List[Tuple[str, float]]]
            List of matches for each target company.

        Examples
        --------
        >>> from company_name_matcher import CompanyNameMatcher
        >>> matcher = CompanyNameMatcher()
        >>> matcher.build_index(["Acme Corp", "Globex Corp", "Initech LLC"])
        >>> matcher.batch_find_matches(["Acme", "Initech"], threshold=0.8)
        [[('Acme Corp', 1.0000001192092896)], [('Initech LLC', 1.0)]]
        """
        return cast(
            List[List[Tuple[str, float]]],
            self.find_matches(target_companies, threshold, k, use_approx, batch_size, n_jobs),
        )

    @staticmethod
    def _cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
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

    def expand_index(self, new_company_list: List[str], save_dir: Optional[str] = None):
        """
        Add new companies to an existing index.

        Parameters
        ----------
        new_company_list : List[str]
            List of new company names to add.
        save_dir : str, optional
            Directory path to save the updated index.

        Raises
        ------
        ValueError
            If no index has been built or loaded.

        Examples
        --------
        >>> from company_name_matcher import CompanyNameMatcher
        >>> matcher = CompanyNameMatcher()
        >>> matcher.build_index(["Acme Corp", "Globex Corp"])
        >>> matcher.expand_index(["Wayne Enterprises", "Stark Industries"])
        """
        if self.vector_store is None:
            raise ValueError("No index available. Call build_index or load_index first.")

        new_embeddings = self.get_embeddings(new_company_list)
        self.vector_store.add_items(new_embeddings, new_company_list, save_dir)
