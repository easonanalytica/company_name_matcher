import h5py
import logging
from typing import List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from joblib import dump, load
import os
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class VectorStore:
    """
    A vector store for embedding-based company name matching with optional KMeans clustering.

    This class allows storing company name embeddings, performing exact or approximate similarity search,
    and managing the index for efficient retrieval.

    Parameters
    ----------
    embeddings : NDArray[np.floating]
        2D array of shape (n_items, embedding_dim) containing the company name embeddings.
    items : List[str]
        List of company names corresponding to the embeddings.

    Examples
    --------
    >>> import numpy as np
    >>> from company_name_matcher import VectorStore
    >>> embeddings = np.random.rand(5, 128)  # 5 company name embeddings with 128 dimensions
    >>> company_names = ['Acme Corp', 'Globex', 'Initech', 'Umbrella', 'Hooli']
    >>> store = VectorStore(embeddings, company_names)
    >>> store.items
    ['Acme Corp', 'Globex', 'Initech', 'Umbrella', 'Hooli']
    """

    def __init__(self, embeddings: NDArray[np.floating], items: List[str]):
        if len(embeddings) == 1 and embeddings[0][0] == 0 and items == ["dummy"]:
            # Special case for dummy initialization
            self.embeddings = embeddings
            self.items = items
        else:
            # Normal case - normalize the embeddings
            self.embeddings = self._normalize(embeddings)
            self.items = items
        self.kmeans = None
        self.clusters: Optional[NDArray[np.int64]] = None

    # ----------------- Internal Helpers -----------------
    def _normalize(self, embeddings: NDArray[np.floating]) -> NDArray[np.floating]:
        """Normalize embeddings to unit vectors."""
        return embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    def _get_index_paths(self, base_path: str) -> Tuple[str, str]:
        """Return file paths for embeddings and KMeans model."""
        return os.path.join(base_path, "embeddings.h5"), os.path.join(base_path, "kmeans_model.joblib")

    def _save_embeddings(self, path: str) -> None:
        """Save embeddings and items to HDF5."""
        with h5py.File(path, "w") as f:
            f.create_dataset("embeddings", data=self.embeddings, compression="gzip")
            dt = h5py.special_dtype(vlen=str)
            items_dataset = f.create_dataset("items", (len(self.items),), dtype=dt)
            items_dataset[:] = self.items

    def _load_embeddings(self, path: str) -> None:
        """Load embeddings and items from HDF5."""
        with h5py.File(path, "r") as f:
            self.embeddings = f["embeddings"][:]
            self.items = [item.decode("utf-8") if isinstance(item, bytes) else item for item in f["items"][:]]

    def _update_clusters(self, new_embeddings: Optional[NDArray[np.floating]] = None) -> None:
        """Update or predict clusters using KMeans."""
        if self.kmeans is not None:
            if new_embeddings is not None:
                new_clusters = self.kmeans.predict(new_embeddings)
                assert self.clusters is not None
                self.clusters = np.concatenate([self.clusters.astype(np.int64), new_clusters])
            else:
                self.clusters = self.kmeans.fit_predict(self.embeddings)

    @staticmethod
    def _cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
        """Calculate cosine similarity between normalized vectors."""
        return np.dot(a, b.T)

    # ----------------- Public Methods -----------------
    def build_index(
        self,
        n_clusters: int = 100,
        save_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        """
        Build a KMeans clustering index for approximate nearest neighbor search.

        Useful for quickly retrieving similar company names when the dataset is large.

        Parameters
        ----------
        n_clusters : int, default=100
            The number of clusters to create in KMeans.
        save_path : str, optional
            Directory path where the index will be saved. If None, the index is not saved.
        overwrite : bool, default=True
            If True, existing files in the save path will be overwritten.

        Examples
        --------
        >>> embeddings = np.random.rand(10, 64)
        >>> company_names = [f'Company{i}' for i in range(10)]
        >>> store = VectorStore(embeddings, company_names)
        >>> store.build_index(n_clusters=3)
        >>> store.clusters.shape
        (10,)
        >>> store.kmeans.cluster_centers_.shape
        (3, 64)
        """
        if len(self.items) <= 1:
            logger.warning("Cannot build KMeans index with 1 or 0 items. Skipping index creation.")
            self.kmeans = None
            self.clusters = np.zeros(len(self.items), dtype=int) if len(self.items) > 0 else np.array([])
            return

        if len(self.items) < n_clusters:
            n_clusters = max(1, min(len(self.items) - 1, len(self.items) // 2))
            logger.info(f"Reduced number of clusters to {n_clusters} based on dataset size")

        self.kmeans = KMeans(n_clusters=n_clusters)
        self._update_clusters()

        if save_path:
            self.save_index(save_path, overwrite=overwrite)

    def save_index(self, save_path: str, overwrite: bool = True) -> None:
        """
        Save the embeddings, company names, and clustering index to disk.

        Parameters
        ----------
        save_path : str
            Directory path to save the index.
        overwrite : bool, default=True
            Whether to overwrite existing files in the save path.

        Examples
        --------
        >>> import numpy as np
        >>> from company_name_matcher import VectorStore
        >>> embeddings = np.random.rand(5, 128)
        >>> company_names = ['Acme Corp', 'Globex', 'Initech', 'Umbrella', 'Hooli']
        >>> store = VectorStore(embeddings, company_names)
        >>> store.build_index(n_clusters=2)
        >>> store.save_index('company_index/')
        """
        if self.kmeans is None:
            raise ValueError("No index to save. Call build_index first.")

        os.makedirs(save_path, exist_ok=True)
        h5_path, model_path = self._get_index_paths(save_path)

        if not overwrite and (os.path.exists(h5_path) or os.path.exists(model_path)):
            raise FileExistsError(f"Index files already exist in {save_path}. Set overwrite=True to replace them.")

        self._save_embeddings(h5_path)
        dump({"kmeans": self.kmeans, "clusters": self.clusters}, model_path)
        logger.info(f"Index saved to {save_path}")

    def load_index(self, load_path: str) -> None:
        """
        Load embeddings, company names, and clustering index from disk.

        Parameters
        ----------
        load_path : str
            Directory path containing the saved index files.

        Examples
        --------
        >>> from company_name_matcher import VectorStore
        >>> store = VectorStore(np.random.rand(1, 128), ['dummy'])  # dummy init
        >>> store.load_index('company_index/')
        >>> store.items
        ['Acme Corp', 'Globex', 'Initech', 'Umbrella', 'Hooli']
        """
        h5_path, model_path = self._get_index_paths(load_path)
        if not os.path.exists(h5_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Index files not found in {load_path}")

        self._load_embeddings(h5_path)
        data = load(model_path)
        self.kmeans = data["kmeans"]
        self.clusters = data["clusters"]

    def search(
        self,
        query_embedding: NDArray[np.floating],
        k: int = 5,
        use_approx: bool = False,
        n_probe_clusters: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Search for company names similar to a given query embedding.

        Parameters
        ----------
        query_embedding : NDArray[np.floating]
            Embedding vector of the query company name.
        k : int, default=5
            Number of top results to return.
        use_approx : bool, default=False
            If True, use approximate search with KMeans clusters for faster retrieval.
        n_probe_clusters : int, default=3
            Number of closest clusters to search in approximate mode.

        Returns
        -------
        List[Tuple[str, float]]
            List of tuples containing the company name and similarity score.

        Examples
        --------
        >>> import numpy as np
        >>> from company_name_matcher import VectorStore
        >>> embeddings = np.random.rand(5, 64)
        >>> company_names = ['Acme Corp', 'Globex', 'Initech', 'Umbrella', 'Hooli']
        >>> store = VectorStore(embeddings, company_names)
        >>> store.build_index(n_clusters=2)
        >>> query_emb = np.random.rand(64)
        >>> results = store.search(query_emb, k=3)
        >>> for name, score in results:
        ...     print(name, score)
        Globex 0.8084718360141672
        Umbrella 0.7423598126674433
        Initech 0.7023631018378196
        """
        if len(self.items) == 0:
            return []

        query_embedding = self._normalize(query_embedding.reshape(1, -1))[0]

        # Determine candidate indices
        if use_approx and self.kmeans is not None:
            distances = self.kmeans.transform(query_embedding.reshape(1, -1))[0]
            closest_clusters = np.argsort(distances)[:n_probe_clusters]

            candidate_indices: NDArray[np.int64] = np.concatenate(
                [np.where(self.clusters == cluster)[0] for cluster in closest_clusters]
            )

            if len(candidate_indices) == 0:
                logger.warning(
                    f"No items found in the {n_probe_clusters} closest clusters. Falling back to exact search."
                )
                candidate_indices = np.arange(len(self.items))
        else:
            candidate_indices = np.arange(len(self.items))

        # Compute similarities and return top-k
        similarities = self._cosine_similarity(query_embedding.reshape(1, -1), self.embeddings[candidate_indices])
        k = min(k, len(candidate_indices))
        top_k_indices = np.argsort(similarities.flatten())[-k:][::-1]

        return [(self.items[candidate_indices[i]], float(similarities.flatten()[i])) for i in top_k_indices]

    def add_items(
        self,
        new_embeddings: NDArray[np.floating],
        new_items: List[str],
        save_dir: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        """
        Add new company names and embeddings to the store and optionally save the updated index.

        Parameters
        ----------
        new_embeddings : NDArray[np.floating]
            2D array containing embeddings of the new company names.
        new_items : List[str]
            List of new company names.
        save_dir : str, optional
            Directory path to save the updated index.
        overwrite : bool, default=True
            Whether to overwrite existing files if saving.

        Examples
        --------
        >>> import numpy as np
        >>> from company_name_matcher import VectorStore
        >>> embeddings = np.random.rand(5, 128)
        >>> company_names = ['Acme Corp', 'Globex', 'Initech', 'Umbrella', 'Hooli']
        >>> store = VectorStore(embeddings, company_names)
        >>> new_embeddings = np.random.rand(2, 128)
        >>> new_companies = ['Stark Industries', 'Wayne Enterprises']
        >>> store.add_items(new_embeddings, new_companies)
        >>> store.items
        ['Acme Corp', 'Globex', 'Initech', 'Umbrella', 'Hooli', 'Stark Industries', 'Wayne Enterprises']
        """
        normalized_embeddings = self._normalize(new_embeddings)
        self.embeddings = np.vstack([self.embeddings, normalized_embeddings])
        self.items.extend(new_items)

        self._update_clusters(new_embeddings=normalized_embeddings)

        if save_dir:
            self.save_index(save_dir, overwrite=overwrite)
