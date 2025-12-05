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
            self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
            self.items = items
        self.kmeans = None
        self.clusters: Optional[NDArray[np.int64]] = None

    def build_index(
        self,
        n_clusters: int = 100,
        save_path: Optional[str] = None,
        overwrite: bool = True,
    ):
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
        # Handle edge cases with very small datasets
        if len(self.items) <= 1:
            logger.warning("Cannot build KMeans index with 1 or 0 items. Skipping index creation.")
            self.kmeans = None
            self.clusters = np.zeros(len(self.items), dtype=int) if len(self.items) > 0 else np.array([])
            return

        if len(self.items) < n_clusters:
            n_clusters = max(1, min(len(self.items) - 1, len(self.items) // 2))
            logger.info(f"Reduced number of clusters to {n_clusters} based on dataset size")

        self.kmeans = KMeans(n_clusters=n_clusters)
        self.clusters = self.kmeans.fit_predict(self.embeddings)

        if save_path:
            self.save_index(save_path, overwrite=overwrite)

    def save_index(self, save_path: str, overwrite: bool = True):
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

        # Check if files already exist
        h5_path = os.path.join(save_path, "embeddings.h5")
        model_path = os.path.join(save_path, "kmeans_model.joblib")

        if not overwrite and (os.path.exists(h5_path) or os.path.exists(model_path)):
            raise FileExistsError(f"Index files already exist in {save_path}. Set overwrite=True to replace them.")

        # Save embeddings and items using h5py
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("embeddings", data=self.embeddings, compression="gzip")
            dt = h5py.special_dtype(vlen=str)
            items_dataset = f.create_dataset("items", (len(self.items),), dtype=dt)
            items_dataset[:] = self.items

        # Save KMeans model and clusters using joblib
        dump({"kmeans": self.kmeans, "clusters": self.clusters}, model_path)

        logger.info(f"Index saved to {save_path}")

    def load_index(self, load_path: str):
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
        h5_path = os.path.join(load_path, "embeddings.h5")
        model_path = os.path.join(load_path, "kmeans_model.joblib")

        if not os.path.exists(h5_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Index files not found in {load_path}")

        # Load embeddings and items from h5py
        with h5py.File(h5_path, "r") as f:
            self.embeddings = f["embeddings"][:]
            # Decode byte strings to regular strings
            self.items = [item.decode("utf-8") if isinstance(item, bytes) else item for item in f["items"][:]]

        # Load KMeans model and clusters from joblib
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
        # Handle empty index case
        if len(self.items) == 0:
            return []

        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        if not use_approx or self.kmeans is None:
            # Exact search using cosine similarity
            similarities = self._cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)
            k = min(k, len(self.items))  # Ensure k is not larger than the number of items
            indices = np.argsort(similarities.flatten())[-k:][::-1]
            return [(self.items[i], float(similarities.flatten()[i])) for i in indices]

        # Approximate search using k-means
        # Get distances to all cluster centers
        distances = self.kmeans.transform(query_embedding.reshape(1, -1))[0]

        # Get indices of the n_probe_clusters closest clusters
        closest_clusters = np.argsort(distances)[:n_probe_clusters]

        # Collect all indices from the closest clusters
        all_indices: NDArray[np.int64] = np.concatenate(
            [np.where(self.clusters == cluster)[0] for cluster in closest_clusters]
        )

        # If no indices found (shouldn't happen but just in case), fall back to exact search
        if len(all_indices) == 0:
            logger.warning(f"No items found in the {n_probe_clusters} closest clusters. Falling back to exact search.")
            return self.search(query_embedding, k, use_approx=False)

        # Calculate similarities only for items in the selected clusters
        all_indices = np.array(all_indices)
        cluster_similarities = self._cosine_similarity(query_embedding.reshape(1, -1), self.embeddings[all_indices])

        # Get top k results from the combined clusters
        k = min(k, len(all_indices))
        top_k_indices = np.argsort(cluster_similarities.flatten())[-k:][::-1]

        return [(self.items[all_indices[i]], float(cluster_similarities.flatten()[i])) for i in top_k_indices]

    @staticmethod
    def _cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
        """Calculate cosine similarity between normalized vectors"""
        # Since vectors are normalized, cosine similarity is just the dot product
        return np.dot(a, b.T)

    def add_items(
        self,
        new_embeddings: NDArray[np.floating],
        new_items: List[str],
        save_dir: Optional[str] = None,
        overwrite: bool = True,
    ):
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
        # Normalize new embeddings
        normalized_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1)[:, np.newaxis]

        # Append to existing embeddings and items
        self.embeddings = np.vstack([self.embeddings, normalized_embeddings])
        self.items.extend(new_items)

        # Update clusters if index exists
        if self.kmeans is not None:
            # Predict clusters for new items
            new_clusters = self.kmeans.predict(normalized_embeddings)
            assert self.clusters is not None, "clusters should not be None here"
            self.clusters = np.concatenate([self.clusters.astype(np.int64), new_clusters])

        # Save updated index if save_dir is provided
        if save_dir:
            self.save_index(save_dir, overwrite=overwrite)
