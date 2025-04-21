import h5py
import logging
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from joblib import dump, load
import os

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embeddings: np.ndarray, items: List[str]):
        if len(embeddings) == 1 and embeddings[0][0] == 0 and items == ["dummy"]:
            # Special case for dummy initialization
            self.embeddings = embeddings
            self.items = items
        else:
            # Normal case - normalize the embeddings
            self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
            self.items = items
        self.kmeans = None
        self.clusters = None
        
    def build_index(self, n_clusters: int = 100, save_path: str = None, overwrite: bool = True):
        """
        Build k-means clustering index for approximate search
        
        Args:
            n_clusters: Number of clusters for KMeans
            save_path: Optional directory path to save the index
            overwrite: Whether to overwrite existing index files (default: True)
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
        Save the index components to disk
        
        Args:
            save_path: Directory path to save the index
            overwrite: Whether to overwrite existing index files (default: True)
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
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('embeddings', data=self.embeddings, compression='gzip')
            dt = h5py.special_dtype(vlen=str)
            items_dataset = f.create_dataset('items', (len(self.items),), dtype=dt)
            items_dataset[:] = self.items
            
        # Save KMeans model and clusters using joblib
        dump({
            'kmeans': self.kmeans,
            'clusters': self.clusters
        }, model_path)
        
        logger.info(f"Index saved to {save_path}")
    
    def load_index(self, load_path: str):
        """Load the index components from disk"""
        h5_path = os.path.join(load_path, "embeddings.h5")
        model_path = os.path.join(load_path, "kmeans_model.joblib")
        

        if not os.path.exists(h5_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Index files not found in {load_path}")
        
        # Load embeddings and items from h5py
        with h5py.File(h5_path, 'r') as f:
            self.embeddings = f['embeddings'][:]
            # Decode byte strings to regular strings
            self.items = [item.decode('utf-8') if isinstance(item, bytes) else item 
                         for item in f['items'][:]]
        
        # Load KMeans model and clusters from joblib
        data = load(model_path)
        self.kmeans = data['kmeans']
        self.clusters = data['clusters']

    def search(self, query_embedding: np.ndarray, k: int = 5, use_approx: bool = False, n_probe_clusters: int = 3) -> List[Tuple[str, float]]:
        """
        Search for similar items using either exact or approximate k-means search
        
        Args:
            query_embedding: The query vector to search for
            k: Number of results to return
            use_approx: Whether to use approximate search with k-means
            n_probe_clusters: Number of closest clusters to search (only for approximate search)
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
            return [(self.items[i], similarities.flatten()[i]) for i in indices]
        
        # Approximate search using k-means
        # Get distances to all cluster centers
        distances = self.kmeans.transform(query_embedding.reshape(1, -1))[0]
        
        # Get indices of the n_probe_clusters closest clusters
        closest_clusters = np.argsort(distances)[:n_probe_clusters]
        
        # Collect all indices from the closest clusters
        all_indices = []
        for cluster in closest_clusters:
            cluster_indices = np.where(self.clusters == cluster)[0]
            all_indices.extend(cluster_indices)
        
        # If no indices found (shouldn't happen but just in case), fall back to exact search
        if len(all_indices) == 0:
            logger.warning(f"No items found in the {n_probe_clusters} closest clusters. Falling back to exact search.")
            return self.search(query_embedding, k, use_approx=False)
        
        # Calculate similarities only for items in the selected clusters
        all_indices = np.array(all_indices)
        cluster_similarities = self._cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings[all_indices]
        )
        
        # Get top k results from the combined clusters
        k = min(k, len(all_indices))
        top_k_indices = np.argsort(cluster_similarities.flatten())[-k:][::-1]
        
        return [(self.items[all_indices[i]], 
                cluster_similarities.flatten()[i]) 
                for i in top_k_indices]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between normalized vectors"""
        # Since vectors are normalized, cosine similarity is just the dot product
        return np.dot(a, b.T) 

    def add_items(self, new_embeddings: np.ndarray, new_items: List[str], save_dir: str = None, overwrite: bool = True):
        """
        Add new items to the existing index
        
        Args:
            new_embeddings: Embeddings for new items to add
            new_items: List of new items to add
            save_dir: Optional directory path to save the updated index
            overwrite: Whether to overwrite existing index files (default: True)
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
            self.clusters = np.concatenate([self.clusters, new_clusters])
        
        # Save updated index if save_dir is provided
        if save_dir:
            self.save_index(save_dir, overwrite=overwrite) 