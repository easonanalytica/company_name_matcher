from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans

class VectorStore:
    def __init__(self, embeddings: np.ndarray, items: List[str]):

        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis] # normalize embeddings to unit length
        self.items = items
        self.kmeans = None
        self.clusters = None
        
    def build_index(self, n_clusters: int = 100):
        """Build k-means clustering index for approximate search"""
        if len(self.items) < n_clusters:
            n_clusters = max(1, len(self.items) // 2)
        
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.clusters = self.kmeans.fit_predict(self.embeddings)
        
    def search(self, query_embedding: np.ndarray, k: int = 5, use_approx: bool = False) -> List[Tuple[str, float]]:
        """Search for similar items using either exact or approximate k-means search"""
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if not use_approx or self.kmeans is None:
            # Exact search using cosine similarity
            similarities = self._cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)
            indices = np.argsort(similarities.flatten())[-k:][::-1]
            return [(self.items[i], similarities.flatten()[i]) for i in indices]
        
        # Approximate search using k-means
        cluster = self.kmeans.predict(query_embedding.reshape(1, -1))[0]
        cluster_indices = np.where(self.clusters == cluster)[0]
        
        # Calculate similarities only for items in the same cluster
        cluster_similarities = self._cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings[cluster_indices]
        )
        
        # Get top k results from the cluster
        k = min(k, len(cluster_indices))
        top_k_indices = np.argsort(cluster_similarities.flatten())[-k:][::-1]
        
        return [(self.items[cluster_indices[i]], 
                cluster_similarities.flatten()[i]) 
                for i in top_k_indices]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between normalized vectors"""
        # Since vectors are normalized, cosine similarity is just the dot product
        return np.dot(a, b.T) 