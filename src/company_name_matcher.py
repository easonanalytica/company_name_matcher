from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np

class CompanyNameMatcher:
    def __init__(self, model_path: str = "models/fine_tuned_model"):
        self.embedder = SentenceTransformer(model_path)

    def _preprocess_company_name(self, name: str) -> str:
        """add special tokens to the company name."""
        return f"${name.strip()}$"

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
        return self._cosine_similarity(embedding_a, embedding_b)

    def find_matches(self, target_company: str, company_list: List[str], threshold: float = 0.9) -> List[Tuple[str, float]]:
        """find matches for a target company in a list of companies."""
        target_embedding = self.get_embedding(target_company)
        company_embeddings = self.get_embeddings(company_list)

        similarities = self._cosine_similarity(target_embedding.reshape(1, -1), company_embeddings)
        similarities = similarities.flatten()  # Flatten the result to a 1D array

        matches = [(company, similarity) for company, similarity in zip(company_list, similarities) if similarity >= threshold]
        return sorted(matches, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """calculate cosine similarity between two vectors or between a vector and a matrix."""
        print(f"Input shapes: a={a.shape}, b={b.shape}")

        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)

        print(f"Reshaped input shapes: a={a.shape}, b={b.shape}")

        # compute the dot product
        dot_product = np.dot(a, b.T)

        # compute the L2 norm
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)

        # compute the cosine similarity
        result = dot_product / (norm_a[:, np.newaxis] * norm_b)

        print(f"Result shape: {result.shape}")

        return result
