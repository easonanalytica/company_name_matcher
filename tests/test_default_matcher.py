import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import pytest
import pandas as pd
from company_name_matcher import CompanyNameMatcher
import re

@pytest.fixture
def default_matcher():
    def preprocess_name(name):
        return re.sub(r'[^a-zA-Z0-9\s]', '', name.lower()).strip()

    return CompanyNameMatcher(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        preprocess_fn=preprocess_name
    )

def test_basic_company_comparison(default_matcher):
    test_cases = [
        ("Apple", "Microsoft Corporation", 0.34, 0.2),  # Expected low similarity
        ("Apple", "Apple Inc", 0.90, 0.2),             # Expected high similarity
        ("Apple", "Apple Computer Inc", 0.80, 0.2),    # Expected high similarity
        ("Google", "Alphabet Inc", 0.20, 0.2),         # Expected medium similarity
    ]

    for company1, company2, expected_score, tolerance in test_cases:
        similarity = default_matcher.compare_companies(company1, company2)
        assert abs(similarity - expected_score) < tolerance, \
            f"Similarity between {company1} and {company2} was {similarity}, expected around {expected_score}"

def test_multilingual_support(default_matcher):
    test_cases = [
        ("Apple", "苹果公司", 0.30, 0.1),          # English-Chinese
        ("Microsoft", "マイクロソフト", 0.30, 0.1), # English-Japanese
        ("Google", "구글", 0.30, 0.1),             # English-Korean
    ]

    for company1, company2, expected_score, tolerance in test_cases:
        similarity = default_matcher.compare_companies(company1, company2)
        assert abs(similarity - expected_score) < tolerance, \
            f"Multilingual similarity between {company1} and {company2} was {similarity}, expected around {expected_score}"

def test_index_operations(default_matcher, tmp_path):
    # Test data
    companies = [
        "Apple Inc",
        "Microsoft Corporation",
        "Google LLC",
        "Amazon.com Inc",
        "Facebook Inc"
    ]

    # Test index building
    index_dir = tmp_path / "test_index"
    default_matcher.build_index(companies, n_clusters=2, save_dir=str(index_dir))

    # Verify index files were created
    assert os.path.exists(index_dir / "embeddings.h5"), "Embeddings file not created"
    assert os.path.exists(index_dir / "kmeans_model.joblib"), "KMeans model file not created"

    # Test index loading
    new_matcher = CompanyNameMatcher("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    new_matcher.load_index(str(index_dir))

    # Test exact search
    matches = new_matcher.find_matches("Apple", threshold=0.7, use_approx=False)
    assert len(matches) > 0, "No matches found for 'Apple'"
    assert matches[0][0] == "Apple Inc", f"Expected 'Apple Inc', got {matches[0][0]}"

    # Test approximate search
    approx_matches = new_matcher.find_matches("Apple", threshold=0.7, use_approx=True, k=1)
    assert len(approx_matches) > 0, "No approximate matches found for 'Apple'"
    assert approx_matches[0][0] == "Apple Inc", f"Expected 'Apple Inc', got {approx_matches[0][0]}"

def test_embedding_generation(default_matcher):
    # Test single embedding
    company_name = "Apple Inc"
    embedding = default_matcher.get_embedding(company_name)

    assert embedding is not None, "Embedding should not be None"
    assert len(embedding.shape) == 1, "Embedding should be 1-dimensional"
    assert embedding.shape[0] == 384, "Embedding dimension should be 384"

def test_index_expansion(default_matcher, tmp_path):
    # Initial companies
    initial_companies = ["Apple Inc", "Microsoft Corporation"]

    # Build initial index
    index_dir = tmp_path / "expansion_test_index"
    default_matcher.build_index(initial_companies, n_clusters=2, save_dir=str(index_dir))

    # New companies to add
    new_companies = ["Google LLC", "Amazon.com Inc"]

    # Expand index
    default_matcher.expand_index(new_companies, save_dir=str(index_dir))

    # Test finding matches for both original and new companies
    for company in initial_companies + new_companies:
        matches = default_matcher.find_matches(company, threshold=0.9, use_approx=False)
        assert len(matches) > 0, f"No matches found for {company}"
        assert company in [match[0] for match in matches], f"Exact match not found for {company}"
