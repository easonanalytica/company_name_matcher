import os
import pytest
from company_name_matcher import CompanyNameMatcher
import time
import re
from typing import List
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture
def default_matcher():
    def preprocess_name(name):
        return re.sub(r"[^a-zA-Z0-9\s]", "", name.lower()).strip()

    return CompanyNameMatcher("paraphrase-multilingual-MiniLM-L12-v2", preprocess_fn=preprocess_name)


@pytest.fixture
def test_companies():
    return [
        "Apple Inc",
        "Microsoft Corporation",
        "Google LLC",
        "Amazon.com Inc",
        "Facebook Inc",
        "Tesla Motors",
        "IBM Corporation",
        "Intel Corporation",
        "Oracle Corporation",
        "Cisco Systems",
    ]


def test_find_matches_single_company(default_matcher, test_companies, tmp_path):
    # Build index
    index_dir = tmp_path / "test_batch_index"
    default_matcher.build_index(test_companies, n_clusters=2, save_dir=str(index_dir))

    # Test with a single company
    single_result = default_matcher.find_matches("Apple", threshold=0.7)

    # Verify result structure for single company
    assert isinstance(single_result, list), "Result should be a list"
    assert all(
        isinstance(item, tuple) and len(item) == 2 for item in single_result
    ), "Each item should be a (company, score) tuple"
    assert len(single_result) > 0, "Should find at least one match"


def test_find_matches_multiple_companies(default_matcher: CompanyNameMatcher, test_companies: List[str], tmp_path: Path):
    # Build index
    index_dir = tmp_path / "test_batch_index"
    default_matcher.build_index(test_companies, n_clusters=2, save_dir=str(index_dir))

    # Test queries
    queries = ["Apple", "Microsoft", "Google"]

    # Test with multiple companies
    multi_result = default_matcher.find_matches(queries, threshold=0.7)

    # Verify result structure for multiple companies
    assert isinstance(multi_result, list), "Result should be a list"
    assert len(multi_result) == len(queries), "Should return one result per query"
    assert all(isinstance(item, list) for item in multi_result), "Each result should be a list"
    assert all(
        isinstance(match, tuple) and len(match) == 2 for result in multi_result for match in result
    ), "Each match should be a (company, score) tuple"


def test_batch_processing_performance(default_matcher, tmp_path):
    # Generate a larger set of test companies
    large_company_set = [f"Company {i}" for i in range(100)]

    # Build index
    index_dir = tmp_path / "test_perf_index"
    default_matcher.build_index(large_company_set, n_clusters=10, save_dir=str(index_dir))

    # Generate test queries
    test_queries = [f"Company {i}" for i in range(0, 50, 5)]  # 10 queries

    # Test sequential processing
    start_time = time.time()
    sequential_results = default_matcher.find_matches(test_queries, threshold=0.7, n_jobs=1)
    sequential_time = time.time() - start_time

    # Test parallel processing
    start_time = time.time()
    parallel_results = default_matcher.find_matches(test_queries, threshold=0.7, n_jobs=2)
    parallel_time = time.time() - start_time

    # Verify results are the same
    assert len(sequential_results) == len(parallel_results), "Sequential and parallel results should have same length"

    # Check that results are similar (may not be identical due to floating point differences)
    for seq_matches, par_matches in zip(sequential_results, parallel_results):
        assert len(seq_matches) == len(par_matches), "Number of matches should be the same"

        # Check companies match (order might differ slightly)
        seq_companies = {match[0] for match in seq_matches}
        par_companies = {match[0] for match in par_matches}
        assert seq_companies == par_companies, "Same companies should be matched"

    # Print performance comparison (not an assertion as performance depends on hardware)
    print("\nPerformance comparison:")
    print(f"Sequential processing time: {sequential_time:.4f}s")
    print(f"Parallel processing time: {parallel_time:.4f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")


def test_batch_find_matches_backward_compatibility(default_matcher, test_companies, tmp_path):
    # Build index
    index_dir = tmp_path / "test_compat_index"
    default_matcher.build_index(test_companies, n_clusters=2, save_dir=str(index_dir))

    # Test queries
    queries = ["Apple", "Microsoft", "Google"]

    # Test with both methods
    new_results = default_matcher.find_matches(queries, threshold=0.7)
    old_results = default_matcher.batch_find_matches(queries, threshold=0.7)

    # Verify results are the same
    assert len(new_results) == len(old_results), "Results from both methods should have same length"

    # Check that results are identical
    for new_matches, old_matches in zip(new_results, old_results):
        assert len(new_matches) == len(old_matches), "Number of matches should be the same"

        for new_match, old_match in zip(new_matches, old_matches):
            assert new_match[0] == old_match[0], "Matched company should be the same"
            assert abs(new_match[1] - old_match[1]) < 1e-6, "Similarity scores should be the same"


def test_n_jobs_parameter(default_matcher, test_companies, tmp_path):
    # Build index
    index_dir = tmp_path / "test_njobs_index"
    default_matcher.build_index(test_companies, n_clusters=2, save_dir=str(index_dir))

    # Test queries
    queries = ["Apple", "Microsoft", "Google", "Amazon", "Facebook"]

    # Test with different n_jobs values
    results_1 = default_matcher.find_matches(queries, threshold=0.7, n_jobs=1)
    results_2 = default_matcher.find_matches(queries, threshold=0.7, n_jobs=2)
    results_neg = default_matcher.find_matches(queries, threshold=0.7, n_jobs=-1)  # All cores

    # Verify all results have the same structure
    assert (
        len(results_1) == len(results_2) == len(results_neg) == len(queries)
    ), "All methods should return same number of results"

    # Check that results contain the same matches (order might differ slightly due to parallel execution)
    for i in range(len(queries)):
        companies_1 = {match[0] for match in results_1[i]}
        companies_2 = {match[0] for match in results_2[i]}
        companies_neg = {match[0] for match in results_neg[i]}

        assert companies_1 == companies_2 == companies_neg, f"Same companies should be matched for query {queries[i]}"
