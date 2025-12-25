import os
import pytest
import pandas as pd
import re
from rapidfuzz import fuzz
from company_name_matcher import CompanyNameMatcher
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Callable, List
from numpy.typing import NDArray
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture
def test_data():
    """Load the test data from CSV file."""
    return pd.read_csv("tests/test_data.csv")


@pytest.fixture
def rapid_fuzz_matcher():
    """Create a RapidFuzz matcher function."""

    def matcher(x1, x2):
        return fuzz.ratio(x1, x2) / 100

    return matcher


@pytest.fixture
def default_matcher():
    """Create a default CompanyNameMatcher."""

    def preprocess_name(name):
        return re.sub(r"[^a-zA-Z0-9\s]", "", name.lower()).strip()

    return CompanyNameMatcher("paraphrase-multilingual-MiniLM-L12-v2", preprocess_fn=preprocess_name)


@pytest.fixture
def finetuned_matcher():
    """Create a fine-tuned CompanyNameMatcher."""

    def preprocess_name(name):
        return "$" + name.strip()  # v2 model pretrained tokens

    return CompanyNameMatcher("easonanalytica/cnm-multilingual-small-v2", preprocess_fn=preprocess_name)


def test_model_comparison(test_data: pd.DataFrame, rapid_fuzz_matcher: Callable[[str,str], float], default_matcher: CompanyNameMatcher, finetuned_matcher: CompanyNameMatcher) -> None:
    """
    Compare the performance of RapidFuzz, default CompanyNameMatcher,
    and fine-tuned CompanyNameMatcher on test data.
    """
    # Thresholds for each model (can be tuned for optimal performance)
    rapid_fuzz_threshold = 0.8
    default_matcher_threshold = 0.7
    finetuned_matcher_threshold = 0.7

    # Store predictions and actual values
    y_true: NDArray[np.int64] = test_data["Targets"].to_numpy()

    # Get predictions for each model
    rapid_fuzz_preds: List[float] = []
    default_matcher_preds: List[float] = []
    finetuned_matcher_preds: List[float] = []

    # Calculate predictions for each pair in the test data
    for _, row in test_data.iterrows():
        name1, name2 = row["Name_x"], row["Name_y"]

        # RapidFuzz prediction
        rapid_fuzz_score = rapid_fuzz_matcher(name1, name2)
        rapid_fuzz_preds.append(1 if rapid_fuzz_score >= rapid_fuzz_threshold else 0)

        # Default matcher prediction
        default_score = default_matcher.compare_companies(name1, name2)
        default_matcher_preds.append(1 if default_score >= default_matcher_threshold else 0)

        # Fine-tuned matcher prediction
        finetuned_score = finetuned_matcher.compare_companies(name1, name2)
        finetuned_matcher_preds.append(1 if finetuned_score >= finetuned_matcher_threshold else 0)

    # Calculate metrics for each model
    models = {
        "RapidFuzz": rapid_fuzz_preds,
        "Default Matcher": default_matcher_preds,
        "Fine-tuned Matcher": finetuned_matcher_preds,
    }

    print("\nModel Comparison Results:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 80)

    for model_name, predictions in models.items():
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)

        print(f"{model_name:<20} {accuracy:.4f}     {precision:.4f}     {recall:.4f}     {f1:.4f}")

        # Assert that each model has reasonable performance
        assert accuracy > 0.5, f"{model_name} accuracy should be better than random guessing"
        assert precision > 0, f"{model_name} precision should be greater than 0"
        assert recall > 0, f"{model_name} recall should be greater than 0"
        assert f1 > 0, f"{model_name} F1 score should be greater than 0"

    print("-" * 80)

    # Optional: Print some example comparisons
    print("\nExample Comparisons:")
    print("-" * 80)
    print(f"{'Name 1':<40} {'Name 2':<40} {'Actual':<8} {'RapidFuzz':<10} {'Default':<10} {'Fine-tuned':<10}")
    print("-" * 80)

    # Print first 5 examples
    for i in range(min(5, len(test_data))):
        row = test_data.iloc[i]
        name1, name2 = row["Name_x"], row["Name_y"]
        actual = row["Targets"]

        rapid_fuzz_score = rapid_fuzz_matcher(name1, name2)
        default_score = default_matcher.compare_companies(name1, name2)
        finetuned_score = finetuned_matcher.compare_companies(name1, name2)

        print(
            f"{name1[:38]:<40} {name2[:38]:<40} {actual:<8} {rapid_fuzz_score:.4f}    {default_score:.4f}\
            {finetuned_score:.4f}"
        )
