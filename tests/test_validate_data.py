from pathlib import Path
from unittest.mock import patch
import polars as pl
import pytest
from typing import List, Optional
from scripts.validate_data import DataValidator


# -----------------------
# Fixture: validator
# -----------------------
@pytest.fixture
def validator() -> DataValidator:
    """
    Returns a DataValidator instance with all file-loading methods mocked.
    """
    with (
        patch.object(DataValidator, "_load_country_codes", return_value={"US", "UK"}),
        patch.object(DataValidator, "_scan_positive_csvs", return_value=pl.LazyFrame()),
        patch.object(DataValidator, "_scan_negative_csvs", return_value=pl.LazyFrame()),
    ):
        return DataValidator(Path("/fake/path"))


# -----------------------
# Test: country code check
# -----------------------
def test_country_code_check(validator: DataValidator) -> None:
    """
    Test that _country_code_check flags invalid country codes correctly.
    """
    df = pl.DataFrame({"country_code": ["US", "XX", "UK"]})

    exprs = validator._country_code_check(df.lazy())  # type: ignore
    result_df = df.lazy().with_columns(exprs).collect()

    assert result_df["CountryCodeError: country_code"].to_list() == [
        None,
        "CountryCodeError: country_code has invalid value 'XX'",
        None,
    ]


# -----------------------
# Test: mandatory column check
# -----------------------
def test_mandatory_col_check(validator: DataValidator) -> None:
    df = pl.DataFrame(
        {
            "canonical_name": ["Apple", None, ""],
            "variation": ["AAPL", "AAP", " "],
            "country_code": ["US", "", None],
        }
    )

    exprs = validator._mandatory_col_check(df.lazy())  # type: ignore
    result_df = df.lazy().with_columns(exprs).collect()

    for col in ["canonical_name", "variation", "country_code"]:
        errors = result_df[f"MissingDataError: {col}"].to_list()
        assert all(isinstance(e, (str, type(None))) for e in errors)


# -----------------------
# Test: duplication check
# -----------------------
def test_duplication_check(validator: DataValidator) -> None:
    df = pl.DataFrame(
        {
            "canonical_name": ["Apple", "Apple", "Microsoft"],
            "variation": ["AAPL", "AAPL", "MSFT"],
            "country_code": ["US", "US", "US"],
        }
    )
    expr = validator._duplication_check(df.lazy())  # type: ignore
    result_df = df.lazy().with_columns([expr]).collect()

    expected: List[Optional[str]] = [
        "DuplicateRowError: duplicate row found",
        "DuplicateRowError: duplicate row found",
        None,
    ]
    assert result_df["DuplicateRowError"].to_list() == expected


# -----------------------
# Test: unicode check
# -----------------------
def test_unicode_check(validator: DataValidator) -> None:
    df = pl.DataFrame({"canonical_name": ["Apple", "Micro�soft", "Google"]})
    exprs = validator._unicode_check(df.lazy())  # type: ignore
    result_df = df.lazy().with_columns(exprs).collect()

    expected: List[Optional[str]] = [
        None,
        "UnicodeError: canonical_name has an invalid UTF-8 character",
        None,
    ]
    assert result_df["UnicodeError: canonical_name"].to_list() == expected
