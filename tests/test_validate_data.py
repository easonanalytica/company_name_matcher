from __future__ import annotations

import pytest
import polars as pl
from pathlib import Path

from validate_data import (
    DataValidator,
    FileNameError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def init_dirs(root: Path) -> None:
    """Create required folder structure."""
    (root / "data" / "positive").mkdir(parents=True, exist_ok=True)
    (root / "data" / "negative").mkdir(parents=True, exist_ok=True)
    (root / "data" / "_reference").mkdir(parents=True, exist_ok=True)


def write_country_codes(root: Path) -> Path:
    """Write a valid country code reference CSV file."""
    init_dirs(root)
    df = pl.DataFrame({"ISO2": ["US", "DE", "FR"]})
    path = root / "data" / "_reference" / "countrycode.csv"
    df.write_csv(path)
    return path


def write_positive_parquet(root: Path, filename: str, df: pl.DataFrame) -> Path:
    """Write a valid positive parquet file."""
    path = root / "data" / "positive" / filename
    df.write_parquet(path)
    return path


def write_negative_parquet(root: Path, filename: str, df: pl.DataFrame) -> Path:
    """Write a valid negative parquet file."""
    path = root / "data" / "negative" / filename
    df.write_parquet(path)
    return path


def dummy_positive(root: Path) -> Path:
    """Placeholder valid positive parquet."""
    df = pl.DataFrame(
        {
            "canonical_name": ["ABC Limited"],
            "variation": ["ABC"],
            "country_code": ["ZZ"],
            "source": ["Test"],
        }
    )
    return write_positive_parquet(root, "ZZ.parquet", df)


def dummy_negative(root: Path) -> Path:
    """Placeholder valid negative parquet."""
    df = pl.DataFrame(
        {
            "canonical_name_x": ["AA Corp"],
            "canonical_name_y": ["BB LLC"],
            "country_code_x": ["AA"],
            "country_code_y": ["BB"],
            "remark": ["Test"],
        }
    )
    return write_negative_parquet(root, "AA_BB.parquet", df)


def run_titlecase_check(df: pl.DataFrame):
    """
    Helper to evaluate the whitespace expressions from _titlecase_check.
    """
    lf = df.lazy()
    validator = DataValidator.__new__(DataValidator)
    exprs = validator._titlecase_check(lf)  # type: ignore
    result = lf.with_columns(exprs).collect()
    return result


def run_whitespace_check(df: pl.DataFrame):
    """
    Helper to evaluate the whitespace expressions from _whitespace_check.
    """
    lf = df.lazy()
    validator = DataValidator.__new__(DataValidator)
    exprs = validator._whitespace_check(lf)  # type: ignore
    result = lf.with_columns(exprs).collect()
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_country_codes(tmp_path: Path) -> None:
    write_country_codes(tmp_path)
    dummy_positive(tmp_path)
    dummy_negative(tmp_path)

    validator = DataValidator(tmp_path)
    assert validator.country_codes == {"US", "DE", "FR"}


def test_country_code_validity_check(tmp_path: Path) -> None:
    write_country_codes(tmp_path)

    df = pl.DataFrame(
        {
            "country_code": ["US", "XX", "DE"],
            "canonical_name": ["a", "b", "c"],
            "variation": ["x", "y", "z"],
            "source": ["s", "s", "s"],
        }
    )
    write_positive_parquet(tmp_path, "US.parquet", df)
    dummy_negative(tmp_path)

    validator = DataValidator(tmp_path)
    lf = validator.positive_parquets

    exprs = validator._country_code_validity_check(lf)  # type: ignore
    assert isinstance(exprs, list)
    assert all(isinstance(e, pl.Expr) for e in exprs)


def test_country_code_filename_match_check_positive(tmp_path: Path) -> None:
    write_country_codes(tmp_path)

    df = pl.DataFrame(
        {
            "canonical_name": ["a"],
            "variation": ["b"],
            "country_code": ["US"],
            "source": ["s"],
        }
    )
    write_positive_parquet(tmp_path, "US.parquet", df)
    dummy_negative(tmp_path)

    validator = DataValidator(tmp_path)
    lf = validator.positive_parquets

    exprs = validator._country_code_filename_match_check(lf, "US.parquet")  # type: ignore
    assert isinstance(exprs, list)
    assert all(isinstance(e, pl.Expr) for e in exprs)


def test_country_code_filename_match_check_negative(tmp_path: Path) -> None:
    write_country_codes(tmp_path)

    df = pl.DataFrame(
        {
            "canonical_name_x": ["a"],
            "canonical_name_y": ["b"],
            "country_code_x": ["US"],
            "country_code_y": ["DE"],
            "remark": ["r"],
        }
    )
    write_negative_parquet(tmp_path, "US_DE.parquet", df)
    dummy_positive(tmp_path)

    validator = DataValidator(tmp_path)
    lf = validator.negative_parquets

    exprs = validator._country_code_filename_match_check(lf, "US_DE.parquet", negative=True)  # type: ignore
    assert isinstance(exprs, list)
    assert all(isinstance(e, pl.Expr) for e in exprs)


def test_scan_positive_parquets(tmp_path: Path) -> None:
    write_country_codes(tmp_path)

    df = pl.DataFrame(
        {
            "canonical_name": ["Apple"],
            "variation": ["Apple Inc."],
            "country_code": ["US"],
            "source": ["manual"],
        }
    )
    write_positive_parquet(tmp_path, "US.parquet", df)
    dummy_negative(tmp_path)

    validator = DataValidator(tmp_path)
    lf = validator.positive_parquets

    assert lf.collect().height == 1


def test_scan_negative_parquets(tmp_path: Path) -> None:
    write_country_codes(tmp_path)
    dummy_positive(tmp_path)

    df = pl.DataFrame(
        {
            "canonical_name_x": ["a"],
            "canonical_name_y": ["b"],
            "country_code_x": ["US"],
            "country_code_y": ["DE"],
            "remark": ["r"],
        }
    )
    write_negative_parquet(tmp_path, "US_DE.parquet", df)

    validator = DataValidator(tmp_path)
    lf = validator.negative_parquets

    assert lf.collect().height == 1


def test_mandatory_col_check(tmp_path: Path) -> None:
    write_country_codes(tmp_path)
    dummy_positive(tmp_path)
    dummy_negative(tmp_path)

    validator = DataValidator(tmp_path)

    df = pl.DataFrame({"canonical_name": ["a"], "variation": ["b"], "country_code": ["US"]})
    lf = df.lazy()

    exprs = validator._mandatory_col_check(lf)  # type: ignore
    assert isinstance(exprs, list)
    assert exprs


def test_duplication_check(tmp_path: Path) -> None:
    write_country_codes(tmp_path)
    dummy_positive(tmp_path)
    dummy_negative(tmp_path)

    validator = DataValidator(tmp_path)

    df = pl.DataFrame(
        {
            "canonical_name": ["a", "a"],
            "variation": ["b", "b"],
            "country_code": ["US", "US"],
        }
    )
    lf = df.lazy()

    expr = validator._duplication_check(lf)  # type: ignore
    result = lf.with_columns(expr).collect()

    assert "DuplicateRowError" in result.columns


def test_difference_check(tmp_path: Path) -> None:
    write_country_codes(tmp_path)
    dummy_positive(tmp_path)
    dummy_negative(tmp_path)

    validator = DataValidator(tmp_path)

    df = pl.DataFrame({"canonical_name": ["Acme"], "variation": ["acme"]})
    lf = df.lazy()

    expr = validator._difference_check(lf)  # type: ignore
    result = lf.with_columns(expr).collect()

    assert "DuplicateNameError" in result.columns


def test_concatenate_errors(tmp_path: Path) -> None:
    write_country_codes(tmp_path)
    dummy_positive(tmp_path)
    dummy_negative(tmp_path)

    validator = DataValidator(tmp_path)

    df = pl.DataFrame({"Error1": [None, "X"], "Error2": ["A", None], "canonical_name": ["a", "b"]})
    lf = df.lazy()

    expr = validator._concatenate_errors(lf)  # type: ignore
    out = lf.with_columns(expr).collect()

    assert "Errors" in out.columns
    assert out["Errors"][0] == "A"
    assert out["Errors"][1] == "X"


def test_filename_check_positive_ok(tmp_path: Path) -> None:
    write_country_codes(tmp_path)
    dummy_negative(tmp_path)

    df = pl.DataFrame(
        {
            "canonical_name": ["a"],
            "variation": ["b"],
            "country_code": ["US"],
            "source": ["s"],
        }
    )
    write_positive_parquet(tmp_path, "US.parquet", df)

    validator = DataValidator(tmp_path)

    # Should not raise
    validator._filename_check(tmp_path / "data" / "positive")  # type: ignore


def test_filename_check_negative_fail(tmp_path: Path) -> None:
    write_country_codes(tmp_path)
    dummy_positive(tmp_path)
    dummy_negative(tmp_path)

    # Bad filename
    bad = tmp_path / "data" / "negative" / "BADNAME.parquet"
    df = pl.DataFrame(
        {
            "canonical_name_x": ["A Corp"],
            "canonical_name_y": ["B LLC"],
            "country_code_x": ["AA"],
            "country_code_y": ["BB"],
            "remark": ["different entities"],
        }
    )
    df.write_parquet(bad)

    with pytest.raises(FileNameError):
        validator = DataValidator(tmp_path)
        validator._filename_check(tmp_path / "data" / "negative", negative=True)  # type: ignore


def test_validate_no_errors(tmp_path: Path) -> None:
    write_country_codes(tmp_path)

    dfp = pl.DataFrame(
        {
            "canonical_name": ["Apple"],
            "variation": ["Apple Inc"],
            "country_code": ["US"],
            "source": ["manual"],
        }
    )
    write_positive_parquet(tmp_path, "US.parquet", dfp)

    dfn = pl.DataFrame(
        {
            "canonical_name_x": ["Apple"],
            "canonical_name_y": ["Banana"],
            "country_code_x": ["DE"],
            "country_code_y": ["US"],
            "remark": ["different entities"],
        }
    )
    write_negative_parquet(tmp_path, "DE_US.parquet", dfn)

    validator = DataValidator(tmp_path)

    # Should not raise any errors
    validator.validate()


def test_titlecase_check_pass() -> None:
    df = pl.DataFrame({"variation": ["Apple Inc."]})
    out = run_titlecase_check(df)

    assert out["CaseError: variation"][0] is None


def test_titlecase_check_fail() -> None:
    df = pl.DataFrame({"variation": ["apple Inc."]})
    out = run_titlecase_check(df)

    assert out["CaseError: variation"][0] == "CaseError: variation is not in titlecase"


def test_leading_whitespace_detected():
    df = pl.DataFrame({"canonical_name": ["  Apple Inc."]})
    out = run_whitespace_check(df)

    assert (
        out["LeadingWhiteSpaceError: canonical_name"][0]
        == "LeadingWhiteSpaceError: canonical_name has a leading whitespace"
    )
    assert out["TrailingWhiteSpaceError: canonical_name"][0] is None
    assert out["DoubleWhiteSpaceError: canonical_name"][0] is None


def test_trailing_whitespace_detected() -> None:
    df = pl.DataFrame({"canonical_name": ["Apple Inc.  "]})
    out = run_whitespace_check(df)

    assert out["LeadingWhiteSpaceError: canonical_name"][0] is None
    assert (
        out["TrailingWhiteSpaceError: canonical_name"][0]
        == "TrailingWhiteSpaceError: canonical_name has a trailing whitespace"
    )
    assert out["DoubleWhiteSpaceError: canonical_name"][0] is None


def test_double_whitespace_detected():
    df = pl.DataFrame({"canonical_name": ["Apple  Inc."]})
    out = run_whitespace_check(df)

    assert out["LeadingWhiteSpaceError: canonical_name"][0] is None
    assert out["TrailingWhiteSpaceError: canonical_name"][0] is None
    assert (
        out["DoubleWhiteSpaceError: canonical_name"][0]
        == "DoubleWhiteSpaceError: canonical_name has a multiple whitespaces"
    )


def test_newlines_are_detected():
    df = pl.DataFrame({"canonical_name": ["\nApple Inc.\n"]})
    out = run_whitespace_check(df)

    assert (
        out["LeadingWhiteSpaceError: canonical_name"][0]
        == "LeadingWhiteSpaceError: canonical_name has a leading whitespace"
    )
    assert (
        out["TrailingWhiteSpaceError: canonical_name"][0]
        == "TrailingWhiteSpaceError: canonical_name has a trailing whitespace"
    )
    assert out["DoubleWhiteSpaceError: canonical_name"][0] is None


def test_no_whitespace_errors():
    df = pl.DataFrame({"canonical_name": ["Apple Inc."]})
    out = run_whitespace_check(df)

    assert out["LeadingWhiteSpaceError: canonical_name"][0] is None
    assert out["TrailingWhiteSpaceError: canonical_name"][0] is None
    assert out["DoubleWhiteSpaceError: canonical_name"][0] is None


def test_non_utf8_column_is_ignored():
    df = pl.DataFrame({"index": [1]})  # Int64 → should not create any error columns
    out = run_whitespace_check(df)

    # Should produce exactly zero new columns
    assert out.columns == ["index"]
