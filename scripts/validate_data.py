#!/usr/bin/env python3
"""
Data Validation Script for Company Name Matcher

Validates parquet data contributions for:
- Format consistency
- Data integrity
- Duplicate prevention
- Business logic compliance

Usage:
    python scripts/validate_data.py
"""

from glob import glob
from pathlib import Path
from typing import Set, List
import polars as pl
import re


class EmptyFileError(Exception):
    """Raised when a parquet file contains headers but has no data rows."""

    pass


class ValidationError(Exception):
    """
    Raised when data validation fails for one or more parquet files.
    Includes the offending Polars DataFrame in the exception message.
    """

    def __init__(self, message: str, df: pl.DataFrame):
        pl.Config.set_tbl_rows(df.height)
        pl.Config.set_tbl_width_chars(-1)
        pl.Config.set_fmt_str_lengths(1_000_000_000)
        full_message = f"{message}\n{df}"
        super().__init__(full_message)
        self.df = df


class FileNameError(Exception):
    """Raised when a parquet file has an invalid filename according to predefined naming conventions."""

    pass


class DataValidator:
    """
    Class responsible for validating parquet data contributions for the
    Company Name Matcher repository.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self.data_dir = self.repo_root / "data"
        self.positive_dir = self.data_dir / "positive"
        self.negative_dir = self.data_dir / "negative"
        self.country_code_file = self.data_dir / "_reference" / "countrycode.csv"

        # Load country codes
        self.country_codes = self._load_country_codes()

        # Scan parquets
        self.positive_parquets = self._scan_positive_parquets()
        self.negative_parquets = self._scan_negative_parquets()

    def _load_country_codes(self) -> Set[str]:
        country_codes = pl.read_csv(self.country_code_file, columns=["ISO2"])
        return set(country_codes["ISO2"].str.strip_chars().to_list())

    def _country_code_validity_check(self, lf: pl.LazyFrame) -> List[pl.Expr]:
        exprs: List[pl.Expr] = []
        for col in lf.schema:
            if col.startswith("country_code"):
                exprs.append(
                    pl.when(pl.col(col).is_in(self.country_codes))
                    .then(None)
                    .otherwise(
                        pl.concat_str(
                            [pl.lit(f"CountryCodeError: {col} has invalid value '"), pl.col(col), pl.lit("'")]
                        )
                    )
                    .alias(f"CountryCodeError: {col}")
                )
        return exprs

    def _country_code_filename_match_check(
        self, lf: pl.LazyFrame, filename: str, negative: bool = False
    ) -> List[pl.Expr]:
        exprs: List[pl.Expr] = []
        base_name = Path(filename).stem

        if not negative:
            exprs.append(
                pl.when(pl.col("country_code") == base_name)
                .then(None)
                .otherwise(pl.lit(f"CountryCodeError: country_code != filename ({base_name})"))
                .alias("CountryCodeError: filename_mismatch")
            )
        else:
            expected = sorted(base_name.split("_"))
            exprs.append(
                pl.when(pl.col("country_code_x") == expected[0])
                .then(None)
                .otherwise(pl.lit(f"CountryCodeError: country_code_x != filename part ({expected[0]})"))
                .alias("CountryCodeError: filename_mismatch_x")
            )
            exprs.append(
                pl.when(pl.col("country_code_y") == expected[1])
                .then(None)
                .otherwise(pl.lit(f"CountryCodeError: country_code_y != filename part ({expected[1]})"))
                .alias("CountryCodeError: filename_mismatch_y")
            )

        return exprs

    def _scan_positive_parquets(self) -> pl.LazyFrame:
        frames: List[pl.LazyFrame] = []
        for f in glob(str(self.positive_dir / "*.parquet")):
            filename = Path(f).name
            lf = pl.scan_parquet(
                f,
                schema={
                    "canonical_name": pl.Utf8,
                    "variation": pl.Utf8,
                    "country_code": pl.Utf8,
                    "source": pl.Utf8,
                },
            )
            if lf.head(1).collect().height == 0:
                raise EmptyFileError(f"File {f} is empty (no rows).")

            lf = (
                lf.with_row_index(name="Row Number", offset=2)
                .with_columns(pl.lit(f"positive/{filename}").alias("Filename"))
                .with_columns(
                    self._country_code_validity_check(lf) + self._country_code_filename_match_check(lf, filename)
                )
            )
            frames.append(lf)

        return pl.concat(frames, how="vertical")

    def _scan_negative_parquets(self) -> pl.LazyFrame:
        frames: List[pl.LazyFrame] = []
        for f in glob(str(self.negative_dir / "*.parquet")):
            filename = Path(f).name
            lf = pl.scan_parquet(
                f,
                schema={
                    "canonical_name_x": pl.Utf8,
                    "canonical_name_y": pl.Utf8,
                    "country_code_x": pl.Utf8,
                    "country_code_y": pl.Utf8,
                    "remark": pl.Utf8,
                },
            )
            if lf.head(1).collect().height == 0:
                raise EmptyFileError(f"File {f} is empty (no rows).")

            lf = (
                lf.with_row_index(name="Row Number", offset=2)
                .with_columns(pl.lit(f"negative/{filename}").alias("Filename"))
                .with_columns(
                    self._country_code_validity_check(lf)
                    + self._country_code_filename_match_check(lf, filename, negative=True)
                )
            )
            frames.append(lf)

        return pl.concat(frames, how="vertical")

    def _mandatory_col_check(self, lf: pl.LazyFrame) -> List[pl.Expr]:
        prefixes = ["canonical_name", "variation", "country_code"]
        exprs: List[pl.Expr] = []
        for col in lf.schema:
            if any(col.startswith(p) for p in prefixes):
                exprs.append(
                    pl.when(pl.col(col).is_null() | (pl.col(col).str.strip_chars() == ""))
                    .then(pl.lit(f"MissingDataError: {col} is mandatory"))
                    .otherwise(None)
                    .alias(f"MissingDataError: {col}")
                )
        return exprs

    def _duplication_check(self, lf: pl.LazyFrame) -> pl.Expr:
        prefixes = ["canonical_name", "variation", "country_code"]
        cols_to_check = [col for col in lf.schema if any(col.startswith(p) for p in prefixes)]
        return (
            pl.when(pl.struct(cols_to_check).is_duplicated())
            .then(pl.lit("DuplicateRowError: duplicate row found"))
            .otherwise(None)
            .alias("DuplicateRowError")
        )

    def _difference_check(self, lf: pl.LazyFrame) -> pl.Expr:
        prefixes = ["canonical_name", "variation"]
        cols = [col for col in lf.schema if any(col.startswith(p) for p in prefixes)]
        col1, col2 = cols[:2]
        return (
            pl.when(
                pl.col(col1).str.to_lowercase().str.strip_chars() == pl.col(col2).str.to_lowercase().str.strip_chars()
            )
            .then(pl.lit(f"DuplicateNameError: {col1} and {col2} have the same name"))
            .alias("DuplicateNameError")
        )

    def _concatenate_errors(self, lf: pl.LazyFrame) -> pl.Expr:
        error_cols = [c for c in lf.schema if "Error" in c]
        concat_expr = pl.concat_str([pl.col(c) for c in error_cols], separator="\n", ignore_nulls=True)
        return pl.when(concat_expr == "").then(None).otherwise(concat_expr).alias("Errors")

    def _filename_check(self, data_dir: Path, negative: bool = False) -> None:
        pattern = re.compile(r"^[A-Z]{2}(_[A-Z]{2})?\.parquet$") if negative else re.compile(r"^[A-Z]{2}\.parquet$")
        bad_files: List[str] = []
        for f in glob(str(data_dir / "*.parquet")):
            filename = Path(f).name
            if not pattern.match(filename):
                bad_files.append(f)
        if bad_files:
            formatted = "\n  - " + "\n  - ".join(sorted(bad_files))
            raise FileNameError(f"Invalid filenames found:\n{formatted}")

    def validate(self) -> None:
        # Check filenames
        self._filename_check(self.positive_dir)
        self._filename_check(self.negative_dir, negative=True)

        positive = self.positive_parquets
        negative = self.negative_parquets

        errors: List[pl.LazyFrame] = []
        for lf in [positive, negative]:
            all_checks: List[pl.Expr] = self._mandatory_col_check(lf) + [
                self._difference_check(lf),
                self._duplication_check(lf),
            ]

            lf = lf.with_columns(all_checks)
            lf = lf.with_columns(self._concatenate_errors(lf))
            lf = lf.select("Filename", "Row Number", "Errors")
            errors.append(lf)

        error_report = pl.concat(errors, how="vertical").filter(pl.col("Errors").is_not_null()).collect()

        if error_report.height > 0:
            raise ValidationError("The following errors were found. Please see the below table: \n", error_report)
        else:
            print("No data errors. Safe to contribute")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent.resolve()
    validator = DataValidator(repo_root)
    validator.validate()
