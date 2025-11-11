#!/usr/bin/env python3
"""
Data Validation Script for Company Name Matcher

Validates CSV data contributions for:
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
    """
    Raised when a CSV file contains headers but has no data rows.
    """

    pass


class ValidationError(Exception):
    """
    Raised when data validation fails for one or more CSV files.
    Includes the offending Polars DataFrame in the exception message.

    Attributes:
        df (pl.DataFrame): The DataFrame containing rows with validation errors.
    """

    def __init__(self, message: str, df: pl.DataFrame):
        """
        Initialize the ValidationError with a message and a DataFrame.
        Configures Polars to display all rows and long strings in full
        for easier debugging.

        Args:
            message (str): The custom error message.
            df (pl.DataFrame): The DataFrame with validation errors.
        """
        pl.Config.set_tbl_rows(df.height)
        pl.Config.set_tbl_width_chars(-1)
        pl.Config.set_fmt_str_lengths(1_000_000_000)
        full_message = f"{message}\n{df}"
        super().__init__(full_message)
        self.df = df


class FileNameError(Exception):
    """
    Raised when a CSV file has an invalid filename according to
    predefined naming conventions.
    """

    pass


class DataValidator:
    """
    Class responsible for validating CSV data contributions for the
    Company Name Matcher repository.

    Attributes:
        repo_root (Path): The root directory of the repository.
        data_dir (Path): Directory containing all CSV data.
        positive_dir (Path): Directory containing positive CSVs.
        negative_dir (Path): Directory containing negative CSVs.
        country_code_file (Path): CSV file containing valid ISO2 country codes.
        country_codes (Set[str]): Set of valid country codes for validation.
        positive_csvs (pl.LazyFrame): Combined lazy frame of all positive CSVs.
        negative_csvs (pl.LazyFrame): Combined lazy frame of all negative CSVs.
    """

    def __init__(self, repo_root: Path) -> None:
        """
        Initialize the DataValidator.

        Args:
            repo_root (Path): Path to the root of the repository.
        """
        self.repo_root = Path(repo_root)
        self.data_dir = self.repo_root / "data"
        self.positive_dir = self.data_dir / "positive"
        self.negative_dir = self.data_dir / "negative"
        self.country_code_file = self.data_dir / "_reference" / "countrycode.csv"

        # Load country codes
        self.country_codes = self._load_country_codes()

        # Scan positve CSVs
        self.positive_csvs = self._scan_positive_csvs()

        # Scan negative CSVs
        self.negative_csvs = self._scan_negative_csvs()

    def _load_country_codes(self) -> Set[str]:
        """
        Load the set of valid country codes from the reference CSV.

        Returns:
            Set[str]: Set of ISO2 country codes.
        """
        country_codes = pl.read_csv(self.country_code_file, columns="ISO2")
        country_codes = set(country_codes["ISO2"].str.strip_chars().to_list())
        return country_codes

    def _scan_positive_csvs(self) -> pl.LazyFrame:
        """
        Scan all positive CSV files into a combined LazyFrame.

        Returns:
            pl.LazyFrame: Combined lazy frame of all positive CSVs.
        """
        # Get paths of CSVs
        csv_paths = glob(str(self.positive_dir / "*.csv"))

        # Scan CSVs into lazyframes
        csv_frames: List[pl.LazyFrame] = []
        for f in csv_paths:
            filename = Path(f).name
            try:
                lf = pl.scan_csv(
                    f,
                    encoding="utf8-lossy",
                    schema={
                        "canonical_name": pl.Utf8,
                        "variation": pl.Utf8,
                        "country_code": pl.Utf8,
                        "source": pl.Utf8,
                    },
                )

                head = lf.head(1).collect()
                if head.height == 0:
                    raise EmptyFileError(f"File {f} is empty (no rows).")

                csv_frames.append(
                    lf.with_row_index(name="Row Number", offset=2).with_columns(
                        pl.lit(f"positive/{str(filename)}").alias("Filename")
                    )
                )

            except pl.exceptions.SchemaError as e:
                raise SystemExit(f" Schema error in {f}: {e}")

        # Combine into one Lazyframe
        positive_csv = pl.concat(csv_frames, how="vertical")
        return positive_csv

    def _scan_negative_csvs(self) -> pl.LazyFrame:
        """
        Scan all negative CSV files into a combined LazyFrame.

        Returns:
            pl.LazyFrame: Combined lazy frame of all negative CSVs.
        """
        # Get paths of CSVs
        csv_paths = glob(str(self.negative_dir / "*.csv"))

        # Scan CSVs into lazyframes
        csv_frames: List[pl.LazyFrame] = []
        for f in csv_paths:
            filename = Path(f).name
            try:
                lf = pl.scan_csv(
                    f,
                    encoding="utf8-lossy",
                    schema={
                        "canonical_name_x": pl.Utf8,
                        "canonical_name_y": pl.Utf8,
                        "country_code_x": pl.Utf8,
                        "country_code_y": pl.Utf8,
                        "remark": pl.Utf8,
                    },
                )

                head = lf.head(1).collect()
                if head.height == 0:
                    raise EmptyFileError(f"File {f} is empty (no rows).")

                csv_frames.append(
                    lf.with_row_index(name="Row Number", offset=2).with_columns(
                        pl.lit(f"negative/{str(filename)}").alias("Filename")
                    )
                )

            except pl.exceptions.SchemaError as e:
                raise SystemExit(f" Schema error in {f}: {e}")

        # Combine into one Lazyframe
        negative_csv = pl.concat(csv_frames, how="vertical")
        return negative_csv

    def _unicode_check(self, lf: pl.LazyFrame) -> list[pl.Expr]:
        """
        Check all string columns in a LazyFrame for invalid UTF-8 characters.

        Args:
            lf (pl.LazyFrame): The LazyFrame to validate.

        Returns:
            list[pl.Expr]: List of expressions flagging invalid UTF-8 characters.
        """
        exprs: List[pl.Expr] = []
        schema = lf.collect_schema()
        for col in schema.names():
            # Only run on Utf8 columns
            if schema[col] == pl.Utf8:
                exprs.append(
                    pl.when(pl.col(col).str.contains("�"))
                    .then(pl.lit(f"UnicodeError: {col} has an invalid UTF-8 character"))
                    .otherwise(None)
                    .alias(f"UnicodeError: {col}")
                )
        return exprs

    def _country_code_check(self, lf: pl.LazyFrame) -> list[pl.Expr]:
        """
        Check that all 'country_code' columns contain valid ISO2 codes.

        Args:
            lf (pl.LazyFrame): LazyFrame to check.

        Returns:
            list[pl.Expr]: Expressions flagging invalid country codes.
        """
        exprs: List[pl.Expr] = []
        for col in lf.collect_schema().names():
            if col.startswith("country_code"):
                exprs.append(
                    pl.when(pl.col(col).is_in(self.country_codes))
                    .then(None)
                    .otherwise(
                        pl.concat_str(
                            [
                                pl.lit(f"CountryCodeError: {col} has invalid value '"),
                                pl.col(col),
                                pl.lit("'"),
                            ]
                        )
                    )
                    .alias(f"CountryCodeError: {col}")
                )
        return exprs

    def _mandatory_col_check(self, lf: pl.LazyFrame) -> list[pl.Expr]:
        """
        Check that mandatory columns are not null or empty.

        Mandatory columns start with:
        - canonical_name
        - variation
        - country_code

        Args:
            lf (pl.LazyFrame): LazyFrame to check.

        Returns:
            list[pl.Expr]: Expressions flagging missing data.
        """
        prefixes = ["canonical_name", "variation", "country_code"]
        exprs: List[pl.Expr] = []
        schema = lf.collect_schema()

        for col in schema.names():
            if any(col.startswith(p) for p in prefixes):
                exprs.append(
                    pl.when(
                        pl.col(col).is_null() | (pl.col(col).str.strip_chars() == "")
                    )
                    .then(pl.lit(f"MissingDataError: {col} is mandatory"))
                    .otherwise(None)
                    .alias(f"MissingDataError: {col}")
                )
        return exprs

    def _duplication_check(self, lf: pl.LazyFrame) -> pl.Expr:
        """
        Flag duplicate rows across columns starting with
        'canonical_name', 'variation', or 'country_code'.

        Args:
            lf (pl.LazyFrame): LazyFrame to check.

        Returns:
            pl.Expr: Expression marking duplicate rows.
        """
        prefixes = ["canonical_name", "variation", "country_code"]
        cols_to_check = [
            col
            for col in lf.collect_schema().names()
            if any(col.startswith(p) for p in prefixes)
        ]

        return (
            pl.when(pl.struct(cols_to_check).is_duplicated())
            .then(pl.lit("DuplicateRowError: duplicate row found"))
            .otherwise(None)
            .alias("DuplicateRowError")
        )

    def _difference_check(self, lf: pl.LazyFrame) -> pl.Expr:
        """
        Check if the first two 'canonical_name' or 'variation' columns
        contain identical values (case-insensitive, trimmed).

        Args:
            lf (pl.LazyFrame): LazyFrame to check.

        Returns:
            pl.Expr: Expression flagging duplicate names within a row.
        """
        prefixes = ["canonical_name", "variation"]
        cols = [
            col
            for col in lf.collect_schema().names()
            if any(col.startswith(p) for p in prefixes)
        ]

        col1, col2 = cols[:2]

        return (
            pl.when(
                pl.col(col1).str.to_lowercase().str.strip_chars()
                == pl.col(col2).str.to_lowercase().str.strip_chars()
            )
            .then(pl.lit(f"DuplicateNameError: {col1} and {col2} have the same name"))
            .alias("DuplicateNameError")
        )

    def _concatenate_errors(self, lf: pl.LazyFrame) -> pl.Expr:
        """
        Concatenate all error columns into a single 'Errors' column.

        Args:
            lf (pl.LazyFrame): LazyFrame containing error columns.

        Returns:
            pl.Expr: Expression generating a single concatenated 'Errors' column.
        """
        error_cols = [c for c in lf.collect_schema().names() if "Error" in c]

        concat_expr = pl.concat_str(
            [pl.col(c) for c in error_cols],
            separator="\n",
            ignore_nulls=True,
        ).alias("Errors")

        return (
            pl.when(concat_expr == "").then(None).otherwise(concat_expr).alias("Errors")
        )

    def _filename_check(self, data_dir: Path) -> None:
        """
        Validate that all CSV filenames in a directory match the pattern '###.csv'.

        Args:
            data_dir (Path): Directory containing CSV files.

        Raises:
            FileNameError: If any CSV file does not match the naming pattern.
        """
        pattern = re.compile(r"^\d{3}\.csv")
        bad_files: List[str] = []
        for f in glob(str(data_dir / "*.csv")):
            filename = Path(f).name
            if not pattern.match(filename):
                bad_files.append(f)

        if bad_files:
            formatted = "\n  - " + "\n  - ".join(sorted(bad_files))
            raise FileNameError(f"Invalid filenames found:\n{formatted}")

    def validate(self) -> None:
        """
        Run full validation on all CSV data contributions.

        Performs:
        - Filename validation
        - Unicode validation
        - Country code validation
        - Mandatory column checks
        - Duplicate row detection
        - Duplicate name detection
        - Aggregation of errors into a single report

        Raises:
            ValidationError: If any validation errors are found.
        """
        # Check filenames
        self._filename_check(self.positive_dir)
        self._filename_check(self.negative_dir)

        positive = self._scan_positive_csvs()
        negative = self._scan_negative_csvs()

        errors: List[pl.LazyFrame] = []
        for lf in [positive, negative]:

            all_checks: List[pl.Expr] = (
                self._unicode_check(lf)
                + self._country_code_check(lf)
                + self._mandatory_col_check(lf)
                + [self._difference_check(lf), self._duplication_check(lf)]
            )

            # Apply all the checks
            lf = lf.with_columns(all_checks)

            # Make the error column
            lf = lf.with_columns(self._concatenate_errors(lf))

            # Select the required columns for the error report
            lf = lf.select("Filename", "Row Number", "Errors")

            errors.append(lf)

        error_report = (
            pl.concat(errors, how="vertical")
            .filter(pl.col("Errors").is_not_null())
            .collect()
        )

        if error_report.height > 0:
            raise ValidationError(
                "The following errors were found. Please see the below table: \n",
                error_report,
            )
        else:
            print("No data errors. Safe to contribute")


if __name__ == "__main__":

    repo_root = Path(__file__).parent.parent.resolve()

    validator = DataValidator(repo_root)
    validator.validate()
