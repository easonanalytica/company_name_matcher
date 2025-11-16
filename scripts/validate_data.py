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
from polars.type_aliases import SchemaDict
import re


class EmptyFileError(Exception):
    """Raised when a parquet file contains headers but has no data rows."""

    pass


class ValidationError(Exception):
    """
    Raised when data validation fails for one or more parquet files.
    Includes the offending Polars DataFrame in the exception message.

    Attributes:
        df (pl.DataFrame): The DataFrame that caused the validation error.
    """

    def __init__(self, message: str, df: pl.DataFrame):
        """
        Initialize ValidationError with a message and the offending DataFrame.

        Args:
            message (str): Description of the validation error.
            df (pl.DataFrame): The DataFrame that failed validation.
        """
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

    Validations performed include:
    - Mandatory column presence
    - Country code validity
    - Filename and country code consistency
    - Duplicate detection
    - Content differences between canonical names and variations
    """

    def __init__(self, repo_root: Path) -> None:
        """
        Initialize the DataValidator.

        Args:
            repo_root (Path): Root path of the repository containing the data directories.
        """
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
        """
        Load the list of valid ISO2 country codes from a reference CSV file.

        Returns:
            Set[str]: Set of valid country codes.
        """
        country_codes = pl.read_csv(self.country_code_file, columns=["ISO2"])
        return set(country_codes["ISO2"].str.strip_chars().to_list())

    def _country_code_validity_check(self, lf: pl.LazyFrame) -> List[pl.Expr]:
        """
        Generate Polars expressions to check if country codes in the DataFrame are valid.

        Args:
            lf (pl.LazyFrame): The lazy DataFrame to validate.

        Returns:
            List[pl.Expr]: List of expressions producing error messages for invalid codes.
        """
        exprs: List[pl.Expr] = []
        schema = lf.collect_schema()
        for col in schema:
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
        """
        Generate expressions to check if country codes match the filename.

        Args:
            lf (pl.LazyFrame): Lazy DataFrame to validate.
            filename (str): Name of the parquet file being checked.
            negative (bool, optional): Whether the file is negative samples. Defaults to False.

        Returns:
            List[pl.Expr]: List of expressions producing filename mismatch errors.
        """
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
        """
        Scan all positive parquet files, validate structure, and attach initial error columns.

        Returns:
            pl.LazyFrame: Concatenated lazy frame of all positive parquet files.
        """
        frames: List[pl.LazyFrame] = []
        schema: SchemaDict={
                    "canonical_name": pl.Utf8,
                    "variation": pl.Utf8,
                    "country_code": pl.Utf8,
                    "source": pl.Utf8,
                }
        for f in glob(str(self.positive_dir / "*.parquet")):
            filename = Path(f).name
            lf = pl.scan_parquet(
                f,
                schema=schema,
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
        """
        Scan all negative parquet files, validate structure, and attach initial error columns.

        Returns:
            pl.LazyFrame: Concatenated lazy frame of all negative parquet files.
        """
        frames: List[pl.LazyFrame] = []
        schema: SchemaDict = {
            "canonical_name_x": pl.Utf8,
            "canonical_name_y": pl.Utf8,
            "country_code_x": pl.Utf8,
            "country_code_y": pl.Utf8,
            "remark": pl.Utf8,
        }
        for f in glob(str(self.negative_dir / "*.parquet")):
            filename = Path(f).name
            lf = pl.scan_parquet(
                f,
                schema=schema,
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
        """
        Create expressions to check mandatory columns for missing or empty values.

        Args:
            lf (pl.LazyFrame): Lazy frame to validate.

        Returns:
            List[pl.Expr]: List of error expressions for missing data.
        """
        prefixes = ["canonical_name", "variation", "country_code"]
        exprs: List[pl.Expr] = []
        schema = lf.collect_schema()
        for col in schema:
            if any(col.startswith(p) for p in prefixes):
                exprs.append(
                    pl.when(pl.col(col).is_null() | (pl.col(col).str.strip_chars() == ""))
                    .then(pl.lit(f"MissingDataError: {col} is mandatory"))
                    .otherwise(None)
                    .alias(f"MissingDataError: {col}")
                )
        return exprs

    def _duplication_check(self, lf: pl.LazyFrame) -> pl.Expr:
        """
        Create an expression to identify duplicate rows based on key columns.

        Args:
            lf (pl.LazyFrame): Lazy frame to check for duplicates.

        Returns:
            pl.Expr: Error expression for duplicate rows.
        """
        schema = lf.collect_schema()
        prefixes = ["canonical_name", "variation", "country_code"]
        cols_to_check = [col for col in schema if any(col.startswith(p) for p in prefixes)]
        return (
            pl.when(pl.struct(cols_to_check).is_duplicated())
            .then(pl.lit("DuplicateRowError: duplicate row found"))
            .otherwise(None)
            .alias("DuplicateRowError")
        )

    def _difference_check(self, lf: pl.LazyFrame) -> pl.Expr:
        """
        Create an expression to detect identical canonical and variation names.

        Args:
            lf (pl.LazyFrame): Lazy frame to check.

        Returns:
            pl.Expr: Error expression for duplicate names.
        """
        schema = lf.collect_schema()
        prefixes = ["canonical_name", "variation"]
        cols = [col for col in schema if any(col.startswith(p) for p in prefixes)]
        col1, col2 = cols[:2]
        return (
            pl.when(
                pl.col(col1).str.to_lowercase().str.strip_chars() == pl.col(col2).str.to_lowercase().str.strip_chars()
            )
            .then(pl.lit(f"DuplicateNameError: {col1} and {col2} have the same name"))
            .alias("DuplicateNameError")
        )

    def _concatenate_errors(self, lf: pl.LazyFrame) -> pl.Expr:
        """
        Combine all individual error columns into a single 'Errors' column.

        Args:
            lf (pl.LazyFrame): Lazy frame containing error columns.

        Returns:
            pl.Expr: Expression concatenating all errors.
        """
        schema = lf.collect_schema()
        error_cols = [c for c in schema if "Error" in c]
        concat_expr = pl.concat_str([pl.col(c) for c in error_cols], separator="\n", ignore_nulls=True)
        return pl.when(concat_expr == "").then(None).otherwise(concat_expr).alias("Errors")

    def _filename_check(self, data_dir: Path, negative: bool = False) -> None:
        """
        Validate filenames against expected patterns for positive and negative datasets.

        Args:
            data_dir (Path): Directory containing parquet files.
            negative (bool, optional): Flag for negative dataset filename pattern. Defaults to False.

        Raises:
            FileNameError: If any files do not match the expected naming conventions.
        """
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
        """
        Execute all validation checks on the dataset.

        Checks performed:
        - File naming conventions
        - Mandatory columns
        - Country code validity
        - Filename-country code consistency
        - Duplicate rows
        - Canonical and variation name uniqueness

        Raises:
            ValidationError: If any data validation errors are found.
        """
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
