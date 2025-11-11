#!/usr/bin/env python3
"""
Data Validation Script for Company Name Matcher

Validates CSV data contributions for:
- Format consistency
- Data integrity
- Duplicate prevention
- Business logic compliance

Usage:
    python scripts/validate_data.py [--file FILE] [--all]

Options:
    --file FILE: Validate a specific CSV file
    --all: Validate all CSV files in data/ directory
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class DataValidator:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.data_dir = self.repo_root / "data"
        self.positive_dir = self.data_dir / "positive"
        self.negative_dir = self.data_dir / "negative"
        self.reference_dir = self.data_dir / "_reference"

        # Load country codes
        self.country_codes = self._load_country_codes()

        # Track all existing data for duplicate checking
        self.existing_positive: Set[Tuple[str, str, str]] = set()
        self.existing_negative: Set[Tuple[str, str, str, str]] = set()

        self._load_existing_data()

    def _load_country_codes(self) -> Set[str]:
        """Load valid country codes from reference file."""
        country_codes = set()
        country_file = self.reference_dir / "countrycode.csv"

        if not country_file.exists():
            print(f"Warning: Country code reference file not found at {country_file}")
            return country_codes

        try:
            with open(country_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "ISO2" in row:
                        country_codes.add(row["ISO2"].strip().upper())
        except Exception as e:
            print(f"Warning: Could not load country codes: {e}")

        return country_codes

    def _load_existing_data(self, exclude_file: Path = None):
        """Load all existing CSV data for duplicate checking, optionally excluding one file."""
        # Load positive data
        for csv_file in glob.glob(str(self.positive_dir / "*.csv")):
            if exclude_file and Path(csv_file) == exclude_file:
                continue
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        canonical = row.get("canonical_name", "").strip()
                        variation = row.get("variation", "").strip()
                        country = row.get("country_code", "").strip().upper()
                        if canonical and variation and country:
                            self.existing_positive.add((canonical, variation, country))
            except Exception as e:
                print(
                    f"Warning: Could not load existing positive data from {csv_file}: {e}"
                )

        # Load negative data
        for csv_file in glob.glob(str(self.negative_dir / "*.csv")):
            if exclude_file and Path(csv_file) == exclude_file:
                continue
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name_x = row.get("canonical_name_x", "").strip()
                        name_y = row.get("canonical_name_y", "").strip()
                        country_x = row.get("country_code_x", "").strip().upper()
                        country_y = row.get("country_code_y", "").strip().upper()
                        if name_x and name_y and country_x and country_y:
                            self.existing_negative.add(
                                (name_x, name_y, country_x, country_y)
                            )
            except Exception as e:
                print(
                    f"Warning: Could not load existing negative data from {csv_file}: {e}"
                )

    def validate_positive_csv(self, file_path: Path) -> List[str]:
        """Validate a positive data CSV file."""
        errors = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row_num = 1  # Start after header

                for row in reader:
                    row_num += 1

                    # Check required columns
                    canonical = row.get("canonical_name", "").strip()
                    variation = row.get("variation", "").strip()
                    country = row.get("country_code", "").strip().upper()
                    source = row.get("source", "").strip()

                    if not canonical:
                        errors.append(f"Row {row_num}: Missing canonical_name")
                        continue
                    if not variation:
                        errors.append(f"Row {row_num}: Missing variation")
                        continue
                    if not country:
                        errors.append(f"Row {row_num}: Missing country_code")
                        continue

                    # Business logic checks
                    if canonical == variation:
                        errors.append(
                            f"Row {row_num}: canonical_name and variation must be different"
                        )

                    if country not in self.country_codes:
                        errors.append(
                            f"Row {row_num}: Invalid country_code '{country}'"
                        )

                    # Check for duplicates (including across existing files)
                    key = (canonical, variation, country)
                    if key in self.existing_positive:
                        errors.append(
                            f"Row {row_num}: Duplicate entry {key} already exists in repository"
                        )

        except UnicodeDecodeError:
            errors.append("File encoding error: File must be UTF-8 encoded")
        except Exception as e:
            errors.append(f"File parsing error: {e}")

        return errors

    def validate_negative_csv(self, file_path: Path) -> List[str]:
        """Validate a negative data CSV file."""
        errors = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row_num = 1  # Start after header

                for row in reader:
                    row_num += 1

                    # Check required columns
                    name_x = row.get("canonical_name_x", "").strip()
                    name_y = row.get("canonical_name_y", "").strip()
                    country_x = row.get("country_code_x", "").strip().upper()
                    country_y = row.get("country_code_y", "").strip().upper()
                    remark = row.get("remark", "").strip()

                    if not name_x:
                        errors.append(f"Row {row_num}: Missing canonical_name_x")
                        continue
                    if not name_y:
                        errors.append(f"Row {row_num}: Missing canonical_name_y")
                        continue
                    if not country_x:
                        errors.append(f"Row {row_num}: Missing country_code_x")
                        continue
                    if not country_y:
                        errors.append(f"Row {row_num}: Missing country_code_y")
                        continue

                    # Business logic checks
                    if name_x == name_y:
                        errors.append(
                            f"Row {row_num}: canonical_name_x and canonical_name_y must be different"
                        )

                    if country_x not in self.country_codes:
                        errors.append(
                            f"Row {row_num}: Invalid country_code_x '{country_x}'"
                        )

                    if country_y not in self.country_codes:
                        errors.append(
                            f"Row {row_num}: Invalid country_code_y '{country_y}'"
                        )

                    # Check for duplicates (including across existing files)
                    key = (name_x, name_y, country_x, country_y)
                    if key in self.existing_negative:
                        errors.append(
                            f"Row {row_num}: Duplicate entry {key} already exists in repository"
                        )

                    # Also check reverse order (pairs should be unique regardless of order)
                    reverse_key = (name_y, name_x, country_y, country_x)
                    if reverse_key in self.existing_negative:
                        errors.append(
                            f"Row {row_num}: Reverse duplicate entry {reverse_key} already exists in repository"
                        )

        except UnicodeDecodeError:
            errors.append("File encoding error: File must be UTF-8 encoded")
        except Exception as e:
            errors.append(f"File parsing error: {e}")

        return errors

    def validate_csv_file(self, file_path: Path) -> bool:
        """Validate a single CSV file and return True if valid."""
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist")
            return False

        # Reload existing data excluding the current file being validated
        self.existing_positive.clear()
        self.existing_negative.clear()
        self._load_existing_data(exclude_file=file_path)

        # Determine if positive or negative based on directory
        if file_path.parent == self.positive_dir:
            errors = self.validate_positive_csv(file_path)
        elif file_path.parent == self.negative_dir:
            errors = self.validate_negative_csv(file_path)
        else:
            print(f"Error: File {file_path} is not in positive/ or negative/ directory")
            return False

        if errors:
            print(f"❌ Validation failed for {file_path}:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print(f"✅ {file_path} passed validation")
            return True

    def validate_all_csv_files(self) -> bool:
        """Validate all CSV files in data/ directory."""
        all_valid = True

        # First pass: validate each file individually without cross-file duplicate checking
        print("🔍 First pass: validating individual file integrity...")

        # Validate positive files
        positive_files = list(self.positive_dir.glob("*.csv"))
        if positive_files:
            print(f"\n🔍 Validating {len(positive_files)} positive data files...")
            for csv_file in positive_files:
                if not self.validate_csv_file_for_integrity(csv_file):
                    all_valid = False

        # Validate negative files
        negative_files = list(self.negative_dir.glob("*.csv"))
        if negative_files:
            print(f"\n🔍 Validating {len(negative_files)} negative data files...")
            for csv_file in negative_files:
                if not self.validate_csv_file_for_integrity(csv_file):
                    all_valid = False

        # Second pass: check for cross-file duplicates
        if all_valid:
            print("\n🔍 Second pass: checking for cross-file duplicates...")
            if not self._check_cross_file_duplicates(positive_files + negative_files):
                all_valid = False

        return all_valid

    def validate_csv_file_for_integrity(self, file_path: Path) -> bool:
        """Validate a CSV file for internal integrity only (no cross-file duplicate checking)."""
        errors = []

        # Clear existing data to avoid cross-file checking
        self.existing_positive.clear()
        self.existing_negative.clear()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row_num = 1  # Start after header
                seen_positive = set()
                seen_negative = set()

                for row in reader:
                    row_num += 1

                    if file_path.parent == self.positive_dir:
                        # Validate positive data structure
                        canonical = row.get("canonical_name", "").strip()
                        variation = row.get("variation", "").strip()
                        country = row.get("country_code", "").strip().upper()

                        if not canonical:
                            errors.append(f"Row {row_num}: Missing canonical_name")
                            continue
                        if not variation:
                            errors.append(f"Row {row_num}: Missing variation")
                            continue
                        if not country:
                            errors.append(f"Row {row_num}: Missing country_code")
                            continue

                        # Business logic
                        if canonical == variation:
                            errors.append(
                                f"Row {row_num}: canonical_name and variation must be different"
                            )

                        if country not in self.country_codes:
                            errors.append(
                                f"Row {row_num}: Invalid country_code '{country}'"
                            )

                        # Check for duplicates within this file
                        key = (canonical, variation, country)
                        if key in seen_positive:
                            errors.append(f"Row {row_num}: Duplicate within file {key}")
                        seen_positive.add(key)

                    elif file_path.parent == self.negative_dir:
                        # Validate negative data structure
                        name_x = row.get("canonical_name_x", "").strip()
                        name_y = row.get("canonical_name_y", "").strip()
                        country_x = row.get("country_code_x", "").strip().upper()
                        country_y = row.get("country_code_y", "").strip().upper()

                        if not name_x:
                            errors.append(f"Row {row_num}: Missing canonical_name_x")
                            continue
                        if not name_y:
                            errors.append(f"Row {row_num}: Missing canonical_name_y")
                            continue
                        if not country_x:
                            errors.append(f"Row {row_num}: Missing country_code_x")
                            continue
                        if not country_y:
                            errors.append(f"Row {row_num}: Missing country_code_y")
                            continue

                        # Business logic
                        if name_x == name_y:
                            errors.append(
                                f"Row {row_num}: canonical_name_x and canonical_name_y must be different"
                            )

                        if country_x not in self.country_codes:
                            errors.append(
                                f"Row {row_num}: Invalid country_code_x '{country_x}'"
                            )

                        if country_y not in self.country_codes:
                            errors.append(
                                f"Row {row_num}: Invalid country_code_y '{country_y}'"
                            )

                        # Check for duplicates within this file
                        key = (name_x, name_y, country_x, country_y)
                        reverse_key = (name_y, name_x, country_y, country_x)
                        if key in seen_negative or reverse_key in seen_negative:
                            errors.append(f"Row {row_num}: Duplicate within file {key}")
                        seen_negative.add(key)

        except UnicodeDecodeError:
            errors.append("File encoding error: File must be UTF-8 encoded")
        except Exception as e:
            errors.append(f"File parsing error: {e}")

        if errors:
            print(f"❌ Validation failed for {file_path}:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print(f"✅ {file_path} passed integrity validation")
            return True

    def _check_cross_file_duplicates(self, all_files: List[Path]) -> bool:
        """Check for duplicates across all files."""
        all_positive_data = set()
        all_negative_data = set()
        no_duplicates = True

        # Collect all data
        for file_path in all_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if file_path.parent == self.positive_dir:
                            canonical = row.get("canonical_name", "").strip()
                            variation = row.get("variation", "").strip()
                            country = row.get("country_code", "").strip().upper()
                            if canonical and variation and country:
                                key = (canonical, variation, country)
                                if key in all_positive_data:
                                    print(
                                        f"❌ Cross-file duplicate found: {key} exists in multiple positive files"
                                    )
                                    no_duplicates = False
                                all_positive_data.add(key)
                        elif file_path.parent == self.negative_dir:
                            name_x = row.get("canonical_name_x", "").strip()
                            name_y = row.get("canonical_name_y", "").strip()
                            country_x = row.get("country_code_x", "").strip().upper()
                            country_y = row.get("country_code_y", "").strip().upper()
                            if name_x and name_y and country_x and country_y:
                                key = (name_x, name_y, country_x, country_y)
                                reverse_key = (name_y, name_x, country_y, country_x)
                                if (
                                    key in all_negative_data
                                    or reverse_key in all_negative_data
                                ):
                                    print(
                                        f"❌ Cross-file duplicate found: {key} exists in multiple negative files"
                                    )
                                    no_duplicates = False
                                all_negative_data.add(key)
            except Exception as e:
                print(
                    f"Warning: Could not read {file_path} for duplicate checking: {e}"
                )

        if no_duplicates:
            print("✅ No cross-file duplicates found")
        return no_duplicates


def main():
    parser = argparse.ArgumentParser(description="Validate CSV data contributions")
    parser.add_argument("--file", type=str, help="Validate a specific CSV file")
    parser.add_argument(
        "--all", action="store_true", help="Validate all CSV files in data/ directory"
    )

    args = parser.parse_args()

    # Find repo root (assume script is in scripts/ subdirectory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    validator = DataValidator(repo_root)

    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = repo_root / file_path
        success = validator.validate_csv_file(file_path)
    elif args.all:
        success = validator.validate_all_csv_files()
    else:
        print("Error: Must specify --file FILE or --all")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
