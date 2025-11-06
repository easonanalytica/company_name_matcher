# 🤗 Contributing to Company Name Matcher

Thank you for your interest in contributing to Company Name Matcher! This document outlines the process for contributing to the project.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Reporting Issues](#reporting-issues)
- [Code Contributions](#code-contributions)
  - [Development Workflow](#development-workflow)
  - [Testing](#testing)
  - [Pull Request Process](#pull-request-process)
- [Data Contributions](#data-contributions)

## Getting Started

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/company_name_matcher.git
   cd company_name_matcher
   git remote add upstream https://github.com/easonanalytica/company_name_matcher.git
   ```

2. **Set up your environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   # Optional: Install test dependencies if you want to run tests
   pip install -e ".[test]"
   ```

3. **Verify setup**:
   ```bash
   pytest tests/
   ```

## Reporting Issues

**Include the category in your issue title** (e.g., `[bug] Memory leak in matcher`, `[docs] Clarify contribution guidelines`).

This helps you categorize the issue type before submitting and helps maintainers prioritize and label issues during triage.

**Issue categories:**
- **`bug`**: Something isn't working
- **`data`**: Data-related issues or contributions
- **`maintenance`**: Code cleanup, refactoring, or technical debt
- **`enhancement`**: New feature or request
- **`documentation`**: Improvements or additions to documentation
- **`question`**: Further information is requested

*Maintainers will apply final labels when triaging issues.*


## Code Contributions

### Development Workflow

**Important**: All pull requests must target the `dev` branch, not `main`.

1. **Create a feature branch** from `dev`:
   ```bash
   git checkout dev
   git pull upstream dev
   git checkout -b <prefix>/your-branch-name
   ```

   **Recommended branch naming conventions** (based on issue labels):
   - `bug` → `fix/` (e.g., `fix/memory-leak`)
   - `data` → `data/` (e.g., `data/add-us-company-names`)
   - `enhancement` → `feature/` (e.g., `feature/add-caching`)
   - `documentation` → `docs/` (e.g., `docs/update-readme`)
   - `maintenance` → `chore/` (e.g., `chore/clean-api`)

2. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Mark as ready for review** when your changes are complete:
   - Click "Ready for review" on your GitHub PR
   - **Automated testing** will be triggered when a maintainer adds the `under-review` label (to conserve CI resources)

### Testing

Run tests before submitting:
```bash
pytest tests/
```

**Requirements**:
- All new features must include tests
- Place tests in `tests/` directory as `test_*.py`

### Pull Request Process

1. **Open a pull request** against the `dev` branch (not `main`)

2. **PR Checklist**:
   - ✅ All tests pass locally
   - ✅ **Automated testing passes**:
     - Ready PRs → `dev`: Core tests run on multiple Python versions (requires `under-review` label from maintainer to trigger)
     - `dev` → `main`: Full test suite + linting + security + build checks
   - ✅ **Data validation passes** (if contributing CSV data files)
   - ✅ New features include tests
   - ✅ Documentation updated (if applicable)
   - ✅ Clear description of changes

3. **Review**: Maintainers will review and may request changes

4. **Merge to dev**: Once approved, your PR will be merged into `dev`

5. **Release Process**: When ready for release, maintainers will:
   - Create a PR from `dev` to `main`
   - **Automated testing** will run on multiple Python versions (3.9-3.12)
   - **Automated data validation** will check all CSV files for duplicates and integrity
   - After merging to `main`, changes are automatically synced back to `dev`
   - This keeps both branches in sync and eliminates manual merge commits

## Data Contributions

We welcome contributions of training data to improve model accuracy! We accept both positive examples (name variations for same company) and negative examples (contrastive pairs for different companies).

**Data Types:**
- **Positive Examples**: Name variations representing the same legal entity
- **Negative Examples**: Pairs that may seem similar but are different companies

For detailed instructions, see the [Data Contribution Guide](data/README.md).

**Quick summary:**
1. Choose type: `data/positive/` (same company) or `data/negative/` (different companies)
2. Create CSV following the format in the respective README
3. **Validate locally**: Run `python scripts/validate_data.py --file your_file.csv`
4. Submit via pull request targeting the `dev` branch

**Data Validation:**
Before submitting, please validate your CSV files:
```bash
# Validate a single file
python scripts/validate_data.py --file data/positive/your_file.csv

# Validate all data files
python scripts/validate_data.py --all
```

The validation checks for:
- Correct CSV format and UTF-8 encoding
- Required columns and valid data
- Duplicate entries (within file and across repository)
- Business logic compliance
- Valid country codes

Contributing data is a great way to help improve the model without writing code!

---

**Questions?** Check existing [issues](https://github.com/easonanalytica/company_name_matcher/issues) or open a new one.

Thank you for contributing! 🎉

