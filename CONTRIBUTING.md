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
   ```

3. **Verify setup**:
   ```bash
   pytest tests/
   ```

## Reporting Issues

When creating an issue, help us categorize it by **including the category in your issue title** (e.g., `[bug] Memory leak in matcher`, `[docs] Clarify contribution guidelines`).

**Issue categories:**
- **`bug`**: Something isn't working
- **`maintenance`**: Code cleanup, refactoring, or technical debt
- **`enhancement`**: New feature or request
- **`documentation`**: Improvements or additions to documentation
- **`question`**: Further information is requested


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
   - ✅ All tests pass
   - ✅ New features include tests
   - ✅ Documentation updated (if applicable)
   - ✅ Clear description of changes

3. **Review**: Maintainers will review and may request changes

4. **Merge to dev**: Once approved, your PR will be merged into `dev`

5. **Release Process**: When ready for release, maintainers will:
   - Create a PR from `dev` to `main`
   - After merging to `main`, changes are automatically synced back to `dev`
   - This keeps both branches in sync and eliminates manual merge commits

## Data Contributions

We welcome contributions of training data to improve model accuracy, especially for multilingual company name matching! For detailed instructions, see the [Data Contribution Guide](data/README.md).

**Quick summary:**
1. Create a CSV file with columns: `canonical_name,variation,country_code,source` (source is optional but recommended)
2. Name it using the pattern `{country_code}_{index}.csv` (e.g., `US_001.csv`, `KR_002.csv`)
3. Submit via pull request targeting the `dev` branch

Contributing data is a great way to help improve the model without writing code!

---

**Questions?** Check existing [issues](https://github.com/easonanalytica/company_name_matcher/issues) or open a new one.

Thank you for contributing! 🎉

