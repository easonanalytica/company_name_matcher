# 🤗 Data Contribution Guide

Contribute company name datasets to improve the model's accuracy, especially for multilingual matching!

## Why Company Name Matching Matters

**Reconciliation is a universal task** in data-driven industries. Company names appear inconsistently across different sources - "Apple Inc", "Apple Incorporated", "苹果公司" might all refer to the same entity.

**Key Use Cases:**
- **📊 Big Data Analytics**: Unify company mentions across news, social media, and financial reports
- **💼 Portfolio Analysis**: Match investment holdings against multiple data sources
- **🔗 Supply Chain Analytics**: Track vendors and suppliers across procurement systems
- **📈 Market Intelligence**: Aggregate company data from diverse sources for comprehensive analysis
- **🏦 Financial Reporting**: Ensure consistent entity identification across regulatory filings

Your contributions directly improve these critical business processes by helping the model learn more variations and handle edge cases.

## Quick Start

1. **Create a CSV file** named `{country_code}_{index}.csv` (e.g., `US_001.csv`, `KR_001.csv`)
2. **Add your data** following the format below
3. **Place the file** in the `data/` directory
4. **Submit via pull request** to the `dev` branch

## Data Format

```csv
canonical_name,variation,country_code,source
"Apple","Apple Inc","US","SEC EDGAR"
"Apple","Apple Incorporated","US","SEC EDGAR"
"Samsung","삼성전자","KR","company website"
"Samsung","Samsung Electronics","KR","company website"
```

**Columns:**
- `canonical_name` *(required)*: Clean, accurate primary name for grouping
- `variation` *(required)*: Any variation, abbreviation, or alternative spelling (can include typos/informal names)
- `country_code` *(required)*: Two-letter code from `_reference/countrycode.csv` (ISO2 column)
- `source` *(optional)*: Where the data came from (e.g., "SEC EDGAR", "company website")

**Rules:**
- One variation per row
- Each `canonical_name` + `country_code` combo identifies a unique company
- Include local language variations for multilingual training

## Quality Guidelines

**Canonical names:**
- Must be clean, accurate, and correctly spelled
- Should represent real companies
- Use the most official/common form

**Variations:**
- Include abbreviations, legal forms, common misspellings, informal names
- Can include messy/imperfect data - that's okay!
- Add local language variations when possible

**Filename:** `{country_code}_{index}.csv` (e.g., `US_001.csv`, `KR_002.csv`)

**Before submitting:**
- No duplicate canonical names within your file
- Valid country codes (check `_reference/countrycode.csv`)
- UTF-8 encoding

## Examples

**US Companies (`US_001.csv`):**
```csv
canonical_name,variation,country_code,source
"Apple","Apple Inc","US","SEC EDGAR"
"Apple","Apple Incorporated","US","SEC EDGAR"
"Alphabet","Alphabet Inc","US","SEC EDGAR"
"Alphabet","Google","US","common usage"
"Amazon","Amazon.com Inc","US","SEC EDGAR"
"Amazon","AMZN","US","stock ticker"
```

**Korean Companies with Local Names (`KR_001.csv`):**
```csv
canonical_name,variation,country_code,source
"Samsung","삼성전자","KR","company website"
"Samsung","Samsung Electronics","KR","company website"
"Hyundai","현대자동차","KR","company website"
"Hyundai","Hyundai Motor Company","KR","company website"
```

## Suggested Data Sources

Use publicly available data: company registries (SEC EDGAR, Companies House), official websites, business directories, or academic datasets.

## Questions?

Check [existing issues](https://github.com/easonanalytica/company_name_matcher/issues) or open a new one with the `question` label. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full contribution guide.

**Thank you for helping improve Company Name Matcher!** 🎯
