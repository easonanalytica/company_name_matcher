# ➕ Contributing Positive Data Examples

Positive examples are pairs of names representing the **same legal entity** but with different surface forms.

## 🎯 What are Positive Examples?

Pairs of `canonical_name` (official registered legal entity name) and `variation` (alternative name) that refer to the same company. Common variations include:

- **Abbreviations & Acronyms**: "International Business Machines Corporation" → "IBM"
- **Legal Entity Suffixes**: "Apple Inc." → "Apple Incorporated", "Apple Corporation"
- **Language Translations**: "Apple Inc." → "苹果公司", "삼성전자주식회사" → "Samsung Electronics"
- **Common Misspellings**: "Microsoft Corporation" → "Microsft Corporation"
- **Punctuation Differences**: "McDonald's Corporation" → "McDonalds", "Nestlé S.A." → "Nestle SA"
- **Stock Tickers**: "Apple Inc." → "AAPL", "Alphabet Inc." → "GOOGL"
- **Historical Names**: "Apple Inc." → "Apple Computer Inc."

## 📋 Parquet Format

**Filename**: `{country_code}.parquet` (e.g., `US.parquet`, `CN.parquet`)

**Columns**:
- `canonical_name` *(required)*: Official registered legal entity name
- `variation` *(required)*: Alternative name/abbreviation/variation
- `country_code` *(required)*: Two-letter ISO country code (see [countrycode.csv](../_reference/countrycode.csv))
- `source` *(optional)*: Data source

**Rules**:
- One variation per row
- `canonical_name` ≠ `variation` (must be different)
- Both represent the **same legal entity**
- Apache Parquet format with zstd compression
- Not included: related-but-different entities (parent/subsidiary, sister companies)
- No trailing/leading and double whitespaces/tabs/newlines

## 📊 Examples

**US Companies** ([`US.parquet`](US.parquet)):
|canonical_name|variation|country_code|source|
|--------------|---------|------------|------|
|Apple Inc.|Apple|US|SEC EDGAR|
|Apple Inc.|Apple Incorporated|US|legal variation|
|Apple Inc.|AAPL|US|stock ticker|
|International Business Machines Corporation|IBM|US|abbreviation|
|Alphabet Inc.|Google|US|common usage|


**Chinese Companies** ([`CN.parquet`](CN.parquet)):

|canonical_name|variation|country_code|source|
|--------------|---------|------------|------|
|苹果电脑贸易（上海）有限公司|Apple Computer Trading Shanghai|CN|english translation|
|苹果电脑贸易（上海）有限公司|苹果上海|CN|common usage|
|阿里巴巴集团控股有限公司|Alibaba Group|CN|common usage|
|阿里巴巴集团控股有限公司|BABA|CN|stock ticker|

**Korean Companies** ([`KR.parquet`](KR.parquet)):

|canonical_name|variation|country_code|source|
|--------------|---------|------------|------|
|삼성전자주식회사|Samsung Electronics|KR|english translation|
|삼성전자주식회사|삼성전자|KR|common usage|
|현대자동차주식회사|Hyundai Motor Company|KR|english translation|

## 🤝 How to Contribute

1. Create a parquet file following the format above
2. Place in `data/positive/` directory
3. Validate by running:
    ```bash
    python scripts/validate_data.py
    ```
    and all validations should pass
4. Submit PR to `dev` branch (see [CONTRIBUTING.md](../../CONTRIBUTING.md))

**Thank you for helping improve company name matching!** 🎯
