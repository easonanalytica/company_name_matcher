# 🤗 Data Contribution Guide

Help improve company name matching by contributing datasets with real-world company name variations!

## Why This Matters

Company names appear inconsistently across data sources. "Apple Inc", "Apple Incorporated", and "苹果公司" might all refer to the same entity. This creates challenges for:

- **Big Data Analytics** – unifying company mentions across sources
- **Portfolio Analysis** – matching holdings against market data
- **Supply Chain** – tracking vendors across systems
- **Market Intelligence** – aggregating data from diverse sources

Your contributions help the model learn these variations and improve matching accuracy.

---

## How to Contribute

### 1. Create Your CSV File

**Filename format:** `{country_code}_{index}.csv`

Examples: `US_001.csv`, `CN_001.csv`, `KR_001.csv`

### 2. Add Your Data

**Format:**
```csv
canonical_name,variation,country_code,source
"Apple Inc","Apple","US","SEC EDGAR"
"Apple Inc","Apple Incorporated","US","company website"
"苹果电脑贸易（上海）有限公司","Apple Computer Trading (Shanghai)","CN","company registry"
"苹果电脑贸易（上海）有限公司","苹果上海","CN","common usage"
```

**Columns:**
- **`canonical_name`** *(required)* – The official registered legal entity name
- **`variation`** *(required)* – Any alternative name, abbreviation, or variation
- **`country_code`** *(required)* – Two-letter ISO code (see `_reference/countrycode.csv`, ISO2 column)
- **`source`** *(optional)* – Where the data came from (e.g., "SEC EDGAR", "company website")

### 3. Place File in `data/` Directory

Put your CSV file directly in the `data/` folder.

### 4. Submit Pull Request

Submit your PR to the `dev` branch. See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

---

## Important: Understanding Legal Entities

**Each `canonical_name` represents a locally registered legal entity.**

For multinationals like Apple:
- **US entity**: "Apple Inc" (registered in California)
- **China entities**: "苹果电脑贸易（上海）有限公司", "苹果（中国）有限公司" (separate legal entities in China)

**These are different companies** with different registrations, even though they're part of the same corporate group.

### What This Means for Contributors

✅ **Correct:**
```csv
canonical_name,variation,country_code,source
"Apple Inc","Apple","US","SEC EDGAR"
"Apple Inc","AAPL","US","stock ticker"
"苹果电脑贸易（上海）有限公司","Apple Computer Trading Shanghai","CN","company registry"
"苹果电脑贸易（上海）有限公司","苹果上海","CN","common usage"
"苹果（中国）有限公司","Apple China Limited","CN","company registry"
```

❌ **Incorrect:**
```csv
canonical_name,variation,country_code,source
"Apple","Apple Inc","US","SEC EDGAR"
"Apple","苹果电脑贸易（上海）有限公司","CN","company registry"  ← Wrong: mixing different legal entities
```

---

## Quality Requirements

### Canonical Names
- ✅ Try to use the **official registered legal name** in the local language
- ✅ Should be accurate and correctly spelled
- ✅ Represents a real legal entity

### Variations
- ✅ Include abbreviations, common names, translations, and alternative spellings
- ✅ Can include informal names or common misspellings – that's valuable data!
- ✅ Add local language variations when possible

### File Requirements
- One variation per row
- `canonical_name` and `variation` must be different (no identical pairs)
- No duplicate canonical names within the same file
- Valid country codes (check `_reference/countrycode.csv`)
- UTF-8 encoding

---

## Examples

### US Companies (`US_001.csv`)
```csv
canonical_name,variation,country_code,source
"Apple Inc","Apple","US","SEC EDGAR"
"Apple Inc","Apple Incorporated","US","SEC EDGAR"
"Apple Inc","AAPL","US","stock ticker"
"Alphabet Inc","Google","US","common usage"
"Alphabet Inc","Alphabet","US","SEC EDGAR"
"Amazon.com Inc","Amazon","US","company website"
"Amazon.com Inc","AMZN","US","stock ticker"
```

### Chinese Companies (`CN_001.csv`)
```csv
canonical_name,variation,country_code,source
"苹果电脑贸易（上海）有限公司","Apple Computer Trading Shanghai","CN","company registry"
"苹果电脑贸易（上海）有限公司","苹果上海","CN","common usage"
"阿里巴巴集团控股有限公司","Alibaba Group","CN","company website"
"阿里巴巴集团控股有限公司","阿里巴巴","CN","common usage"
"阿里巴巴集团控股有限公司","BABA","CN","stock ticker"
```

### Korean Companies (`KR_001.csv`)
```csv
canonical_name,variation,country_code,source
"삼성전자주식회사","Samsung Electronics","KR","company website"
"삼성전자주식회사","삼성전자","KR","common usage"
"현대자동차주식회사","Hyundai Motor Company","KR","company website"
"현대자동차주식회사","현대자동차","KR","common usage"
```

---

## Data Sources

Use publicly available data from:
- Company registries (SEC EDGAR, Companies House, etc.)
- Official company websites
- Business directories
- Stock exchanges
- Academic datasets

---

## Questions?

Open an [issue](https://github.com/easonanalytica/company_name_matcher/issues) with the `[question]` label or check [CONTRIBUTING.md](../CONTRIBUTING.md).

**Thank you for contributing!** 🎯
