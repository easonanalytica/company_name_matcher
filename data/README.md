# 🤗 Data Contribution Guide

Help improve company name matching by contributing real-world company name variations and contrastive examples!

## 🎯 Model Objective & Fine-tuning Requirements

The fine-tuned model aims to behave as follows:

- **Perfect positive similarity** (score ≈ 1.0): Same legal entity, different names (e.g., "Apple Inc" ↔ "Apple")
- **Natural positive similarity** (score ≈ 0.5-0.8): Related entities in same corporate group (e.g., "Apple Inc" ↔ "苹果（中国）有限公司")
- **Perfect negative similarity** (score ≈ 0.0): Unrelated companies (e.g., "Apple Inc" ↔ "Microsoft Corporation")

### Fine-tuning Data Needed

To achieve this, we use **contrastive learning** with two types of examples:

1. **Positive Examples**: Pairs of `canonical_name` and `variation` representing the **same legal entity**
2. **Negative Examples**: Pairs of `canonical_name_x` and `canonical_name_y` representing **unrelated companies**

**Critical principles**:
- **Use canonical legal entity names in all examples**: Positives teach surface-form variations around a canonical anchor; negatives compare relationships only between canonical entities, avoiding ambiguity across languages and sources.
- **Do not include related entities as negatives**: Parent/subsidiary or same-group pairs should not be labeled 0 (that would push them apart during training). Let the model naturally place them in the neutral → positive range.

## 📁 Data Structure

```
data/
├── positive/           # Name variations for same company
│   ├── README.md      # Guidelines for positive examples
│   └── *.csv          # Country-specific positive examples
├── negative/           # Contrastive examples (different companies)
│   ├── README.md      # Guidelines for negative examples
│   └── *.csv          # Negative example pairs
└── _reference/         # Supporting data
    └── countrycode.csv # ISO country codes
```

## 🚀 Quick Start

**For Positive Examples**: See [`positive/README.md`](positive/README.md)  
**For Negative Examples**: See [`negative/README.md`](negative/README.md)

**Submit Your Contribution**:
1. Create CSV following the appropriate format
2. Place in correct folder (`positive/` or `negative/`)
3. Submit PR to `dev` branch

---

## Understanding Legal Entities (Canonical Names)

**Each `canonical_name` represents a locally registered legal entity.**

Multinational corporations have separate legal entities in each country:

- **Apple Inc** (US entity, registered in California)
- **苹果电脑贸易（上海）有限公司** (China entity, registered in Shanghai)
- **苹果（中国）有限公司** (China entity, different registration)

**These are distinct legal entities**, even though they're part of the same corporate group.

### Correct Usage Examples

✅ **Positive Examples** (same legal entity):
```csv
canonical_name,variation,country_code,source
"Apple Inc","Apple","US","SEC EDGAR"
"Apple Inc","AAPL","US","stock ticker"
"苹果电脑贸易（上海）有限公司","Apple Computer Trading Shanghai","CN","company registry"
```

✅ **Negative Examples** (unrelated companies):
```csv
canonical_name_x,canonical_name_y,country_code_x,country_code_y,remark
"Apple Inc","Microsoft Corporation","US","US","unrelated tech companies"
"Samsung Electronics Co., Ltd.","LG Electronics Inc.","KR","KR","competitors"
"Apple Inc","Orange SA","US","FR","unrelated companies with similar naming theme"
```

⚠️ **NOT for negatives** (related entities — let model learn natural similarity):
```csv
"Apple Inc","苹果电脑贸易（上海）有限公司","US","CN"  ← Related: same corporate group
"Samsung Electronics Co., Ltd.","Samsung Heavy Industries Co., Ltd.","KR","KR"  ← Related: same corporate group
```

❌ **Incorrect** (mixing legal entities):
```csv
canonical_name,variation,country_code,source
"Apple","Apple Inc","US","SEC EDGAR"  ← Wrong: "Apple" is not a legal entity name
"Apple","苹果电脑贸易（上海）有限公司","CN","company registry"  ← Wrong: mixing different legal entities
```

---

## File Requirements

- UTF-8 encoding
- Valid country codes (check [countrycode.csv](../_reference/countrycode.csv), `ISO2` column)
- No duplicate pairs within the same file
- One pair per row

## Data Sources

Use publicly available data from:
- Company registries (SEC EDGAR, Companies House, etc.)
- Official company websites
- Business directories
- Stock exchanges

---

## Questions?

Open an [issue](https://github.com/easonanalytica/company_name_matcher/issues) with the `[question]` label or check [CONTRIBUTING.md](../CONTRIBUTING.md).

**Thank you for contributing!** 🎯
