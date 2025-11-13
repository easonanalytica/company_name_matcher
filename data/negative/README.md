# ➖ Contributing Negative Data Examples

Negative examples are pairs of **canonical legal entity names** that represent **unrelated companies** but may produce similar embeddings in pre-trained models.

## 🎯 What are Negative Examples?

Pairs of `canonical_name_x` and `canonical_name_y` representing **unrelated companies**. These help the model distinguish between:

- **Industry Competitors**: "삼성전자주식회사" vs "엘지전자 주식회사" (unrelated competitors)
- **Contextually Similar**: "Apple Inc" vs "Adidas AG" (both consumer brands, frequently co-mentioned)
- **Similar Naming**: "Capital One Financial Corporation" vs "Capital Group Companies Inc" (similar names, unrelated)

⚠️ **IMPORTANT — Do NOT Include**:
- **Same corporate group**: "Apple Inc" vs "苹果（中国）有限公司" (different legal entities, but same group)
- **Parent/subsidiary**: "Alphabet Inc" vs "Google LLC"
- **Sister companies**: "삼성전자주식회사" vs "삼성중공업 주식회사"

These related entities should naturally score in the middle range (0.5-0.8), not as negatives (0.0). Labeling them 0 would incorrectly train the model to push them apart.

**Note**: Negative examples can be derived from positive examples across **unrelated** `canonical_name` entries.

## 📋 Parquet Format

**Filename**: `{country_code_x}_{country_code_y}.parquet` (country codes sorted alphabetically, e.g., `CN_US.parquet`, `US_US.parquet`, `JP_KR.parquet`)

**Columns**:
- `canonical_name_x` *(required)*: Official legal entity name (first company)
- `canonical_name_y` *(required)*: Official legal entity name (second company)
- `country_code_x` *(required)*: Two-letter ISO country code for first company
- `country_code_y` *(required)*: Two-letter ISO country code for second company
- `remark` *(optional)*: Explanation of why these are negative examples

**Rules**:
- One negative pair per row
- Both must be **different legal entities**
- Use canonical legal entity names (same as positive examples)
- Apache Parquet format with zstd compression
- Country codes in filename must be sorted alphabetically (CN_US.parquet, not US_CN.parquet)

## 📊 Examples

**Industry Competitors** (`KR_KR.parquet`):
```csv
canonical_name_x,canonical_name_y,country_code_x,country_code_y,remark
"삼성전자주식회사","엘지전자 주식회사","KR","KR","direct competitors in consumer electronics"
```

**Contextual Similarities** (`DE_US.parquet`):
```csv
canonical_name_x,canonical_name_y,country_code_x,country_code_y,remark
"Adidas AG","Nike Inc","DE","US","athletic footwear competitors"
"Adidas AG","Apple Inc","DE","US","both consumer brands, frequently co-mentioned"
```

**Similar Naming Patterns** (`FR_US.parquet`):
```csv
canonical_name_x,canonical_name_y,country_code_x,country_code_y,remark
"Orange SA","Orange Inc","FR","US","same brand name, completely different companies"
```

**US-US Competitors** (`US_US.parquet`):
```csv
canonical_name_x,canonical_name_y,country_code_x,country_code_y,remark
"Coca-Cola Company","PepsiCo Inc.","US","US","direct competitors in beverage industry"
"Alphabet Inc","Meta Platforms Inc","US","US","both tech giants, frequently co-mentioned"
"Starbucks Corporation","McDonald's Corporation","US","US","both restaurant chains"
"Capital One Financial Corporation","Capital Group Companies Inc","US","US","similar naming pattern, different companies"
```

## 🔍 Deriving from Positive Examples

You can create negative pairs by taking canonical names from **unrelated** positive example entries:

**From positive examples** (across different .parquet files):
```csv
canonical_name,variation,country_code,source
"Apple Inc","Apple","US","SEC EDGAR"
"Microsoft Corporation","Microsoft","US","SEC EDGAR"
"삼성전자주식회사","삼성전자","KR","company website"
"엘지전자 주식회사","LG전자","KR","company website"
```

**Create negative pairs** (only unrelated companies, save to `KR_US.parquet`):
```csv
canonical_name_x,canonical_name_y,country_code_x,country_code_y,remark
"삼성전자주식회사","Apple Inc","KR","US","unrelated companies"
"엘지전자 주식회사","Microsoft Corporation","KR","US","unrelated companies"
```

**Do NOT create** (related companies in same group):
```csv
"Apple Inc","苹果电脑贸易（上海）有限公司","US","CN"  ← Same corporate group
"삼성전자주식회사","삼성중공업 주식회사","KR","KR"  ← Same corporate group
```

## 🤝 How to Contribute

1. Create Parquet file following the format above (use zstd compression)
2. Place in `data/negative/` directory with `{country_code_x}_{country_code_y}.parquet` naming (country codes sorted alphabetically)
3. Submit PR to `dev` branch (see [CONTRIBUTING.md](../../CONTRIBUTING.md))

**Thank you for helping improve company name matching!** 🎯
