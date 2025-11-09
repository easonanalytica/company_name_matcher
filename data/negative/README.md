# ➖ Contributing Negative Data Examples

Negative examples are pairs of **canonical legal entity names** that represent **unrelated companies** but may produce similar embeddings in pre-trained models.

## 🎯 What are Negative Examples?

Pairs of `canonical_name_x` and `canonical_name_y` representing **unrelated companies**. These help the model distinguish between:

- **Industry Competitors**: "Samsung Electronics Co., Ltd." vs "LG Electronics Inc." (unrelated competitors)
- **Contextually Similar**: "Apple Inc" vs "Adidas AG" (both consumer brands, frequently co-mentioned)
- **Similar Naming**: "Capital One Financial Corporation" vs "Capital Group Companies Inc" (similar names, unrelated)

⚠️ **IMPORTANT — Do NOT Include**:
- **Same corporate group**: "Apple Inc" vs "苹果（中国）有限公司" (different legal entities, but same group)
- **Parent/subsidiary**: "Alphabet Inc" vs "Google LLC"
- **Sister companies**: "Samsung Electronics" vs "Samsung Heavy Industries"

These related entities should naturally score in the middle range (0.5-0.8), not as negatives (0.0). Labeling them 0 would incorrectly train the model to push them apart.

**Note**: Negative examples can be derived from positive examples across **unrelated** `canonical_name` entries.

## 📋 CSV Format

**Filename**: `{index}.csv` (e.g., `001.csv`, `002.csv`)

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
- UTF-8 encoding

## 📊 Examples

**Industry Competitors** (`001.csv`):
```csv
canonical_name_x,canonical_name_y,country_code_x,country_code_y,remark
"Samsung Electronics Co., Ltd.","LG Electronics Inc.","KR","KR","direct competitors in consumer electronics"
"Toyota Motor Corporation","Honda Motor Co., Ltd.","JP","JP","major Japanese automakers"
"Coca-Cola Company","PepsiCo Inc.","US","US","direct competitors in beverage industry"
"Nike Inc","Adidas AG","US","DE","athletic footwear competitors"
```

**Contextual Similarities** (`002.csv`):
```csv
canonical_name_x,canonical_name_y,country_code_x,country_code_y,remark
"Apple Inc","Adidas AG","US","DE","both consumer brands, frequently co-mentioned"
"Alphabet Inc","Meta Platforms Inc","US","US","both tech giants, frequently co-mentioned"
"Starbucks Corporation","McDonald's Corporation","US","US","both restaurant chains"
"Nike Inc","Adidas AG","US","DE","athletic footwear competitors"
```

**Similar Naming Patterns** (`003.csv`):
```csv
canonical_name_x,canonical_name_y,country_code_x,country_code_y,remark
"Capital One Financial Corporation","Capital Group Companies Inc","US","US","similar naming pattern, different companies"
"Orange SA","Orange Inc","FR","US","same brand name, different companies"
```

## 🔍 Deriving from Positive Examples

You can create negative pairs by taking canonical names from **unrelated** positive example entries:

**From positive examples**:
```csv
canonical_name,variation,country_code,source
"Apple Inc","Apple","US","SEC EDGAR"
"Microsoft Corporation","Microsoft","US","SEC EDGAR"
"Samsung Electronics Co., Ltd.","삼성전자","KR","company website"
"LG Electronics Inc.","LG전자","KR","company website"
```

**Create negative pairs** (only unrelated companies):
```csv
canonical_name_x,canonical_name_y,country_code_x,country_code_y,remark
"Apple Inc","Microsoft Corporation","US","US","unrelated tech companies"
"Samsung Electronics Co., Ltd.","LG Electronics Inc.","KR","KR","unrelated competitors"
```

**Do NOT create** (related companies in same group):
```csv
"Apple Inc","苹果电脑贸易（上海）有限公司","US","CN"  ← Same corporate group
"Samsung Electronics Co., Ltd.","Samsung Heavy Industries Co., Ltd.","KR","KR"  ← Same corporate group
```

## 🤝 How to Contribute

1. Create CSV file following the format above
2. Place in `data/negative/` directory
3. Submit PR to `dev` branch (see [CONTRIBUTING.md](../../CONTRIBUTING.md))

**Thank you for helping improve company name matching!** 🎯
