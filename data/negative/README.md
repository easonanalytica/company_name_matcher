# ➖ Contributing Negative Data Examples

Negative examples are pairs of **canonical legal entity names** that represent **unrelated companies** but may produce similar embeddings in pre-trained models.

## 🎯 What are Negative Examples?

Pairs of `canonical_name_x` and `canonical_name_y` representing **unrelated companies**. These help the model distinguish between:

- **Industry Competitors**: "Samsung Electronics Co., Ltd." vs "LG Electronics Inc." (unrelated competitors)
- **Contextually Similar**: "Apple Inc." vs "Adidas A.G." (both consumer brands, frequently co-mentioned)
- **Similar Naming**: "Capital One Financial Corporation" vs "Capital Group Companies Inc." (similar names, unrelated)

⚠️ **IMPORTANT — Do NOT Include**:
- **Same corporate group**: "Apple Inc." vs "苹果（中国）有限公司" (different legal entities, but same group)
- **Parent/subsidiary**: "Alphabet Inc." vs "Google LLC"
- **Sister companies**: "Samsung Electronics" vs "Samsung Heavy Industries"

These related entities should naturally score in the middle range (0.5-0.8), not as negatives (0.0). Labeling them 0 would incorrectly train the model to push them apart.

**Note**: Negative examples can be derived from positive examples across **unrelated** `canonical_name` entries.

## 📋 Parquet Format

**Filename**: `{country_code_x}_{country_code_y}.parquet` (e.g., [`US_US.parquet`](US_US.parquet), [`FR_US.parquet`](FR_US.parquet))

**Columns**:
- `canonical_name_x` *(required)*: Official legal entity name (first company)
- `canonical_name_y` *(required)*: Official legal entity name (second company)
- `country_code_x` *(required)*: Two-letter ISO country code for first company
- `country_code_y` *(required)*: Two-letter ISO country code for second company
- `remark` *(optional)*: Explanation of why these are negative examples

**Rules**:
- The filename should have the country codes in **alphabetical order** only (e.g. ✅ `FR_US.parquet`, ❌ `US_FR.parquet`)
- `country_code_x` & `canonical_name_x` refer to the **first part** of the filename, while `country_code_y` & `canonical_name_y` refer to the **second part** of the filename (`{country_code_x}_{country_code_y}.parquet`) 
- One negative pair per row
- Both must be **different legal entities**
- Use canonical legal entity names (same as positive examples)
- Apache Parquet format with zstd compression
- Country codes in filename must be sorted alphabetically (CN_US.parquet, not US_CN.parquet)
-`country_code_x` and `country_code_y` should be in titlecase

## 📊 Examples

**Industry Competitors**:

1. [`US_US.parquet`](US_US.parquet)

    |canonical_name_x|canonical_name_y|country_code_x|country_code_y|remark|
    |----------------|----------------|--------------|--------------|------|
    |Coca-Cola Company|PepsiCo Inc.|US|US|direct competitors in beverage industry|

2. [`DE_US.parquet`](DE_US.parquet)

    |canonical_name_x|canonical_name_y|country_code_x|country_code_y|remark|
    |----------------|----------------|--------------|--------------|------|
    |Adidas A.G.|Nike Inc.|DE|US|athletic footwear competitors|

**Contextual Similarities**:

1. [`US_US.parquet`](US_US.parquet)

    |canonical_name_x|canonical_name_y|country_code_x|country_code_y|remark|
    |----------------|----------------|--------------|--------------|------|
    |Alphabet Inc.|Meta Platforms Inc.|US|US|both tech giants, frequently co-mentioned|
    |Starbucks Corporation|McDonald's Corporation|US|US|both restaurant chains|

2. [`DE_US.parquet`](DE_US.parquet)

    |canonical_name_x|canonical_name_y|country_code_x|country_code_y|remark|
    |----------------|----------------|--------------|--------------|------|
    |Adidas A.G.|Nike Inc.|DE|US|athletic footwear competitors|
    |Adidas A.G.|"Apple Inc.|DE|US|both consumer brands, frequently co-mentioned|


**Similar Naming Patterns**:

1. [`US_US.parquet`](US_US.parquet)

    |canonical_name_x|canonical_name_y|country_code_x|country_code_y|remark|
    |----------------|----------------|--------------|--------------|------|
    |Capital One Financial Corporation|Capital Group Companies Inc.|US|US|similar naming pattern, different companies|

2. [`FR_US.parquet`](FR_US.parquet)

    |canonical_name_x|canonical_name_y|country_code_x|country_code_y|remark|
    |----------------|----------------|--------------|--------------|------|
    |Orange S.A.|Orange Inc.|FR|US|same brand name, different companies|


## 🔍 Deriving from Positive Examples

You can create negative pairs by taking canonical names from **unrelated** positive example entries:

**From positive examples**:

|canonical_name|variation|country_code|source|
|--------------|---------|------------|------|
|Apple Inc.|Apple|US|SEC EDGAR|
|International Business Machines Corporation|IBM|US|abbreviation|
|Alphabet Inc.|Google|US|common usage|

**Create negative pairs** (only unrelated companies):

|canonical_name_x|canonical_name_y|country_code_x|country_code_y|remark|
|----------------|----------------|--------------|--------------|------|
|Apple Inc.|International Business Machines Corporation|US|US|unrelated tech companies|
|Apple Inc.|Alphabet Inc.|US|US|unrelated tech companies|
|Alphabet Inc.|International Business Machines Corporation|US|US|unrelated tech companies|


**Do NOT create** (related companies in same group):

|canonical_name_x|canonical_name_y|country_code_x|country_code_y|remark|
|----------------|----------------|--------------|--------------|------|
|苹果电脑贸易（上海）有限公司|Apple Inc.|CN|US|Same corporate group|


## 🤝 How to Contribute

1. Create a parquet file following the format above
2. Place in `data/negative/` directory
3. Validate by running:
    ```bash
    python scripts/validate_data.py
    ```
    and all validations should pass
4. Submit PR to `dev` branch (see [CONTRIBUTING.md](../../CONTRIBUTING.md))

**Thank you for helping improve company name matching!** 🎯
