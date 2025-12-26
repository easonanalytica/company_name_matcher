<p align="center">
  <img src="https://github.com/easonanalytica/company_name_matcher/blob/dev/assets/logo.png?raw=true" alt="Company Name Matcher Logo" width="200"/>
</p>
<h1 align="center">Company Name Matcher</h1>
<p align="center">
  <a href="https://badge.fury.io/py/company_name_matcher"><img src="https://badge.fury.io/py/company_name_matcher.svg" alt="PyPI version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <a href="https://pepy.tech/project/company-name-matcher"><img src="https://static.pepy.tech/badge/company-name-matcher" alt="Downloads"></a>
</p>


Company Name Matcher is a library for efficient matching of company names using vector search. It leverages a language model to generate embeddings specifically tailored for company names.

## Why Company Name Matching?

**Reconciliation is a universal task** in data-driven industries. Company names appear inconsistently across different sources - "Apple Inc", "Apple Incorporated", "苹果公司" might all refer to the same entity.

**Key Use Cases:**
- **📊 Big Data Analytics**: Unify company mentions across news, social media, and financial reports
- **💼 Portfolio Analysis**: Match investment holdings against multiple data sources
- **🔗 Supply Chain Analytics**: Track vendors and suppliers across procurement systems
- **📈 Market Intelligence**: Aggregate company data from diverse sources for comprehensive analysis
- **🏦 Financial Reporting**: Ensure consistent entity identification across regulatory filings

## Advantages over Traditional Methods

While traditional string matching algorithms like those used in RapidFuzz are fast for small datasets, Company Name Matcher offers several advantages, especially when dealing with larger datasets:

1. **Scalability**: Embedding-based approach offers superior scalability for larger datasets through efficient vector search techniques. Unlike traditional methods that require comparing every name in list A with every name in list B (O(n * m) complexity), this approach reduces computational complexity to O(n log m) by leveraging optimized vector search, where n and m represent the lengths of the respective name lists.

2. **Contextual Understanding**: Embeddings capture the context and semantics of company names, allowing for more intelligent matching that goes beyond simple string similarity.

3. **Customizability**: The underlying model can be fine-tuned on domain-specific data, allowing for improved performance in specialized use cases.


## 🚀 Installation

```
pip install company-name-matcher
```

## 📣 Features

- **K-Means approximated matching**: Use vector search with either exact or approximate matching
- **Easily expand index**: Easily add new companies to the existing index without rebuilding the index from scratch
- **Efficient batch processing**: Process multiple companies in parallel and with caching for faster matching

## 📚 Quick Start

### 1. Basic Usage

```python
from company_name_matcher import CompanyNameMatcher

# Initialize with default model
matcher = CompanyNameMatcher("paraphrase-multilingual-MiniLM-L12-v2")

# Or initialize with custom preprocessing
def preprocess_name(name):
    return name.lower().strip()

matcher = CompanyNameMatcher(
    "paraphrase-multilingual-MiniLM-L12-v2",
    preprocess_fn=preprocess_name
)

# Compare two company names
similarity = matcher.compare_companies("Apple Inc", "Apple Incorporated")
print(f"Similarity: {similarity}")
```

### 2. Bulk Matching with Vector Search

For large datasets, you can use vector search with either exact or approximate matching:

```python
# Your list of companies to match against
companies_to_match = ["Microsoft Corporation", "Apple Inc", "Google LLC", ...]

# Build and save index (only needed once)
matcher.build_index(
    companies_to_match,
    n_clusters=20,  # Adjust based on dataset size
    save_dir="index_files"  # Optional: save index to disk
)

# Or load a previously saved index
matcher.load_index(load_dir="index_files")

# 1. Exact Search (more accurate but slower)
exact_matches = matcher.find_matches(
    "Apple",
    threshold=0.7,
    k=5,
    use_approx=False
)
print("Exact matches:", exact_matches)

# 2. Approximate Search (faster but may miss some matches)
approx_matches = matcher.find_matches(
    "Apple",
    threshold=0.7,
    k=5,
    use_approx=True
)
print("Approximate matches:", approx_matches)
```

### 3. Working with Embeddings

You can also work directly with the embeddings:

```python
# Get embedding for a single company
embedding = matcher.get_embedding("Apple Inc")
print(f"Embedding shape: {embedding.shape}")

# Get embeddings for multiple companies
embeddings = matcher.get_embeddings(["Microsoft", "Google"])
print(f"Embeddings shape: {embeddings.shape}")
```


## 🤖 Fine-tune Your Model

As PoC, we provide a complementary fine-tuned model on [Hugging Face](https://huggingface.co/easonanalytica/cnm-multilingual-small-v2). See a demo [here](demo.ipynb).

1. **Fine-tuned Embeddings**: We used a lightweight multilingual sentence transformer model fine-tuned specifically for company names. This model was trained using contrastive learning, minimizing the cosine distance between similar company names.

2. **Special Tokens**: During the training process, we added special tokens **$** to the training data. These tokens guide the model's understanding, explicitly informing it that it's embedding company names. This results in more accurate and context-aware embeddings.

3. **Cosine Similarity**: We use cosine similarity to compare the resulting embeddings, providing a robust measure of similarity that works well with high-dimensional data.

## 📊 Performance Comparison

| Model            | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| Fine-tuned Matcher | 0.724   | 0.974    | 0.521 | 0.679   |
| Default Matcher    | 0.685   | 0.861    | 0.521 | 0.649   |
| RapidFuzz         | 0.614   | 1.000    | 0.310 | 0.473   |

**Based on evaluation of hard test data** ([`tests/test_data.csv`](tests/test_data.csv)) containing company name pairs in English, Chinese (中文), Japanese (日本語), Korean (한국어). Decision threshold: 0.7 for all three.

*Note: Performance metrics may vary based on your specific use case and data distribution.*


## 🤝 Contributing

We welcome contributions! Whether you're improving code, adding training data, or reporting issues:

- **Code contributions**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
- **Bug reports & feature requests**: Open an [issue on GitHub](https://github.com/easonanalytica/company_name_matcher/issues).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
