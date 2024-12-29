<p align="center">
  <img src="https://github.com/easonanalytica/company_name_matcher/blob/dev/assets/logo.png?raw=true" alt="Company Name Matcher Logo" width="200"/>
</p>
<h1 align="center">Company Name Matcher</h1>
<p align="center">
  <a href="https://badge.fury.io/py/company_name_matcher"><img src="https://badge.fury.io/py/company_name_matcher.svg" alt="PyPI version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <a href="https://github.com/easonanalytica/company_name_matcher/stargazers"><img src="https://img.shields.io/github/stars/easonanalytica/company_name_matcher.svg" alt="GitHub stars"></a>
</p>


Company Name Matcher is a library for efficient matching of company names using advanced embedding techniques. It leverages a language model to generate embeddings specifically tailored for company names.

## Advantages over Traditional Methods

While traditional string matching algorithms like those used in RapidFuzz are fast for small datasets, Company Name Matcher offers several advantages, especially when dealing with larger datasets:

1. **Scalability**: Embedding-based approach offers superior scalability for larger datasets through efficient vector search techniques. Unlike traditional methods that require comparing every name in list A with every name in list B (O(n * m) complexity), this approach reduces computational complexity to O(n log m) by leveraging optimized vector search, where n and m represent the lengths of the respective name lists.

2. **Contextual Understanding**: Embeddings capture the context and semantics of company names, allowing for more intelligent matching that goes beyond simple string similarity.

3. **Customizability**: The underlying model can be fine-tuned on domain-specific data, allowing for improved performance in specialized use cases.


## 🚀 Installation

```
pip install company-name-matcher
```

An optional installation with "pip install . --no-binary scikit-learn" is recommended to fix an OpenMP compatibility issue with sklearn.

## 📣 Features

- **K-Means approximated matching**: Use vector search with either exact or approximate matching
- **Easily expand index**: Easily add new companies to the existing index without rebuilding the index from scratch
- **Working with embeddings**: Direct access to embeddings for further analysis

## 📚 Quick Start

### 1. Basic Usage

```python
from company_name_matcher import CompanyNameMatcher

# Initialize with default model
matcher = CompanyNameMatcher("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Or initialize with custom preprocessing
def preprocess_name(name):
    return name.lower().strip()

matcher = CompanyNameMatcher(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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

## 📊 Performance Considerations

1. For small datasets (<10,000 companies), use exact matching (`use_approx=False`)
2. For large datasets, use approximate matching (`use_approx=True`) with appropriate `n_clusters`
3. When using approximate matching:
   - Build the index once and save it to disk
   - Load the index for subsequent uses
   - Adjust `n_clusters` based on your dataset size and speed/accuracy requirements


## 🤖 (Complementary) fine-tuned model

While you can load your own model into CompanyNameMatcher, we provide our complementary fine-tuned model avaliable for download [here](https://drive.google.com/file/d/11LaI2-1Ahqqfo73CKOPNgRSCJe_y9nnG/view?usp=sharing) on Google Drive. See demo [here](demo.ipynb).

1. **Fine-tuned Embeddings**: We use a lightweight multilingual sentence transformer model (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) fine-tuned specifically for company names. This model was trained using contrastive learning, minimizing the cosine distance between similar company names.

2. **Special Tokens**: During the training process, we added special tokens to the training data. These tokens guide the model's understanding, explicitly informing it that it's embedding company names. This results in more accurate and context-aware embeddings.

3. **Cosine Similarity**: We use cosine similarity to compare the resulting embeddings, providing a robust measure of similarity that works well with high-dimensional data.

### Performance Comparison

Here's a sample comparison of Company Name Matcher with RapidFuzz on the test dataset with threshold 0.9:

| Metric        | Company Name Matcher | RapidFuzz |
|---------------|----------------------|-----------|
| Accuracy      | 0.855                | 0.670     |
| Precision     | 0.928                | 0.886     |
| Recall        | 0.770                | 0.390     |
| F1 Score      | 0.842                | 0.542     |

While RapidFuzz is faster, Company Name Matcher provides better accuracy and scalability (as the lists for matching increase in size, we can use k-means approximated matching).


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
