#  🤼‍♂️ Company Name Matcher

Company Name Matcher is a library for efficient and accurate matching of company names using advanced embedding techniques. It leverages a fine-tuned sentence transformer model to generate embeddings specifically tailored for company names.

## Features

- Efficient company name comparison using embeddings and cosine similarity
- Scalable vector search for large datasets
- Context-aware matching that understands company name nuances
- Significantly faster than traditional string matching algorithms for large datasets

## Advantages over Traditional Methods

While traditional string matching algorithms like those used in RapidFuzz are fast for small datasets, Company Name Matcher offers several advantages, especially when dealing with larger datasets:

1. **Scalability**: As the lists of company names grow, our embedding-based approach scales much better. The computational complexity is reduced from O(n * m) to O(n log m), where n is the length of name list A and m is the length of name list B.

2. **Contextual Understanding**: Our embeddings capture the context and semantics of company names, allowing for more intelligent matching that goes beyond simple string similarity.

3. **Vector Search**: The use of embeddings enables efficient vector search techniques, which are highly scalable and can be optimized for large datasets.

4. **Customizability**: The underlying model can be fine-tuned on domain-specific data, allowing for improved performance in specialized use cases.

## Methodology (Model fine-tuning)

1. **Fine-tuned Embeddings**: We use a sentence transformer model fine-tuned specifically for company names. This model was trained using contrastive learning, minimizing the cosine distance between similar company names.

2. **Special Tokens**: During the training process, we added special tokens to the training data. These tokens guide the model's understanding, explicitly informing it that it's embedding company names. This results in more accurate and context-aware embeddings.

3. **Cosine Similarity**: We use cosine similarity to compare the resulting embeddings, providing a robust measure of similarity that works well with high-dimensional data.

## Performance Comparison

Here's a comparison of Company Name Matcher with RapidFuzz on a test dataset:

| Metric        | Company Name Matcher | RapidFuzz |
|---------------|----------------------|-----------|
| Time taken    | 8.9385 seconds       | 0.0042 seconds |
| Accuracy      | 0.855                | 0.670     |
| Precision     | 0.928                | 0.886     |
| Recall        | 0.770                | 0.390     |
| F1 Score      | 0.842                | 0.542     |

While RapidFuzz is faster, Company Name Matcher provides better accuracy and scalability (as the lists for matching increase in size, we can use k-means approximated matching).

## Installation

```
!git clone https://github.com/easonanalytica/company_name_matcher.git
pip install .
```

Download our fine-tuned model [here](https://drive.google.com/file/d/13L-yKsb0TYMb3UF68-XOvO8Lf-efTRiT/view?usp=sharing) on Google Drive.

## Usage

First, import the `CompanyNameMatcher` class and initialize it with your model path:

```python
from company_name_matcher import CompanyNameMatcher

matcher = CompanyNameMatcher("models/fine_tuned_model")

# Compare two company names
similarity = matcher.compare_companies("Apple Inc", "Apple Incorporated")
print(f"Similarity: {similarity}")

# Find matches in a list of companies
company_list = ["Microsoft Corporation", "Apple Inc", "Google LLC", "Apple Computer Inc"]
matches = matcher.find_matches("Apple", company_list, threshold=0.8)
print("Matches:", matches)

# Get embedding for a single company
embedding = matcher.get_embedding("Apple Inc")
print(f"Embedding: {embedding}")

# Get embeddings for multiple companies
embeddings = matcher.get_embeddings(["Microsoft", "Google"])
print(f"Embeddings: {embeddings}")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
