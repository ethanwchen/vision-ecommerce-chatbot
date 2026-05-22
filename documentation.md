# Multimodal E-Commerce Chatbot: Documentation

## Overview

This system is a multimodal conversational chatbot for e-commerce product support. It can answer product questions from both text queries and uploaded product images. Under the hood it uses CLIP for embedding, FAISS for retrieval, and Claude as the LLM.

---

## Project Structure

```
final/
  amazon_data/                  Raw dataset
    home/sdf/
      *.csv                     Amazon Product Dataset 2020 (10,002 products)
  build_index.py                One-time script: builds CLIP embeddings and FAISS index
  rag.py                        Retrieval utilities
  evaluate.py                   Offline evaluation (Recall@1/5/10)
  app.py                        Streamlit application
  requirements.txt              Python dependencies
  product_index.faiss           Generated FAISS index (10,002 x 512 float32)
  product_metadata.json         Generated product metadata
  report.tex                    LaTeX research report
  documentation.md              This file
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the index (run once)

```bash
python build_index.py
```

This loads the CSV, generates CLIP text embeddings for all 10,002 products in batches of 128,
and saves the FAISS index and metadata to disk. Takes roughly 4-5 minutes on CPU.

### 3. Run the app

```bash
streamlit run app.py
```

Enter your Anthropic API key in the sidebar. The index loads automatically on first use.

### 4. Run evaluation (optional)

```bash
python evaluate.py
```

Outputs Accuracy@1, Recall@1, Recall@5, and Recall@10.

---

## Data Preprocessing

**Dataset:** Amazon Product Dataset 2020 (Kaggle)
**Size:** 10,002 products across 28 columns

**Fields used for product descriptions:**

| Field | Notes |
|---|---|
| Product Name | Always included first |
| Brand Name | Formatted as "Brand: X" |
| Category | Top-level category only (first segment before `\|`) |
| Selling Price | Formatted as "Price: X" |
| About Product | Truncated to 300 chars; pipe delimiters replaced with `. ` |

Fields like `Product Specification`, `Technical Details`, and `Variants` were excluded.
They contain structured markup (inline HTML, encoded characters) that hurts CLIP embedding quality.

Combined descriptions are tokenized with `truncation=True, max_length=77` to fit CLIP's limit.

---

## Model Architecture

### CLIP (openai/clip-vit-base-patch32)

CLIP learns a shared 512-dimensional embedding space for both text and images. This enables
cross-modal retrieval: a text query and a visually similar image will map to nearby vectors.

In transformers v5.x, `get_text_features()` returns a dataclass object rather than a tensor.
The workaround is to call the sub-models directly:

```python
# Text
out   = model.text_model(input_ids=..., attention_mask=...)
feats = model.text_projection(out.pooler_output)
feats = feats / feats.norm(dim=-1, keepdim=True)

# Image
out   = model.vision_model(pixel_values=...)
feats = model.visual_projection(out.pooler_output)
feats = feats / feats.norm(dim=-1, keepdim=True)
```

### FAISS (IndexFlatIP)

Exact inner product search over L2-normalized vectors is equivalent to cosine similarity search.
`IndexFlatIP` is brute-force and exact, which is fine at 10k scale. No ANN approximation needed.

### Claude (claude-sonnet-4-6)

The LLM used for response generation. It accepts both text and image inputs in a single request,
making it straightforward to pass an uploaded image alongside the retrieved product context.

---

## RAG Pipeline

```
Query (text or image)
  -> CLIP encode -> 512-dim vector
  -> FAISS search -> top-k product indices
  -> Fetch metadata -> build context string
  -> Claude API call (system prompt + context + query)
  -> Natural language response
```

The LLM sees structured product context like:

```
Product 1: DB Longboards CoreFlex Crossbow 41" Bamboo Fiberglass Longboard
  Brand: DB Longboards
  Category: Sports & Outdoors
  Price: $237.68
  Description: RESPONSIVE FLEX: The Crossbow features a bamboo core...
```

---

## Prompting Modes

| Mode | System prompt includes |
|---|---|
| Zero-shot | Role definition and task only |
| Few-shot | 3 worked Q&A examples (features, comparison, image ID) |
| Multi-shot | 5 worked Q&A examples (price, suitability, comparison, usage, variants) |

---

## Evaluation Results

Test set: 200 products sampled with seed=42. Each product's name is used as the query.

| Metric | Score |
|---|---|
| Accuracy@1 | 0.9400 |
| Recall@1 | 0.9400 |
| Recall@5 | 0.9850 |
| Recall@10 | 0.9900 |

---

## Known Limitations

- **Image URL availability:** Many Amazon CDN URLs from 2020 return 403 or timeout. Images are
  fetched on-demand with error handling; failures show no image rather than crashing.
- **Cross-modal evaluation:** The evaluation only covers text-to-text retrieval. There is no
  labeled image test set in this dataset.
- **CLIP token limit:** 77 tokens is short for detailed product descriptions. Pre-truncation
  to 300 characters for the `About Product` field keeps the most important fields in budget.
