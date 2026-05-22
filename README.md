# Vision E-Commerce Chatbot

Multimodal e-commerce chatbot that answers product questions from text or image inputs. Uses CLIP for retrieval, FAISS for vector search, and Claude as the LLM. Built on the Amazon Product Dataset 2020 (10,002 products).

## Stack

- **CLIP** (ViT-B/32) for shared text and image embeddings
- **FAISS** for fast cosine similarity search over 10k product vectors
- **Claude** (claude-sonnet-4-6) for response generation
- **Streamlit** for the UI

## Setup

```bash
pip install -r requirements.txt
```

Build the index once (downloads CLIP, embeds all products, ~5 min on CPU):

```bash
python build_index.py
```

Run the app:

```bash
streamlit run app.py
```

Enter your [Anthropic API key](https://console.anthropic.com/) in the sidebar.

## Usage

- **Text query:** type a product question in the chat box
- **Image query:** upload a product image in the sidebar, then ask about it
- Switch between zero-shot, few-shot, and multi-shot prompting modes in the sidebar
- The Evaluation tab shows Recall@1/5/10 metrics for the retrieval system
- The Documentation tab explains the architecture and preprocessing decisions

## Retrieval Performance

| Metric | Score |
|---|---|
| Recall@1 | 0.94 |
| Recall@5 | 0.985 |
| Recall@10 | 0.99 |

Evaluated on 200 random product-name queries against 10,002 products.

## Files

```
build_index.py       Preprocessing + CLIP embedding + FAISS index builder
rag.py               Retrieval utilities
evaluate.py          Offline Recall@1/5/10 evaluation
app.py               Streamlit application
report.tex           LaTeX research report (Overleaf-ready)
documentation.md     Detailed system documentation
requirements.txt     Dependencies
```
