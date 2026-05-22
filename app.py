"""
app.py

Streamlit chatbot for multimodal e-commerce Q&A.
Supports text queries and product image uploads.
Uses CLIP + FAISS for retrieval and Claude (claude-sonnet-4-6) as the LLM.

Usage:
    streamlit run app.py
"""

import base64
import io
import requests
import streamlit as st
from PIL import Image
import anthropic

import rag

# Page config
st.set_page_config(
    page_title="E-Commerce AI Assistant",
    page_icon="ðŸ›’",
    layout="wide",
)

# Prompt templates

ZERO_SHOT_SYSTEM = """You are a helpful e-commerce product assistant.
Answer the customer's question using only the product information provided in the context.
If the context does not contain enough information to answer, say so honestly.
Keep answers clear and concise."""

FEW_SHOT_SYSTEM = """You are a helpful e-commerce product assistant.
Answer the customer's question using only the product information provided in the context.
If the context does not contain enough information to answer, say so honestly.
Keep answers clear and concise.

Examples of good responses:

Q: What are the features of the Samsung Galaxy S21?
A: The Samsung Galaxy S21 comes with a 6.2-inch Dynamic AMOLED display, a triple-camera \
setup (12MP wide, 64MP telephoto, 12MP ultrawide), and a 4000mAh battery.

Q: Can you compare the Amazon Echo Dot with the Google Nest Mini?
A: The Amazon Echo Dot features Alexa voice assistant, a 1.6-inch speaker, and Bluetooth \
connectivity. The Google Nest Mini comes with Google Assistant, a 40mm driver, and supports \
both Bluetooth and Wi-Fi. Both are designed for smart home control and music playback.

Q: Can you identify this product from the image?
A: Based on the image and the product database, this appears to be [product name]. \
It is used for [purpose]. Key features include [features]."""

MULTI_SHOT_SYSTEM = """You are a helpful e-commerce product assistant.
Answer the customer's question using only the product information provided in the context.
If the context does not contain enough information to answer, say so honestly.
Keep answers clear and concise.

Examples:

Q: What is the price of this item?
A: Based on the product details, the price is $X.XX.

Q: Is this suitable for children?
A: Looking at the product specifications and age recommendations, [answer based on context].

Q: What are the main differences between these two products?
A: Product A features [features]. Product B features [features]. \
The key differences are [differences].

Q: How do I use this product?
A: According to the product description, [usage instructions from context].

Q: Does this come in different colors?
A: Based on the product listing, [color/variant information from context]."""


def get_system_prompt(mode):
    if mode == "Zero-shot":
        return ZERO_SHOT_SYSTEM
    elif mode == "Few-shot":
        return FEW_SHOT_SYSTEM
    else:
        return MULTI_SHOT_SYSTEM


# Helpers
def build_context(results):
    lines = []
    for i, p in enumerate(results, 1):
        lines.append(f"Product {i}: {p['name']}")
        if p.get("brand"):
            lines.append(f"  Brand: {p['brand']}")
        if p.get("category"):
            lines.append(f"  Category: {p['category']}")
        if p.get("price"):
            lines.append(f"  Price: {p['price']}")
        if p.get("about"):
            lines.append(f"  Description: {p['about'][:500]}")
        lines.append("")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def load_image_from_url(url):
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def call_llm(api_key, system_prompt, conversation_history, context, user_text, image_bytes=None):
    client = anthropic.Anthropic(api_key=api_key)
    user_content = []
    if image_bytes is not None:
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(image_bytes).decode(),
            },
        })
    prompt_text = f"Product context from our database:\n\n{context}\n\nCustomer question: {user_text}"
    user_content.append({"type": "text", "text": prompt_text})
    messages = list(conversation_history) + [{"role": "user", "content": user_content}]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text

# Load RAG resources once
@st.cache_resource(show_spinner="Loading retrieval index...")
def init_rag():
    rag.load_resources()
    return True


try:
    init_rag()
    rag_ready = True
except FileNotFoundError as e:
    rag_ready = False
    index_error = str(e)

# Sidebar (global settings used by Chat tab)
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Anthropic API Key", type="password")
    top_k   = st.slider("Products to retrieve", min_value=1, max_value=10, value=5)
    mode    = st.selectbox("Prompt mode", ["Zero-shot", "Few-shot", "Multi-shot"])
    st.divider()
    st.subheader("Upload a product image")
    uploaded_file = st.file_uploader(
        "Upload to identify or ask about",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages       = []
        st.session_state.last_retrieved = []
        st.rerun()
    st.caption("CLIP + FAISS + Claude claude-sonnet-4-6")

if not rag_ready:
    st.error(f"Index not found. Run `python build_index.py` first.\n\n{index_error}")
    st.stop()

# Tabs
tab_chat, tab_eval, tab_docs = st.tabs(["Chat", "Evaluation", "Documentation"])

# TAB 1: CHAT
with tab_chat:
    st.header("E-Commerce AI Assistant")
    st.caption("Ask about any product by text, or upload an image in the sidebar.")

    if "messages" not in st.session_state:
        st.session_state.messages       = []
        st.session_state.last_retrieved = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.last_retrieved:
        with st.expander("Products retrieved in last query", expanded=False):
            cols = st.columns(min(len(st.session_state.last_retrieved), 3))
            for i, product in enumerate(st.session_state.last_retrieved[:3]):
                with cols[i]:
                    if product.get("image_urls"):
                        img = load_image_from_url(product["image_urls"][0])
                        if img:
                            st.image(img, use_container_width=True)
                    st.markdown(f"**{product['name'][:60]}**")
                    if product.get("price"):
                        st.caption(product["price"])
                    st.caption(f"Similarity: {product['score']:.3f}")

    user_input = st.chat_input("Ask about a product...")

    if user_input:
        if not api_key:
            st.warning("Enter your Anthropic API key in the sidebar.")
            st.stop()

        image_bytes  = None
        query_image  = None
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            query_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with st.chat_message("user"):
            st.write(user_input)
            if query_image is not None:
                st.image(query_image, width=200)

        with st.spinner("Searching product database..."):
            results = rag.retrieve(
                query_text=user_input if query_image is None else None,
                query_image=query_image,
                k=top_k,
            )
        st.session_state.last_retrieved = results
        context = build_context(results)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        with st.spinner("Generating response..."):
            try:
                answer = call_llm(
                    api_key=api_key,
                    system_prompt=get_system_prompt(mode),
                    conversation_history=history,
                    context=context,
                    user_text=user_input,
                    image_bytes=image_bytes,
                )
            except anthropic.AuthenticationError:
                st.error("Invalid API key.")
                st.stop()
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.stop()

        with st.chat_message("assistant"):
            st.write(answer)
            if results:
                cols = st.columns(min(len(results), 3))
                for i, product in enumerate(results[:3]):
                    with cols[i]:
                        if product.get("image_urls"):
                            img = load_image_from_url(product["image_urls"][0])
                            if img:
                                st.image(img, caption=product["name"][:40], use_container_width=True)

        st.session_state.messages.append({"role": "user",      "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
        
# TAB 2: EVALUATION
with tab_eval:
    st.header("Retrieval Evaluation")
    st.markdown(
        "The retrieval system is evaluated using a synthetic test set. "
        "Each product's name is used as the query, and the ground truth is that "
        "the same product should appear in the top-k results."
    )

    # Pre-computed results (from running evaluate.py)
    PRE_COMPUTED = {
        "Accuracy@1": 0.9400,
        "Recall@1":   0.9400,
        "Recall@5":   0.9850,
        "Recall@10":  0.9900,
    }

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Pre-computed Results (n=200)")
        for metric, value in PRE_COMPUTED.items():
            st.metric(label=metric, value=f"{value:.2%}")

    with col2:
        st.subheader("Metric Chart")
        import pandas as pd
        df = pd.DataFrame({
            "Metric": list(PRE_COMPUTED.keys()),
            "Score":  list(PRE_COMPUTED.values()),
        })
        st.bar_chart(df.set_index("Metric"))

    st.divider()
    st.subheader("Run Live Evaluation")
    st.caption("Runs CLIP on 200 product-name queries and checks retrieval. Takes ~2 minutes.")

    if st.button("Run Evaluation Now"):
        import random

        products = rag.get_products()
        random.seed(42)
        sample   = random.sample(products, min(200, len(products)))
        queries  = [{"query": p["name"], "product_id": p["id"]} for p in sample if p["name"]]

        progress = st.progress(0, text="Running evaluation...")
        hits     = {1: 0, 5: 0, 10: 0}
        n        = len(queries)

        for i, item in enumerate(queries):
            results     = rag.retrieve(query_text=item["query"], k=10)
            retrieved   = [r["id"] for r in results]
            for k in (1, 5, 10):
                if item["product_id"] in retrieved[:k]:
                    hits[k] += 1
            progress.progress((i + 1) / n, text=f"Evaluated {i+1}/{n}")

        live_results = {
            "Accuracy@1": round(hits[1] / n, 4),
            "Recall@1":   round(hits[1] / n, 4),
            "Recall@5":   round(hits[5] / n, 4),
            "Recall@10":  round(hits[10] / n, 4),
        }
        progress.empty()
        st.success("Evaluation complete.")
        cols = st.columns(4)
        for col, (metric, value) in zip(cols, live_results.items()):
            col.metric(label=metric, value=f"{value:.2%}")

    st.divider()
    st.subheader("Methodology")
    st.markdown("""
**Test set construction:** 200 products are sampled uniformly at random from the 10,002-product
index. Each product's exact name is used as the query. The ground truth is that the queried
product should appear in the top-k retrieved results.

**Why this works as a proxy:** CLIP encodes both the stored description and the query in the
same embedding space. A product name query that exactly matches the indexed description should
rank near the top if retrieval is working correctly. This is a strict test because 10,001 other
products are competing distractors.

**Limitations:** This only measures the text-to-text retrieval path. Cross-modal retrieval
(image query to text index) is harder to evaluate without a manually labeled image test set.
    """)

# TAB 3: DOCUMENTATION
with tab_docs:
    st.header("System Documentation")

    st.subheader("Architecture Overview")
    st.markdown("""
```
User Query (text or image)
        |
        v
  CLIP Encoder
  (openai/clip-vit-base-patch32)
        |
        v
  512-dim L2-normalized embedding
        |
        v
  FAISS IndexFlatIP
  (10,002 product embeddings)
        |
        v
  Top-k retrieved products
  (name, brand, price, description, image URLs)
        |
        v
  Claude claude-sonnet-4-6
  (system prompt + retrieved context + query)
        |
        v
  Natural language response + product images
```
    """)

    st.subheader("Data Preprocessing")
    st.markdown("""
**Dataset:** Amazon Product Dataset 2020 (10,002 products)

**Fields used to build product descriptions:**
| Field | Usage |
|---|---|
| Product Name | Primary identifier, included first |
| Brand Name | Appended as "Brand: X" |
| Category | Top-level category only (before first pipe character) |
| Selling Price | Appended as "Price: X" |
| About Product | First 300 characters, pipe-delimited features converted to sentences |

Fields like `Product Specification`, `Technical Details`, and `Variants` were excluded
because they contain structured markup that degrades CLIP text encoding quality.
Image URLs are stored in metadata but not embedded at index time.

**Text truncation:** CLIP's text encoder has a hard limit of 77 tokens. The combined
description is passed through the processor with `truncation=True, max_length=77`.
    """)

    st.subheader("CLIP Embedding Model")
    st.markdown("""
**Model:** `openai/clip-vit-base-patch32`

CLIP (Contrastive Language-Image Pre-training) learns a shared embedding space for text
and images by training on 400 million image-text pairs. The key property that makes it
useful here is **cross-modal alignment**: a text query like "red running shoes" and an
image of red running shoes map to nearby vectors in the same 512-dimensional space.

For this project:
- **Text embeddings** are generated via `text_model` + `text_projection` (512-dim)
- **Image embeddings** are generated via `vision_model` + `visual_projection` (512-dim)
- All vectors are L2-normalized so inner product equals cosine similarity

Batch size of 128 is used during index construction. Total embedding time for 10,002
products is roughly 3-4 minutes on CPU.
    """)

    st.subheader("Vector Database (FAISS)")
    st.markdown("""
**Index type:** `faiss.IndexFlatIP` (exact inner product search)

Since all embeddings are L2-normalized, inner product is equivalent to cosine similarity.
`IndexFlatIP` does an exact brute-force search, which is appropriate at this dataset size
(10k vectors, 512 dimensions). At larger scale, an approximate index like `IndexIVFFlat`
or `IndexHNSW` would be preferred.

The index and metadata (product dicts) are saved to disk as `product_index.faiss` and
`product_metadata.json` so they only need to be built once.
    """)

    st.subheader("RAG Pipeline")
    st.markdown("""
At query time:
1. The user's text or uploaded image is encoded with CLIP
2. FAISS returns the top-k most similar products (default k=5)
3. Product metadata (name, brand, price, description) is formatted into a context block
4. The context block + user question are sent to Claude as a user message

This is a **retrieve-then-read** RAG pattern. The retrieval step grounds the LLM in
real product data, reducing hallucination. The LLM's job is to synthesize and explain
the retrieved information, not to recall product facts from training weights.
    """)

    st.subheader("LLM and Prompting")
    st.markdown("""
**Model:** `claude-sonnet-4-6` via the Anthropic API

Claude is used because it natively supports both text and image inputs in the same
request, which is needed for the image upload path. The retrieved product context is
always passed as text regardless of whether the query was an image or text.

**Three prompt modes are available:**

| Mode | Description |
|---|---|
| Zero-shot | System prompt sets the role and task. No examples. |
| Few-shot | System prompt includes 3 worked examples of product Q&A. |
| Multi-shot | System prompt includes 5 examples covering price, comparison, usage, and variant queries. |

Conversation history is maintained so follow-up questions work correctly. Images are
only attached on the turn they are uploaded; subsequent turns use text only.
    """)

    st.subheader("File Structure")
    st.markdown("""
```
final/
  amazon_data/          Raw dataset CSV
  build_index.py        One-time preprocessing + CLIP embedding + FAISS build
  rag.py                Retrieval utilities (load index, encode query, search)
  evaluate.py           Offline evaluation script (Recall@1/5/10)
  app.py                This Streamlit application
  requirements.txt      Python dependencies
  product_index.faiss   Generated FAISS index (10,002 vectors, 512-dim)
  product_metadata.json Generated product metadata
```
    """)

