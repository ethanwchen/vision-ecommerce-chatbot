"""
rag.py

Retrieval utilities: load the FAISS index and CLIP model once, then expose
retrieve() for both text and image queries.
"""

import json
import numpy as np
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

INDEX_PATH = "product_index.faiss"
META_PATH  = "product_metadata.json"

# Module-level singletons so resources are loaded only once per process.
_index     = None
_products  = None
_model     = None
_processor = None


def load_resources():
    global _index, _products, _model, _processor
    if _index is not None:
        return
    if not __import__("os").path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"{INDEX_PATH} not found. Run build_index.py first."
        )
    _index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        _products = json.load(f)
    _model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _model.eval()


def retrieve(query_text=None, query_image=None, k=5):
    """
    Retrieve the top-k most similar products.

    Args:
        query_text:  str  - a natural-language query
        query_image: PIL.Image or None - an uploaded product image
        k:           int  - number of results to return

    Returns:
        list of product dicts, each with an added "score" field.
    """
    load_resources()

    if query_image is not None:
        inputs = _processor(images=query_image, return_tensors="pt")
        with torch.no_grad():
            out   = _model.vision_model(**inputs)
            feats = _model.visual_projection(out.pooler_output)
    else:
        inputs = _processor(
            text=[query_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            out   = _model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            feats = _model.text_projection(out.pooler_output)

    feats = feats / feats.norm(dim=-1, keepdim=True)
    query_vec = feats.cpu().numpy().astype("float32")

    scores, indices = _index.search(query_vec, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            product = dict(_products[idx])
            product["score"] = float(score)
            results.append(product)
    return results


def get_products():
    load_resources()
    return _products
