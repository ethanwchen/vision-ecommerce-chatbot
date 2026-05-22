"""
build_index.py

Run once to preprocess the Amazon Product Dataset, generate CLIP text embeddings
for all products, and store them in a FAISS index alongside product metadata.

Usage:
    python build_index.py
"""

import csv
import json
import os
import numpy as np
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

DATA_PATH = os.path.join(
    "amazon_data", "home", "sdf",
    "marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv"
)
INDEX_PATH = "product_index.faiss"
META_PATH  = "product_metadata.json"

BATCH_SIZE = 128


def make_description(row):
    """Combine key product fields into a single text description for CLIP."""
    parts = []
    if row.get("Product Name"):
        parts.append(row["Product Name"].strip())
    if row.get("Brand Name"):
        parts.append("Brand: " + row["Brand Name"].strip())
    if row.get("Category"):
        top_cat = row["Category"].split("|")[0].strip()
        parts.append("Category: " + top_cat)
    if row.get("Selling Price"):
        parts.append("Price: " + row["Selling Price"].strip())
    if row.get("About Product"):
        about = row["About Product"].replace("|", ". ").strip()[:300]
        parts.append(about)
    return " | ".join(parts)


def load_products():
    products = []
    with open(DATA_PATH, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            desc = make_description(row)
            if not desc.strip():
                continue
            image_urls = [u.strip() for u in row.get("Image", "").split("|") if u.strip()]
            products.append({
                "id":         row.get("Uniq Id", ""),
                "name":       row.get("Product Name", "").strip(),
                "brand":      row.get("Brand Name", "").strip(),
                "category":   row.get("Category", "").split("|")[0].strip(),
                "price":      row.get("Selling Price", "").strip(),
                "about":      row.get("About Product", "").replace("|", ". ").strip()[:800],
                "image_urls": image_urls[:3],
                "description": desc,
            })
    return products


def get_text_embeddings(texts, model, processor):
    """Get L2-normalised CLIP text embeddings via text_model + text_projection."""
    out   = model.text_model(input_ids=texts["input_ids"], attention_mask=texts["attention_mask"])
    feats = model.text_projection(out.pooler_output)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def generate_embeddings(descriptions, model, processor):
    all_embeddings = []
    total = len(descriptions)
    for i in range(0, total, BATCH_SIZE):
        batch  = descriptions[i : i + BATCH_SIZE]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            feats = get_text_embeddings(inputs, model, processor)
        all_embeddings.append(feats.cpu().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Embedded {min(i + BATCH_SIZE, total)}/{total}")
    return np.vstack(all_embeddings).astype("float32")


def build_index():
    print("Loading product data...")
    products = load_products()
    print(f"  {len(products)} products loaded")

    print("Loading CLIP model (openai/clip-vit-base-patch32)...")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    print("Generating text embeddings...")
    descriptions = [p["description"] for p in products]
    embeddings   = generate_embeddings(descriptions, model, processor)

    print("Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product on L2-normalised vecs = cosine sim
    index.add(embeddings)

    print(f"Saving index to {INDEX_PATH} ...")
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving metadata to {META_PATH} ...")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False)

    print(f"Done. Index contains {index.ntotal} vectors of dimension {dim}.")


if __name__ == "__main__":
    build_index()
