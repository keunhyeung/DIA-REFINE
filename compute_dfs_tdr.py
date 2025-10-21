#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anonymized Standalone Script for Calculating DFS and TDR Metrics

Description:
This script calculates the Dialect Fidelity Score (DFS) and Target Dialect 
Ratio (TDR) for a given set of translations from a single CSV file.

It assumes you have a CSV file with three specific columns:
1.  'source': The original standard language text.
2.  'reference': The ground-truth dialect text.
3.  'hypothesis': The model's generated hypothesis text.

Metrics:
-   DFS (Dialect Fidelity Score): Measures if the 'hypothesis' is linguistically
    closer to the 'reference' than to the 'source'.
    Formula: avg[ log(1+cos(hyp,ref)) - log(1+cos(hyp,std)) ]

-   TDR (Target Dialect Ratio): Uses an ensemble of dialect classifiers
    to predict the dialect of the 'hypothesis' text. This is the ratio
    of sentences correctly classified as the 'TARGET_DIALECT'.

Instructions for Use:
1.  Set up your Python environment with the required libraries 
    (pandas, numpy, torch, transformers, sentence-transformers, tqdm).
2.  Fill in all variables in the 'USER CONFIGURATION' section below,
    providing paths to your models and data files.
3.  Run the script.
"""

import os
import math
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import util
from collections import Counter
from tqdm.auto import tqdm

# (Optional) Uncomment to set GPU device ID.
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ======================= USER CONFIGURATION ======================= #

# 1. Path to the input CSV file.
#    Must contain 'source', 'reference', and 'hypothesis' columns.
INPUT_CSV_FILE = "path/to/your/my_results.csv"

# 2. Path to save the output CSV file with per-sentence scores.
OUTPUT_CSV_FILE = "path/to/your/my_results_with_scores.csv"

# 3. The target dialect label as it appears in the classifier (e.g., "제주도").
TARGET_DIALECT = "제주도"

# 4. Path or Hugging Face name of the sentence embedding model (for DFS).
EMB_MODEL_PATH = "path/to/your/embedding_model" # e.g., "BM-K/KoSimCSE-roberta-base"

# 5. List of paths to the *classifier* models (for TDR ensemble).
CLS_CKPT_PATHS = [
    "path/to/your/classifier/backbone_1",
    "path/to/your/classifier/backbone_2",
]

# --- Hyperparameters (usually no need to change) ---
CLS_MAX_LEN = 128
CLS_BATCH_SIZE = 64
EMB_BATCH_SIZE = 256
EPS = 1e-6
# =================== END OF USER CONFIGURATION =================== #


@torch.no_grad()
def _encode(texts, tokenizer, model, device):
    """
    Converts a list of texts into mean-pooled, normalized embeddings.
    """
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=CLS_MAX_LEN,
        return_tensors="pt"
    ).to(device)
    out = model(**enc)
    last_hidden = out.last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
    
    summed = (last_hidden * mask).sum(dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    emb = summed / denom
    emb = F.normalize(emb, p=2, dim=1)
    return emb

@torch.no_grad()
def compute_dfs_batch(std_texts, hyp_texts, ref_texts, tokenizer, emb_model, device):
    """
    Calculates the Dialect Fidelity Score (DFS) for a batch of texts.
    (hyp_texts = hypothesis, std_texts = source, ref_texts = reference)
    """
    if not all(isinstance(lst, list) for lst in [std_texts, hyp_texts, ref_texts]):
        print("[Error] Input texts must be lists.")
        return []
    
    n = len(hyp_texts)
    out = []
    
    for i in tqdm(range(0, n, EMB_BATCH_SIZE), desc="Calculating DFS (LogRatio)", leave=False):
        std_batch = std_texts[i:i+EMB_BATCH_SIZE]
        hyp_batch = hyp_texts[i:i+EMB_BATCH_SIZE]
        ref_batch = ref_texts[i:i+EMB_BATCH_SIZE]
        if not hyp_batch: 
            continue
        
        std_embs = _encode(std_batch, tokenizer, emb_model, device)
        hyp_embs = _encode(hyp_batch, tokenizer, emb_model, device)
        ref_embs = _encode(ref_batch, tokenizer, emb_model, device)
        
        # A = cos(hypothesis, source)
        A = torch.diagonal(util.cos_sim(hyp_embs, std_embs))
        # B = cos(hypothesis, reference)
        B = torch.diagonal(util.cos_sim(hyp_embs, ref_embs))
        
        # Formula: log(1 + B + EPS) - log(1 + A + EPS)
        numerator = torch.log(1 + B + EPS)
        denominator = torch.log(1 + A + EPS)
        log_ratio = numerator - denominator
        
        out.extend(log_ratio.cpu().tolist())
        del std_embs, hyp_embs, ref_embs
        torch.cuda.empty_cache()
        
    return out

@torch.no_grad()
def predict_dialect_ensemble_batch(texts, models, tokenizers, id2label, device):
    """
    Predicts dialect labels using an ensemble of models via soft voting.
    """
    if not texts: 
        return []
    
    all_model_probs = [[] for _ in models] # Stores probs for each model
    
    for idx, (model, tokenizer) in enumerate(tqdm(zip(models, tokenizers), total=len(models), desc="Classify per model", leave=False)):
        for i in range(0, len(texts), CLS_BATCH_SIZE):
            batch = texts[i:i+CLS_BATCH_SIZE]
            enc = tokenizer(
                batch, truncation=True, padding=True, 
                max_length=CLS_MAX_LEN, return_tensors="pt"
            ).to(device)
            
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            all_model_probs[idx].append(probs)

    # Concatenate batch results for each model
    final_probs_per_model = [torch.cat(probs_list, dim=0) for probs_list in all_model_probs]
    
    # Average probabilities across all models (soft voting)
    avg_probs = torch.stack(final_probs_per_model).mean(dim=0)
    
    # Get the final predicted label
    top_preds_indices = torch.argmax(avg_probs, dim=-1).tolist()
    final_labels = [id2label[t] for t in top_preds_indices]

    del all_model_probs, final_probs_per_model, avg_probs
    torch.cuda.empty_cache()
    
    return final_labels

def main():
    # --- 1. Setup Environment ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Processing file: {INPUT_CSV_FILE}")

    # --- 2. Load Models ---
    print(f"\n[INFO] Loading Embedding model (for DFS) from: {EMB_MODEL_PATH}")
    try:
        emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_PATH)
        emb_model = AutoModel.from_pretrained(EMB_MODEL_PATH).to(device).eval()
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model. Check path: {EMB_MODEL_PATH}")
        print(f"Details: {e}")
        return

    cls_models = []
    cls_tokenizers = []
    print(f"\n[INFO] Loading {len(CLS_CKPT_PATHS)} classifier models (for TDR)...")
    if not CLS_CKPT_PATHS:
        print("[ERROR] No classifier paths provided in CLS_CKPT_PATHS list.")
        return
        
    for i, path in enumerate(CLS_CKPT_PATHS):
        try:
            print(f"  -> Loading model {i+1}/{len(CLS_CKPT_PATHS)}: {path}")
            model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(path)
            cls_models.append(model)
            cls_tokenizers.append(tokenizer)
        except Exception as e:
            print(f"[ERROR] Failed to load classifier model. Check path: {path}")
            print(f"Details: {e}")
            return
    
    id2label = {int(k): v for k, v in cls_models[0].config.id2label.items()}
    print(f"[INFO] Classifier labels: {sorted(list(id2label.values()))}")
    
    if TARGET_DIALECT not in id2label.values():
        print(f"[ERROR] Target dialect '{TARGET_DIALECT}' not found in classifier labels!")
        print(f"Available labels: {list(id2label.values())}")
        return

    # --- 3. Load and Prepare Data ---
    print(f"\n[INFO] Loading and preparing data...")
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {INPUT_CSV_FILE}")
        return

    required_cols = ['source', 'reference', 'hypothesis']
    if not all(col in df.columns for col in required_cols):
        print(f"[ERROR] Input CSV must contain all of these columns: {required_cols}")
        return

    std_texts = df['source'].fillna("").astype(str).tolist()
    ref_texts = df['reference'].fillna("").astype(str).tolist()
    hyp_texts = df['hypothesis'].fillna("").astype(str).tolist()

    # --- 4. Calculate Metrics ---
    print(f"[INFO] Calculating metrics for {len(df)} sentences...")
    
    # Calculate DFS (Dialect Fidelity Score)
    dfs_scores = compute_dfs_batch(
        std_texts, hyp_texts, ref_texts, 
        tokenizer=emb_tokenizer, emb_model=emb_model, device=device
    )
    
    # Calculate TDR (Target Dialect Ratio)
    predicted_labels = predict_dialect_ensemble_batch(
        hyp_texts, cls_models, cls_tokenizers, id2label, device=device
    )
    
    # --- 5. Aggregate and Report ---
    print("\n[INFO] Aggregating results...")
    
    # Add per-sentence results to DataFrame
    df['dfs'] = dfs_scores
    df['predicted_label'] = predicted_labels
    df['is_target_dialect'] = (df['predicted_label'] == TARGET_DIALECT)

    # Calculate final average scores
    avg_dfs = np.nanmean(dfs_scores) if dfs_scores else 0.0
    tdr = df['is_target_dialect'].mean()
    
    # Save detailed output file
    try:
        output_dir = os.path.dirname(OUTPUT_CSV_FILE)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"\n[SUCCESS] Detailed results saved to: {OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save output file: {e}")

    # Print final summary
    print("\n--- Final Metrics Summary ---")
    print(f"Total Sentences:    {len(df)}")
    print(f"Average DFS:          {avg_dfs:.4f}")
    print(f"TDR ({TARGET_DIALECT}): {tdr:.4f} ({df['is_target_dialect'].sum()} / {len(df)})")
    print("-----------------------------\n")

if __name__ == "__main__":
    main()