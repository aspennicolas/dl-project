# Ambiance — Technical Architecture

## Overview

Ambiance is an interior design style matcher. The user uploads 3–5 mood board images → the model decodes their aesthetic profile → they upload a product photo → the app scores how well it matches. The core task is **19-class image classification** with a **cosine similarity** layer on top for matching.

---

## Layer 1 — Data Pipeline (`src/preprocess.py`)

```
data/raw/
  dataset_train/   ← ~14,900 images across 19 subfolders (1 per style)
  dataset_test/    ← ~3,700  images, same structure
```

Three `ImageDataGenerator` streams are built:

| Generator | Source | Augmentation | Purpose |
|---|---|---|---|
| `train_gen` | `dataset_train` (85%) | Flip, ±20° rotate, ±20% brightness, 10% zoom | Train the model |
| `val_gen` | `dataset_train` (15%) | None | Monitor overfitting each epoch |
| `test_gen` | `dataset_test` | None | Final unbiased evaluation |

**Preprocessing:** ResNet50's `preprocess_input` is applied — channel-mean subtraction (not `/255` scaling) to match what the frozen base was trained on. Images are resized to **224×224×3**. Labels are one-hot encoded (`class_mode="categorical"`).

---

## Layer 2 — Model Architecture (`src/model.py`)

```
Input: 224 × 224 × 3 image
         │
         ▼
┌─────────────────────────────────┐
│   ResNet50 (pretrained)         │
│                                 │
│  Layers 0–145  ── FROZEN        │  ~23.5M non-trainable params
│  Low-level features:            │  (edges, textures, shapes)
│  edges → textures → patterns    │
│                                 │
│  Layers 146–175 ── FINE-TUNED   │  Top 30 layers unlocked
│  High-level features:           │  to adapt to interior design
│  materials, room layouts        │
│                                 │
│  GlobalAveragePooling2D         │  (7,7,2048) → (2048,)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Custom ANN Head (trainable)   │
│                                 │
│  Dense(512, linear)             │  ~1.05M params
│  BatchNormalization             │  normalises pre-activation values
│  Activation('relu')             │  non-linearity
│  Dropout(0.5)                   │  drops 50% of neurons → anti-overfit
│                                 │
│  Dense(19, softmax)             │  ~9,747 params
└─────────────────────────────────┘
         │
         ▼
Output: 19-dim probability vector
  e.g. [0.60 Scandinavian, 0.20 Modern, 0.10 Industrial, ...]
```

**Key design decisions:**
- **GlobalAveragePooling2D** over Flatten: avoids 100K+ parameters that would come from `7×7×2048` flattening
- **BatchNorm before activation**: normalises inputs *into* the ReLU, not after — prevents BN running-stats corruption from Dropout
- **Adam at lr=1e-4**: 10× lower than Adam's default to avoid destroying pretrained weights during fine-tuning
- **Loss:** `categorical_crossentropy` (standard for multi-class softmax output)

---

## Layer 3 — Prediction & Matching Logic (`src/predictor.py`)

This is where the app's unique functionality lives. The model outputs style vectors; this module combines them into a mood board and scores products against it.

```
Step 1 — Per-image inference
  PIL image
    → convert RGB, resize 224×224
    → preprocess_input()
    → model.predict()           → shape (19,) probability vector

Step 2 — Mood board aggregation
  3–5 style vectors
    → np.mean(vectors, axis=0)  → single (19,) composite profile
  e.g. "60% Scandinavian, 20% Modern, 10% Industrial"

Step 3 — Product inference
  Product photo → same pipeline → (19,) product vector

Step 4 — Cosine similarity match score
  similarity = dot(moodboard, product) / (||moodboard|| × ||product||)
  score = round(similarity × 100, 1)   → 0–100%

Step 5 — Top-N style labels
  np.argsort(vector)[-n:][::-1]  → top N class indices → human-readable names
```

**Why cosine similarity?** It measures the *angle* between two probability distributions in 19D space, not their magnitude — so a softmax vector with a confident single class and one spread across many classes still compare meaningfully.

---

## Layer 4 — Frontend (`app.py`)

Built with **Streamlit**. The entire UI is a single-page Python script; no JavaScript or separate backend.

```
app.py
  │
  ├── @st.cache_resource  → loads model once, reuses across re-renders
  │
  ├── Step 01: Mood Board
  │     st.file_uploader (3–5 images)
  │     → get_moodboard_vector()
  │     → top_styles(n=5) → HTML bar chart (rendered via st.markdown)
  │     → dominant style callout
  │
  └── Step 02: Product Check  (only renders if moodboard_vector exists)
        st.file_uploader (1 image)
        → get_style_vector()
        → match_score()  → green / amber / red score card
        → top_styles(n=3) on the product → style pill tags
```

**Score thresholds:**

| Score | Verdict | Color |
|---|---|---|
| ≥ 80% | Strong Match | Green `#5C8C6A` |
| 65–79% | Good Match | Amber `#B8924A` |
| < 65% | Poor Match | Red `#A85C5C` |

Custom CSS injects Google Fonts (Cormorant Garamond + Inter) and overrides Streamlit's default chrome.

---

## Full Data/Control Flow

```
[User] uploads mood board images (3–5)
         │
         ▼
[app.py] PIL.Image.open() for each file
         │
         ▼
[predictor.py] get_moodboard_vector()
  ├── _preprocess() × N images
  ├── model.predict() × N images   ← TF/Keras inference
  └── np.mean() → moodboard_vector (19,)
         │
         ▼
[app.py] renders style profile bar chart + dominant style

[User] uploads product image
         │
         ▼
[predictor.py] get_style_vector() → product_vector (19,)
         │
         ▼
[predictor.py] match_score(moodboard_vector, product_vector)
  └── cosine_similarity × 100  → float 0–100
         │
         ▼
[app.py] renders score card (green/amber/red) + style comparison
```

---

## Evaluation Summary

| Metric | Value |
|---|---|
| Top-1 Accuracy | 34.1% |
| Top-5 Accuracy | 73.1% |
| Macro F1 | 0.341 |
| Best class | Industrial (F1 = 0.447) |
| Worst class | French Country (F1 = 0.223) |
| Base model | ResNet50, ImageNet pretrained |
| Dataset | ~18,600 images, 19 classes |

Top-5 accuracy (73.1%) is the headline metric because the app uses the full probability vector for cosine similarity — the *exact* top-1 label matters less than whether the vector's shape correctly captures the style family.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep learning | TensorFlow / Keras |
| Pretrained base | ResNet50 (Keras Applications) |
| Image handling | PIL / NumPy |
| Training notebooks | Jupyter / Google Colab |
| Frontend | Streamlit |
| Similarity metric | NumPy cosine similarity |
| Model serialisation | `.h5` (Keras HDF5 format) |
