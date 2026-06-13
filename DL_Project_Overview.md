# 🏠 Ambiance — Interior Design Style Matcher
### Deep Learning Final Project — IE University

---

## 📌 Project Summary

**Ambiance** is an MVP that helps users answer one question before buying furniture or décor:

> *"Does this item actually match my home aesthetic?"*

The user uploads 3–5 Pinterest inspiration images to define their personal style profile. The model decodes their aesthetic (e.g. "60% Minimalist, 30% Scandinavian, 10% Botanical") and then checks any product photo against that profile — returning a match score and explanation.

---

## 🎯 Business Problem & Value Proposition

### The Problem
People spend hours comparing furniture to their inspiration boards — and still get it wrong. The average household returns **15–30% of furniture purchases** due to style mismatch, costing retailers and consumers billions annually.

### The Solution
An AI-powered style compatibility checker that:
- Decodes your aesthetic from your mood board automatically
- Validates any product photo against your personal style profile instantly
- Tells you *why* something matches or doesn't

### Value Proposition (maps directly to rubric)
| Criteria | How Ambiance delivers |
|---|---|
| ⏱️ Saves time | Eliminates hours of manual comparison |
| 💰 Reduces costs | Prevents costly returns before purchase |
| 📈 Increases revenue | Higher purchase confidence = higher conversion for retailers |

### Target Customers
- Individual homeowners furnishing or redecorating
- Interior designers validating client purchases
- Furniture retailers (IKEA, Wayfair, H&M Home) as a "shop the vibe" feature

### Market Context
IKEA already has IKEA Kreativ (room visualization) and a GPT-powered assistant — but **no tool decodes a user's aesthetic DNA from a raw mood board and validates purchases against it**. Ambiance fills that gap.

---

## 🧠 Technical Architecture

### Model: Transfer Learning CNN (ResNet50)
This is a **Convolutional Neural Network** — the correct architecture for image classification tasks.

#### Why CNN and not ANN or RNN?
- Input is **image data** → CNNs are specifically designed for spatial/visual pattern recognition
- ANN would ignore spatial relationships between pixels
- RNN is for sequential data (time series, text) — not applicable here

#### Why Transfer Learning (pretrained ResNet50)?
- ResNet50 was pretrained on **ImageNet (1.2M images)** — it already knows edges, textures, shapes, and patterns
- We **freeze the base layers** and only train the final custom ANN head
- This means we don't need millions of images or weeks of GPU time
- Standard industry practice, fully legitimate to present

#### Full Architecture
```
Input Image (224x224x3)
        ↓
ResNet50 Base (FROZEN — pretrained on ImageNet)
├── Convolutional layers — edge/texture/pattern detection
├── Residual blocks — deep feature extraction
└── Global Average Pooling — 2048-dim feature vector
        ↓
Custom ANN Head (TRAINABLE — fine-tuned on our data)
├── Dense layer (512 neurons, linear)
├── Batch Normalization
├── ReLU activation
├── Dropout (0.5 — overfitting prevention)
└── Output layer (19 neurons, Softmax activation)
        ↓
Style Classification:
[Asian, Coastal, Contemporary, Craftsman, Eclectic, Farmhouse, French Country,
 Industrial, Mediterranean, Mid-Century Modern, Modern, Rustic, Scandinavian,
 Shabby Chic, Southwestern, Traditional, Transitional, Tropical, Victorian]
```

#### Prediction Logic (Mood Board → Match Score)
```
Step 1: Run CNN on each of 3–5 inspo images
        → get style probability vector per image
Step 2: Average all vectors → composite aesthetic profile
        e.g. [0.60, 0.10, 0.05, 0.20, 0.05, 0.00]
        = "60% Minimalist, 20% Botanical, 10% Industrial"
Step 3: Run CNN on product image → product style vector
Step 4: Compute cosine similarity between profile and product
        → match score 0–100%
Step 5: Generate explanation based on dominant style categories
```

---

## 📦 Dataset

### Source: Kaggle Interior Design Style Dataset
- **~18,600 images** across 19 interior design style categories
- Already split into train (~14,900 images) and test (~3,700 images)
- Pre-labeled by style — no manual organisation required

### Style Categories (19 classes, ~780 train / ~195 test images each)

| Category | Visual Signature |
|---|---|
| **Asian** | Minimalist forms, natural materials, zen balance |
| **Coastal** | Light blues, whites, natural textures, beach-inspired |
| **Contemporary** | Clean lines, neutral palette, current trends |
| **Craftsman** | Handcrafted woodwork, earthy tones, structural details |
| **Eclectic** | Bold mix of styles, colours, and eras |
| **Farmhouse** | Shiplap, neutrals, vintage accents, cozy |
| **French Country** | Ornate details, warm tones, rustic elegance |
| **Industrial** | Concrete, metal, exposed pipes, dark tones |
| **Mediterranean** | Terracotta, arches, warm colours, tile patterns |
| **Mid-Century Modern** | Organic shapes, bold accents, functional furniture |
| **Modern** | Minimal, clean geometry, monochrome palette |
| **Rustic** | Raw wood, stone, warm textures, natural materials |
| **Scandinavian** | White, light wood, functional simplicity |
| **Shabby Chic** | Distressed finishes, pastels, vintage florals |
| **Southwestern** | Adobe tones, bold patterns, desert-inspired |
| **Traditional** | Formal symmetry, rich woods, classic details |
| **Transitional** | Blend of traditional and contemporary |
| **Tropical** | Lush greens, natural fibres, open-air feel |
| **Victorian** | Ornate, dark woods, layered patterns, antiques |

```
data/
  raw/
    dataset_train/
      asian/  coastal/  contemporary/  craftsman/  eclectic/
      farmhouse/  french-country/  industrial/  mediterranean/
      mid-century-modern/  modern/  rustic/  scandinavian/
      shabby-chic-style/  southwestern/  traditional/
      transitional/  tropical/  victorian/
    dataset_test/
      (same 19 subfolders)
    test_labels.csv
```

### Data Split
- Training: dataset_train (~80% of total)
- Test: dataset_test (~20% of total)
- Validation: split from training set during model training (e.g. 15% of train)

---

## 🖥️ Frontend

Built with **Streamlit** — a Python library that turns a model into a web app.

### User Flow
1. Upload 3–5 Pinterest inspo images
2. See your decoded aesthetic profile with percentages + bar chart
3. Upload a product/décor photo
4. See match score (0–100%) + ✅ or ❌ + style breakdown explanation

### Why Streamlit?
- Pure Python — no frontend experience needed
- Runs locally, no deployment required for demo
- Clean, professional UI out of the box
- Handles image uploads natively

---

## 📊 Evaluation Metrics

| Metric | Why |
|---|---|
| **Accuracy** | Overall classification correctness |
| **F1 Score (per class)** | Handles class imbalance across style categories |
| **Confusion Matrix** | Shows which styles get confused with each other |
| **Training vs Validation Loss curves** | Evidence of overfitting prevention |

---

## 🛡️ Overfitting Prevention

- **Dropout (0.5)** on the dense layer
- **Data Augmentation** — random flips, rotations, brightness changes
- **Early stopping** — stops training when validation loss stops improving
- **Batch Normalization** — stabilizes training

---

## 💻 Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Pretrained Model | ResNet50 (Keras Applications) |
| Data Processing | NumPy, Pandas, PIL |
| Visualization | Matplotlib, Seaborn |
| Explainability | Confusion matrix + per-class F1 + misclassified image analysis |
| Frontend | Streamlit |
| Version Control | Git + GitHub |
| IDE | VS Code |

---

## 💻 Your Setup (ASUS Zephyrus G14)

### GPU Activation (NVIDIA GPU — do this first)
Your Zephyrus G14 has an NVIDIA GPU. To use it for training:

**Step 1: Check your GPU**
```bash
nvidia-smi
```

**Step 2: Install CUDA Toolkit**
- Download from: https://developer.nvidia.com/cuda-downloads
- Select: Windows → x86_64 → your version → exe (local)
- Install CUDA 11.8 or 12.x (check TensorFlow compatibility)

**Step 3: Install cuDNN**
- Download from: https://developer.nvidia.com/cudnn
- Requires free NVIDIA account
- Follow installation guide for your CUDA version

**Step 4: Install TensorFlow with GPU support**
```bash
pip install tensorflow[and-cuda]
```

**Step 5: Verify GPU is detected**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should print something like [PhysicalDevice(name='/physical_device:GPU:0')]
```

### VS Code Setup
- Install Python extension
- Install Jupyter extension (useful for exploratory work)
- Set interpreter to your project virtual environment

### Git Bash Setup
```bash
# Clone your repo
git clone https://github.com/yourusername/ambiance.git
cd ambiance

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Git Bash on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Recommended Project Structure

```
ambiance/
├── data/
│   ├── raw/
│   │   ├── dataset_train/   ← 19 style subfolders, ~780 images each
│   │   ├── dataset_test/    ← 19 style subfolders, ~195 images each
│   │   └── test_labels.csv
│   └── processed/
├── models/
│   └── ambiance_model.h5
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── predict.py
│   └── utils.py
├── app.py              ← Streamlit frontend
├── requirements.txt
└── README.md
```

---

## 🗂️ Build Phases

| Phase | Task | Owner |
|---|---|---|
| 1 | Data download & organization | Person 1 |
| 2 | Preprocessing pipeline | Person 1 |
| 3 | Model building (ResNet50 + ANN head) | Person 2 |
| 4 | Training & evaluation | Person 2 |
| 5 | Prediction logic & match scoring | Person 3 |
| 6 | Streamlit frontend | Person 4 |
| 7 | GitHub repo + README | Everyone |
| 8 | Presentation deck | Everyone |
| 9 | Demo prep & practice | Everyone |

---

## 🎤 Presentation Structure (15 min)

| Section | Time | Content |
|---|---|---|
| Hook | 1 min | "Have you ever bought furniture that looked perfect online but felt completely wrong at home?" |
| Problem | 2 min | Furniture returns, style mismatch, $50B problem |
| Solution | 2 min | Ambiance — aesthetic DNA decoder + purchase validator |
| Market context | 1 min | IKEA Kreativ exists but doesn't do this |
| Technical deep dive | 3 min | ResNet50 + transfer learning + cosine similarity |
| Evaluation | 1 min | Metrics, overfitting prevention |
| Live demo | 3 min | Upload mood board → decode profile → check product |
| Business model & future | 1 min | B2B API for retailers, Siamese network next iteration |
| Q&A | 1 min | |

---

## 🔮 Future Roadmap (mention in presentation)

- **Siamese Network** — end-to-end similarity learning instead of classification + cosine similarity
- **Pinterest API integration** — import boards directly instead of manual upload
- **B2B API** — furniture retailers integrate Ambiance into their product pages
- **"Shop the match"** — when something doesn't match, suggest items that do
- **Room photo input** — photograph your actual room instead of uploading inspo images

---

## ❓ Anticipated Q&A

**Q: Why CNN and not a simpler model like Random Forest?**
> CNNs learn spatial hierarchies in images — edges → textures → patterns → styles. A Random Forest treats each pixel independently and loses all spatial context. For image data, CNN is the correct choice.

**Q: Why ResNet50 specifically?**
> ResNet50 uses residual connections that solve the vanishing gradient problem in deep networks. It's proven on ImageNet and widely used in transfer learning. Alternative: VGG16 (simpler but slower), EfficientNet (more efficient but harder to explain).

**Q: Why not train from scratch?**
> We have ~10K images per class. Training ResNet50 from scratch requires millions of images and weeks of GPU time. Transfer learning lets us leverage features already learned from 1.2M ImageNet images and fine-tune only the task-specific layers. This is standard industry practice.

**Q: How do you prevent overfitting?**
> Dropout (0.5) randomly deactivates neurons during training. Data augmentation artificially expands training data. Early stopping halts training when validation loss plateaus. Batch normalization stabilizes gradient updates.

**Q: Why these 19 style categories?**
> The dataset comes pre-labeled with 19 real-world interior design styles used by professional designers and home decor platforms. Using all 19 gives the model more granular style understanding and makes the probability vectors more informative — a room can score across 19 dimensions rather than being forced into 6 broad buckets. The categories have enough visual distinctiveness for the CNN to learn meaningful decision boundaries (e.g. Victorian vs Scandinavian vs Industrial are visually very different), and the ~780 training images per class is sufficient for fine-tuning a pretrained ResNet50.

**Q: Couldn't someone just use IKEA Kreativ?**
> IKEA Kreativ visualizes furniture in your room — it doesn't decode your aesthetic profile from a mood board or validate arbitrary product photos against your personal style. These are complementary tools solving different problems.
