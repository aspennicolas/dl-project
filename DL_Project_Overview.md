# 🏠 VibeCheck — Interior Design Style Matcher
### Deep Learning Final Project — IE University

---

## 📌 Project Summary

**VibeCheck** is an MVP that helps users answer one question before buying furniture or décor:

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
| Criteria | How VibeCheck delivers |
|---|---|
| ⏱️ Saves time | Eliminates hours of manual comparison |
| 💰 Reduces costs | Prevents costly returns before purchase |
| 📈 Increases revenue | Higher purchase confidence = higher conversion for retailers |

### Target Customers
- Individual homeowners furnishing or redecorating
- Interior designers validating client purchases
- Furniture retailers (IKEA, Wayfair, H&M Home) as a "shop the vibe" feature

### Market Context
IKEA already has IKEA Kreativ (room visualization) and a GPT-powered assistant — but **no tool decodes a user's aesthetic DNA from a raw mood board and validates purchases against it**. VibeCheck fills that gap.

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
├── Dense layer (512 neurons, ReLU activation)
├── Dropout (0.5 — overfitting prevention)
├── Batch Normalization
└── Output layer (6 neurons, Softmax activation)
        ↓
Style Classification:
[Minimalist, Industrial, Maximalist, Botanical, Dark & Moody, Rustic & Farmhouse]
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

### Primary: Places365
- **1.8 million** scene images including living rooms, bedrooms, kitchens, cafes
- Pre-labeled by place type — filter for interior categories
- Free to download: http://places2.csail.mit.edu/

### Secondary: IKEA Product Dataset (Kaggle)
- Furniture and décor product images with category labels
- Useful for product-side testing
- Available on Kaggle

### Style Categories & Labels
You will organize images into 6 folders, chosen for maximum visual distinctiveness:

| Category | Visual Signature |
|---|---|
| 🤍 **Minimalist** | White, clean lines, empty space, no clutter |
| 🏗️ **Industrial** | Concrete, metal, exposed pipes, dark tones |
| 👑 **Maximalist** | Bold colours, patterns, layered textures, lots of objects |
| 🌿 **Botanical** | Plants, natural materials, warm greens and browns |
| 🕯️ **Dark & Moody** | Deep colours, dramatic lighting, rich textures |
| 🪵 **Rustic & Farmhouse** | Wood, warm tones, vintage, cozy |

> **Note on styles not covered (e.g. Japandi, Moroccan, Zen):** The model outputs probability vectors across all 6 dimensions rather than hard labels. A Japanese-style room would naturally score high on Minimalist + Botanical — capturing it accurately as a combination. Regional and hybrid aesthetics are on the Version 2 roadmap.

```
/data
  /minimalist
  /industrial
  /maximalist
  /botanical
  /dark_moody
  /rustic_farmhouse
```

### Data Split
- Training: 70%
- Validation: 15%
- Test: 15%

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
| Explainability | SHAP or manual feature importance |
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
git clone https://github.com/yourusername/vibecheck.git
cd vibecheck

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Git Bash on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Recommended Project Structure

```
vibecheck/
├── data/
│   ├── minimalist/
│   ├── scandinavian/
│   ├── botanical/
│   ├── industrial/
│   ├── maximalist/
│   └── rustic/
├── models/
│   └── vibecheck_model.h5
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
| Solution | 2 min | VibeCheck — aesthetic DNA decoder + purchase validator |
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
- **B2B API** — furniture retailers integrate VibeCheck into their product pages
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

**Q: Why these 6 style categories specifically?**
> We chose categories with maximum visual distinctiveness so the CNN can learn clear decision boundaries. Minimalist (white, clean, empty) looks nothing like Maximalist (bold, layered, busy) or Industrial (concrete, metal, dark). Overlapping styles like Scandinavian were merged into Minimalist to avoid noisy boundaries. Regional aesthetics like Japandi naturally emerge as combinations — a Japanese room scores high on Minimalist + Botanical, which captures it accurately without needing a separate class.

**Q: Couldn't someone just use IKEA Kreativ?**
> IKEA Kreativ visualizes furniture in your room — it doesn't decode your aesthetic profile from a mood board or validate arbitrary product photos against your personal style. These are complementary tools solving different problems.
