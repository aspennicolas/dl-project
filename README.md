# Ambiance — Interior Style Matcher

A deep learning app that analyses your interior design aesthetic from inspiration images and scores how well a product matches your style.

Built with ResNet50 (transfer learning + fine-tuning) trained on 19 interior design style classes.

---

## How it works

1. Upload 3–5 inspiration images of rooms you love
2. The model analyses each image and builds your **style profile** (a probability vector across 19 styles)
3. Upload a product image (furniture, decor, etc.)
4. The app computes **cosine similarity** between the product and your style profile → outputs a 0–100% match score

---

## Project structure

```
ambiance/
├── app.py                  ← Streamlit frontend (run this)
├── src/
│   ├── model.py            ← ResNet50 + custom head architecture
│   ├── predictor.py        ← Style vectors, moodboard averaging, match score
│   └── preprocess.py       ← Data preprocessing pipeline
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training_colab2.ipynb   ← Training (run on Google Colab)
│   └── 03_evaluation.ipynb              ← Evaluation metrics
├── models/
│   └── ambiance_model.h5   ← Trained model weights (download separately)
├── PROGRESS.md             ← Full project progress tracker
├── EVALUATION_RESULTS.md   ← Model evaluation results and analysis
└── CNN_EXPLAINED.md        ← Architecture explanation
```

---

## Setup

### 1. Clone the repo
```bash
git clone <repo-url>
cd dl-project
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 3. Download the trained model
Download `ambiance_model.h5` from Google Drive and place it in the `models/` folder:
```
models/ambiance_model.h5
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Model performance

| Metric | Score |
|---|---|
| Top-1 Accuracy | 34.1% |
| Top-5 Accuracy | 73.1% |
| Macro F1 | 0.341 |
| Classes | 19 interior design styles |
| Base model | ResNet50 (ImageNet pretrained) |

Top-5 accuracy of 73.1% means the correct style appears in the model's top 5 predictions 3 out of 4 times — strong performance for 19 visually similar style categories.

---

## Style classes

Asian, Coastal, Contemporary, Craftsman, Eclectic, Farmhouse, French Country, Industrial, Mediterranean, Mid-Century Modern, Modern, Rustic, Scandinavian, Shabby Chic, Southwestern, Traditional, Transitional, Tropical, Victorian
