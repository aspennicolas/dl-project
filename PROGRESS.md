# Project Progress

> Rubric weights: Business (20%) | Technical Depth (25%) | MVP Integration (25%) | Presentation (20%) | Live Demo (10%)
> Exceptional on Technical requires: justified hyperparameter choices, justified preprocessing steps, excellent eval metrics, overfitting mitigation.
> Exceptional on MVP Integration requires: seamless real-time predictions, intuitive professional UI.

---

## Phase 1 — Project Setup & Dataset Acquisition ✅
- Decided on Kaggle Interior Design Style dataset (19 classes, ~18,600 images)
- Defined project structure (`data/raw/dataset_train`, `data/raw/dataset_test`)
- Set up `.gitignore`, `requirements.txt`, virtual environment
- Removed web scraping approach (icrawler, clean_images.py)
- Updated `DL_Project_Overview.md` to reflect 19 classes and actual dataset

---

## Phase 2 — Data Preprocessing ✅
**Goal:** Get the raw images into a format the model can consume.
**Rubric relevance:** Technical Depth — "flawless data pipeline" and "justified preprocessing steps" are explicitly called out for Exceptional.

Tasks:
- [x] Resize all images to 224×224 (ResNet50 input size — justify why)
- [x] Normalize pixel values using `preprocess_input` (ResNet50 channel-mean subtraction) — **not** `rescale=1./255` (see bug fix below)
- [x] Carve out a 15% validation split from `dataset_train`
- [x] Set up data augmentation pipeline (horizontal flip, ±20° rotation, ±20% brightness, ±10% zoom) — each justified in code comments
- [x] Verify class balance across all 19 categories — class balance chart + class weights added to training notebooks
- [x] Built ImageDataGenerator pipeline (`src/preprocess.py`)
- [x] Data exploration notebook (`notebooks/01_data_exploration.ipynb`)

### Decision Log

- **`preprocess_input` (ResNet50)** — subtracts the ImageNet per-channel means (R=123.68, G=116.78, B=103.94) from each pixel, producing values roughly in `[-128, +151]`. This must match exactly how the ResNet50 weights were produced — using `rescale=1./255` instead gives the frozen base inputs it was never trained on, causing it to produce garbage features and keeping accuracy at random-chance levels (~5%). Fixed from original `rescale=1./255` after diagnosing near-random training accuracy.
- **`horizontal_flip=True`** — mirrors images left-right. A Scandinavian living room is still Scandinavian when flipped, so this is safe and doubles effective data variety.
- **`rotation_range=20`** — rotates up to ±20°, simulating a photographer standing at a slight angle.
- **`brightness_range=[0.8, 1.2]`** — darkens or brightens ±20%, accounting for different lighting conditions between photos.
- **`zoom_range=0.1`** — zooms in or out ±10%, simulating different shooting distances.
- **`validation_split=0.15`** — Keras holds out 15% of `dataset_train` as validation. `train_gen` and `val_gen` use the **same seed** — if the seeds differed, the same image could appear in both splits, which is data leakage.
- **`shuffle=False` on val/test** — evaluation order stays fixed so predictions align with true labels when building the confusion matrix later.
- **No augmentation on val/test** — evaluation must happen on clean, unmodified images so the metrics reflect true real-world performance.

---

## Phase 3 — Model Building ✅
**Goal:** Build the ResNet50 + custom head architecture.
**Rubric relevance:** Technical Depth — strong architectural justification (CNN), justified hyperparameter choices (layer sizes, dropout rate, learning rate).

Tasks:
- [x] Load pretrained ResNet50 (frozen, ImageNet weights) — justify transfer learning choice
- [x] Add custom ANN head: Dense(512) → BatchNorm → ReLU → Dropout(0.5) → Dense(19, Softmax)
- [x] Justify each hyperparameter: 512 neurons, 0.5 dropout, Adam optimizer, learning rate
- [x] Compile: Adam optimizer, categorical crossentropy loss, accuracy metric

### Decision Log

- **ResNet50 (not VGG16 or EfficientNet)** — ResNet50 uses residual connections that solve the vanishing gradient problem in deep networks. Gradients flow directly through shortcut paths, so early layers still learn even at 50 layers deep. VGG16 is simpler but slower and lacks residuals. EfficientNet is more parameter-efficient but harder to explain to a panel.
- **`include_top=False`** — removes ResNet50's original 1000-class ImageNet output layer so we can attach our own 19-class head.
- **`trainable=False` (frozen base)** — locks all ~25M ResNet50 weights. We have ~780 images per class — far too few to retrain a 50-layer CNN from scratch without severe overfitting. Freezing preserves ImageNet features (edges, textures, patterns) and lets us train only the small custom head.
- **`GlobalAveragePooling2D`** — compresses the (7, 7, 2048) ResNet output to a (2048,) vector by averaging each channel. Alternative is Flatten (7×7×2048 = 100,352 inputs — massive parameter count, more overfitting risk). GAP has zero trainable parameters and is more robust.
- **`Dense(512, ReLU)`** — learns which combinations of the 2048 ResNet features predict each interior style. 512 sits between the 2048 input and 19 output — large enough for complexity, small enough to train quickly. ReLU adds non-linearity; without it this layer is just a linear transformation.
- **`Dropout(0.5)`** — randomly deactivates 50% of neurons each training step. Forces redundant learning across neurons — the model can't lean on any single one. 0.5 is the standard rate for fully-connected layers. Automatically disabled during inference.
- **`BatchNormalization`** — normalizes activations within each mini-batch (subtract mean, divide by std), then applies learnable scale/shift. Stabilizes training, reduces sensitivity to learning rate, mild regularization. Placed **before** activation and Dropout: `Dense (linear) → BatchNorm → ReLU → Dropout`. Placing BN after Dropout would corrupt its running statistics because Dropout zeros 50% of values during training but not at inference, creating a train/test mismatch in the normalisation.
- **`Dense(19, Softmax)`** — 19 output neurons, one per style class. Softmax converts raw scores to a probability distribution summing to 1.0 — e.g. [0.60, 0.10, 0.05, ...]. This vector is also what feeds the cosine similarity calculation in Phase 5.
- **`Adam(learning_rate=1e-4)`** — Adam adapts the learning rate per parameter based on gradient history. Rate is set to 1e-4 (10× lower than Adam's default 1e-3) because we're fine-tuning on pretrained weights — smaller updates preserve what ResNet50 already learned while still fitting the head to our style classes.
- **`categorical_crossentropy`** — standard loss for multi-class classification with one-hot labels (which is what `class_mode='categorical'` produces). Penalizes confident wrong predictions more heavily than uncertain ones.

---

## Phase 4 — Training & Evaluation 🔄 In Progress
**Goal:** Train the model and measure performance.
**Rubric relevance:** Technical Depth — "excellent evaluation metrics and overfitting mitigation" required for Exceptional.

Tasks:
- [x] Build training notebook — `notebooks/02_model_training.ipynb` (local) + `notebooks/02_model_training_colab.ipynb` (Colab)
- [x] Two-phase training: Phase 4a (frozen base) + Phase 4b (fine-tune top 30 ResNet50 layers)
- [x] Callbacks: EarlyStopping (patience=5), ModelCheckpoint (save best only), ReduceLROnPlateau (factor=0.5)
- [x] Class weights computed and passed to model.fit() to handle any class imbalance
- [x] Training history saved to JSON so curves can be re-plotted without retraining
- [x] Training curves plot (loss + accuracy, both phases, phase boundary line)
- [x] Built evaluation notebook (`notebooks/03_evaluation.ipynb`) — overall accuracy, top-5 accuracy, per-class F1 bar chart, normalised confusion matrix heatmap, sample misclassified images, training curves reload
- [x] Phase 4a complete — 20 epochs, best val accuracy 31.9% at epoch 15 (early stopping triggered). Checkpoint saved to Drive.
- [x] Phase 4b started — fine-tuning top 30 ResNet50 layers at lr=1e-5. Last checkpoint at epoch 5 (val_loss 2.2736, val_accuracy 32.7%). Resuming next session.
- [x] Phase 4b restarted from checkpoint — created `02_model_training_colab2.ipynb` which loads `ambiance_model.h5` directly (skips Phase 4a entirely). Removed redundant model-load cell, removed 4a cell (was a trap that would have overwritten checkpoint). Fine-tuning resumes from the partially fine-tuned checkpoint, saving best weights back to `ambiance_model.h5` via `save_best_only=True`.
- [x] Phase 4b complete — EarlyStopping triggered at epoch 6 (patience=5, no val_loss improvement after epoch 1). Best val accuracy: 33.35%, val_loss: 2.2579. Weights restored to epoch 1. Overfitting observed: training accuracy climbed to 62% while val accuracy stayed flat at ~33%. Model saved to Drive.
- [x] Run 03_evaluation.ipynb on the final model — results: Top-1 34.1%, Top-5 73.1%, Macro F1 0.341. Best class: Industrial (0.447), Worst: French-Country (0.223). Artefacts saved to Drive (classification_report.txt, confusion_matrix.png, f1_per_class.png, misclassified_samples.png).
- [x] Save best model weights to `models/ambiance_model.h5`

### Decision Log

- **Two-phase training** — Phase 4a trains only the custom head (frozen ResNet50 base, lr=1e-4). Phase 4b unfreezes the top 30 ResNet50 layers and fine-tunes at lr=1e-5. Fine-tuning adapts ResNet's deeper layers to interior design patterns (colour palettes, materials, structural elements) — typically pushes accuracy from ~70% to ~80%+.
- **Why top 30 layers only** — Early ResNet layers detect universal low-level features (edges, gradients) that transfer perfectly from ImageNet. Only the later, more abstract layers need domain adaptation. Unfreezing all 175 layers with our dataset size would cause overfitting.
- **Fine-tuning lr=1e-5** — 10× lower than head training rate. The pretrained weights are already good — we nudge them, not overwrite them.
- **`class_weight='balanced'`** — `sklearn.utils.class_weight.compute_class_weight` sets weight = total_samples / (num_classes × class_count). Rare classes get upweighted so the loss function penalises mistakes on them more heavily.
- **`build_model()` returns `(model, base_model)`** — ResNet50's layers are flattened into the combined model graph, so `model.layers[0]` is the `InputLayer`, not ResNet50. Returning `base_model` as a separate reference is required for fine-tuning — the layer objects are shared, so `base_model.layers[-30:]` correctly targets the right layers inside the combined model.

---

## Phase 5 — Prediction Logic ✅
**Goal:** Turn model outputs into a mood board match score.
**Rubric relevance:** Technical Depth + MVP Integration — the cosine similarity pipeline must run in real-time for the frontend.

Tasks:
- [x] Build mood board pipeline: run CNN on 3–5 images → average probability vectors
- [x] Build product check: run CNN on product image → style vector
- [x] Compute cosine similarity → match score (0–100%)
- [x] Generate text explanation based on dominant style categories
- [x] Ensure inference is fast enough for real-time frontend use

---

## Phase 6 — Streamlit Frontend ✅
**Goal:** Build the user-facing web app.
**Rubric relevance:** MVP Integration (25%) — seamless, real-time, intuitive UI is the biggest technical chunk of the grade alongside model quality.

Tasks:
- [x] Upload 3–5 inspo images → display decoded aesthetic profile (bar chart)
- [x] Upload product image → display match score + ✅/❌ + explanation
- [x] Ensure real-time predictions (no pre-computed results)
- [x] Polish UI — professional look, no friction, clear user flow

---

## Phase 7 — Presentation Prep ⬜ Not Started
**Goal:** Prepare slides and live demo for the 15-minute final presentation.
**Rubric relevance:** Presentation (20%) + Live Demo (10%) = 30% of grade.

Tasks:
- [ ] Finalise presentation deck: hook → problem → solution → tech → eval → demo → business
- [ ] Quantify business metrics (ROI, cost reduction %, efficiency gains) — required for Exceptional on Business pillar
- [ ] Assign speaking sections to all team members (everyone must speak)
- [ ] Rehearse to exactly 15 minutes including live demo and Q&A buffer
- [ ] Stress-test the live demo — must work under pressure

---

## Submission Checklist ⬜
- [x] GitHub repo with clean, documented backend + frontend code
- [x] `README.md` with instructions on how to run the project
- [ ] Final presentation slides (PDF format)
