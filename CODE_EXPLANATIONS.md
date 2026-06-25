# Ambiance — Code Explanations for Teacher Q&A

A full breakdown of every file and notebook: what each line does, why it was written that way, and how to answer questions about it.

---

## How the Files Relate to Each Other

```
preprocess.py  →  "how do we load and prepare images?"
model.py       →  "what is the neural network?"
predictor.py   →  "how do we use the trained model in the app?"
```

| File | Used by | When |
|---|---|---|
| `preprocess.py` | Training nb, Exploration nb, Evaluation nb | Training time |
| `model.py` | Training notebook | Training time |
| `predictor.py` | `app.py` (Streamlit frontend) | Inference time (live app) |

```
preprocess.py ──→ training notebook (loads data)
model.py ───────→ training notebook (builds model)
                         ↓
                   trains & saves
                   ambiance_model.h5
                         ↓
predictor.py ───→ app.py (loads .h5, runs predictions live)
```

`preprocess.py` and `model.py` are training-time files. `predictor.py` is the inference-time file that makes the Streamlit app work. The training notebook never touches `predictor.py`.

---

---

# `preprocess.py` — The Data Pipeline

> **What this file does:** This is the data loading and preparation layer for the entire project. It defines how raw images from disk get transformed into the batches of numbers the model trains on. It creates three data generators — one for training (with augmentation), one for validation (clean images), and one for the test set (clean images). Every other file that needs data imports from here, making it the single source of truth for image preprocessing. If a preprocessing decision ever needs to change, you change it once here and everything updates automatically.

---

## Path Setup

*Builds absolute paths to the data folders so the script works correctly no matter which machine or directory it is run from.*

```python
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "dataset_train")
```

**What it does:** Finds the project root dynamically regardless of where you run the script from. `__file__` is the path to `preprocess.py` itself. `dirname` goes up one folder (out of `src/`), then `dirname` again goes up to the project root. Then it navigates down into the data folder.

**Why not hardcode the path?** A hardcoded path like `C:/Users/yourname/ambiance/data/...` breaks on any other machine. This approach works on anyone's computer, including a teacher's if they clone the repo.

---

## Hyperparameters

```python
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT  = 0.15
SEED       = 42
```

**`IMG_SIZE = (224, 224)`** — ResNet50's architecture was specifically designed around 224×224 inputs. Feed it a different size and the layer dimensions break.

**`BATCH_SIZE = 32`** — 32 images are loaded, forward-passed, and used to compute one gradient update before weights change. Smaller batches = noisier gradients but faster updates. Larger = smoother but more memory. 32 is the standard starting point.

**`VAL_SPLIT = 0.15`** — 15% of training images held out for validation. Enough to get reliable metrics without wasting too much training data.

**`SEED = 42`** — Critical. Both `train_gen` and `val_gen` use the same seed so Keras assigns the same files to each split every run. Without it, the split is random each time — you'd get different results on every run and can't reproduce experiments.

---

## Training Generator

```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=20,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    validation_split=VAL_SPLIT
)
```

**`preprocessing_function=preprocess_input`** — The most important line in the file. ResNet50 was originally trained in a framework called Caffe, which subtracted the ImageNet per-channel means (R=123.68, G=116.78, B=103.94) from every pixel instead of dividing by 255. `preprocess_input` replicates that exact transformation.

If you used `rescale=1./255` instead, the frozen ResNet50 layers receive inputs in a completely different scale than they were trained on and produce garbage features — accuracy stays near random chance (~5%). This was a real bug caught and fixed during development.

**Why augmentation only on training data?** Augmentation artificially expands training variety so the model can't memorise specific photos. Validation and test must use clean images — you need to measure real-world performance, not performance on modified versions.

**Each augmentation parameter justified:**

| Parameter | Value | Why |
|---|---|---|
| `horizontal_flip` | True | A Scandinavian room flipped left-right is still Scandinavian. Free data variety. |
| `rotation_range` | 20 | Simulates photographer standing at a slight angle. ±20° is subtle and realistic. |
| `brightness_range` | [0.8, 1.2] | Accounts for different lighting conditions between photos. ±20% is realistic. |
| `zoom_range` | 0.1 | Simulates different shooting distances. ±10% is barely noticeable. |

---

## The Seed Problem — Why It Matters

```python
# train_gen — uses train_datagen, seed=42
train_datagen.flow_from_directory(..., subset="training", seed=SEED)

# val_gen — same datagen, same seed
train_datagen.flow_from_directory(..., subset="validation", seed=SEED)
```

**Why does `val_gen` use `train_datagen` and not `eval_datagen`?** Because `validation_split=0.15` was defined on `train_datagen`. Keras needs the same generator object to know which files belong to the 85% and which to the 15%.

Using the same seed guarantees the split is identical every run. Different seeds could cause the same image to appear in both train and val — that's **data leakage**, and your metrics would be artificially inflated.

---

## `shuffle=False` on val and test

Evaluation metrics are computed by comparing predictions to true labels position by position. If the generator shuffles order, prediction index 0 no longer corresponds to label index 0 — your confusion matrix becomes meaningless. Shuffle only during training, where order doesn't matter.

---

## Eval Generator (no augmentation)

```python
eval_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input  # normalisation only, no augmentation
)
```

Test images get the same `preprocess_input` normalisation (so ResNet50 can read them), but no augmentation. Evaluation must happen on clean, unmodified images so metrics reflect true real-world performance.

---

---

# `model.py` — The Neural Network Architecture

> **What this file does:** This file defines the entire model architecture and compiles it ready for training. It loads the pretrained ResNet50 base, freezes its weights, and bolts on a custom classification head that learns to map ResNet's image features to 19 interior design style categories. It exposes a single `build_model()` function that the training notebook calls to get a compiled, ready-to-train model. It also returns the ResNet50 base as a separate reference so the training notebook can unfreeze specific layers during fine-tuning.

---

## Loading ResNet50

```python
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=IMG_SHAPE
)
```

**`weights='imagenet'`** — Loads weights pretrained on 1.2M ImageNet images. This is the entire point of transfer learning — borrowing knowledge the model already has.

**`include_top=False`** — Removes ResNet50's original 1000-class output layer so you can attach your own 19-class head.

**`input_shape=(224, 224, 3)`** — Must match what is set in `preprocess.py`. The 3 is RGB channels.

**Why ResNet50 specifically?** ResNet50 uses residual connections — shortcuts that let gradients flow through deep layers without vanishing. Earlier networks (VGG16, AlexNet) suffered from vanishing gradients in deep layers, causing early layers to stop learning. ResNet solved this. It is also proven on ImageNet and widely understood, making it easier to explain to a panel.

---

## Freezing the Base

```python
base_model.trainable = False
```

One line, massive consequence. Sets all ~25M ResNet50 weights to non-trainable. Without this, your small dataset (~780 images per class) would catastrophically overfit trying to retrain a 50-layer network — you'd destroy the ImageNet knowledge the model came with.

---

## The Custom Head — Order Matters

```python
x = GlobalAveragePooling2D()(x)
x = Dense(512)(x)            # linear only — no activation here
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
output = Dense(19, activation='softmax')(x)
```

**`GlobalAveragePooling2D`** — ResNet50 outputs shape `(7, 7, 2048)`. GAP averages each of the 2048 channels across the 7×7 grid, producing a `(2048,)` vector. The alternative, `Flatten`, would produce 7×7×2048 = 100,352 inputs — vastly more parameters, much higher overfitting risk. GAP has zero trainable parameters of its own.

**`Dense(512)` with no activation inline** — The order is deliberately `Dense (linear) → BatchNorm → ReLU → Dropout`. BatchNorm needs to see raw pre-activation values to normalise them properly. If you wrote `Dense(512, activation='relu')`, the activation fires before BatchNorm gets to normalise — you lose most of the stabilisation benefit.

**Why 512 neurons?** A practical middle ground between the 2048 input and the 19 output. Large enough to learn complex style combinations, small enough to train quickly on your dataset size.

**BatchNorm before Dropout — not after** — If BatchNorm came after Dropout, it would try to normalise values where 50% are zeroed during training but 0% are zeroed at inference. That mismatch corrupts BatchNorm's running statistics and makes inference behave differently from training.

**`Dropout(0.5)`** — Randomly deactivates 50% of the 512 neurons each training step. Forces every neuron to learn independently — no single neuron can be relied on. At inference, all neurons are active and outputs are scaled accordingly. This is the primary overfitting defence in the head.

**`Dense(19, activation='softmax')`** — 19 outputs, one per style. Softmax converts raw scores to probabilities summing to exactly 1.0 — e.g. `[0.60, 0.05, 0.10, ...]`. This vector also feeds the cosine similarity calculation in `predictor.py`.

---

## Compilation

```python
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**`Adam(lr=1e-4)`** — Adam adapts the learning rate per parameter based on gradient history, making it more efficient than plain SGD. 1e-4 is 10× lower than Adam's default (1e-3) — you're fine-tuning pretrained weights, so small nudges are needed, not large overwrites that destroy what ResNet already learned.

**`categorical_crossentropy`** — The correct loss for multi-class classification when labels are one-hot encoded (which `class_mode='categorical'` produces). Penalises confident wrong predictions much more heavily than uncertain ones.

---

## Why `build_model()` Returns Both `model` and `base_model`

```python
return model, base_model
```

When you call `Model(inputs=base_model.input, outputs=output)`, ResNet50's layers get flattened into the combined model graph. `model.layers[0]` is the `InputLayer`, not ResNet50 — you can't access the ResNet50 layers cleanly through `model` alone.

Returning `base_model` as a separate reference lets you do this cleanly in Phase 4b fine-tuning:

```python
for layer in base_model.layers[-30:]:
    layer.trainable = True
```

Because the layer objects are shared between `model` and `base_model`, setting `trainable=True` on `base_model.layers[-30:]` correctly updates those layers inside the combined model.

---

---

# `predictor.py` — Running Predictions in the Live App

> **What this file does:** This is the inference layer — the bridge between the trained model and the Streamlit frontend. It is never used during training. Instead, it gets called by `app.py` at runtime when a user uploads images. It handles everything needed to go from a raw user-uploaded image to a match score: preprocessing the image into the right format, running it through the model to get a style probability vector, averaging multiple images into a mood board profile, and computing cosine similarity between that profile and a product image. Every function here is designed to run fast enough for real-time use in the app.

---

## `_preprocess()`

```python
def _preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)
```

**`.convert('RGB')`** — User uploads could be PNG with transparency (RGBA) or greyscale. The model expects exactly 3 channels. Converting to RGB normalises every possible input format.

**`.resize(IMG_SIZE)`** — Product photos can be any size. Resizing to 224×224 matches training input exactly.

**`preprocess_input`** — Same transformation as in `preprocess.py`. Must be identical — if training used channel-mean subtraction but inference used `/255`, the model receives inputs in a different scale and predictions are wrong.

**`np.expand_dims(..., axis=0)`** — The model expects batches, shape `(batch, 224, 224, 3)`. A single image is `(224, 224, 3)`. Adding a dimension at axis 0 gives `(1, 224, 224, 3)` — a batch of one.

---

## `get_moodboard_vector()`

```python
vectors = [get_style_vector(model, img) for img in images]
return np.mean(vectors, axis=0)
```

Runs the model on each of the 3–5 inspiration images separately, then averages the resulting probability vectors element-wise. `axis=0` means average across images (not across the 19 classes). The result is a single `(19,)` vector representing the blended aesthetic of the entire mood board.

**Why average and not concatenate or sum?**
- Averaging keeps the result as a valid probability-like vector in the same scale as a single prediction
- Concatenating would give `(57,)` or `(95,)` depending on how many images — incomparable to a single product vector
- Summing would inflate values based on how many images were uploaded

---

## `_cosine_similarity()`

```python
def _cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)
```

**`np.dot(a, b)`** — The dot product: multiply corresponding elements and sum. High when both vectors have large values in the same positions (same dominant styles).

**`np.linalg.norm`** — The magnitude (length) of a vector. Dividing by it removes the effect of magnitude so only direction matters.

**`if norm == 0`** — Edge case guard. If a vector is all zeros, division by zero would crash. Returns 0.0 — no match.

**Why cosine and not Euclidean distance?** Euclidean distance cares about magnitude — a "strongly Scandinavian" room and a "mildly Scandinavian" room would seem far apart. Cosine similarity only cares about direction — both point toward Scandinavian and score high against a Scandinavian mood board. That's the right behaviour for style matching.

---

## `top_styles()`

```python
top_indices = np.argsort(vector)[-n:][::-1]
return [(CLASS_NAMES[i], round(float(vector[i]) * 100, 1)) for i in top_indices]
```

**`np.argsort`** — Returns indices that would sort the array in ascending order. `[-n:]` takes the last n (highest values). `[::-1]` reverses to descending — highest first.

**Why not just `np.argmax`?** `argmax` only gives the single highest index. `argsort` gives all indices ranked, so you can take the top 3 (or any n) — essential for displaying "60% Scandinavian, 20% Modern, 10% Industrial."

---

---

# `01_data_exploration.ipynb` — Cell by Cell

> **What this notebook does:** This is the sanity-check notebook run before any training starts. Its sole purpose is to verify that the data pipeline is working correctly. It checks that Keras found the right number of images across all 19 classes, that the class distribution is balanced, that images are being loaded and preprocessed correctly, and that data augmentation looks natural and not too aggressive. Nothing here changes the model or the data — it is purely diagnostic. Running this notebook gives confidence that the pipeline is solid before committing hours of GPU time to training.

---

## Cell 1 — Imports & Path Setup

```python
sys.path.insert(0, os.path.join(os.getcwd(), '..'))
from src.preprocess import get_generators, TRAIN_DIR, IMG_SIZE
```

The notebook lives inside `/notebooks/`. Python only looks for modules in the current folder by default, so `from src.preprocess import ...` would fail. `sys.path.insert(0, '..')` manually adds the project root to Python's search path.

**Why index 0?** Inserts at the front of the search list — your `src` is checked before any installed packages, avoiding naming conflicts.

**Why import from `src/preprocess.py` rather than writing the code directly in the notebook?** `preprocess.py` is the single source of truth for the data pipeline. If you change a parameter there, both the notebook and the training script update automatically. Avoids duplication and inconsistency.

---

## Cell 3 — Load Generators & Verify

```python
train_gen, val_gen, test_gen = get_generators()
print("Class → index mapping:")
for name, idx in train_gen.class_indices.items():
    print(f"  {idx:>2}  {name}")
```

**Output:**
```
Found 12654 images belonging to 19 classes.
Found 2222 images belonging to 19 classes.
Found 3729 images belonging to 19 classes.
```

**What `class_indices` is:** A dictionary Keras builds automatically by reading folder names. Each subfolder name becomes a class label assigned an integer. This mapping must stay consistent between training and inference — if `scandinavian` is index 12 during training, it must be 12 when the app makes predictions.

**The split math:** 12654 + 2222 = 14876 training pool. 2222/14876 = ~15% — exactly the `VAL_SPLIT` configured. Test set (3729) is completely separate and untouched during training.

**Why verify at all?** Silent data loading failures are common. If a folder had a typo or images were in the wrong place, Keras would just find fewer images and you'd never know without checking here.

---

## Cell 5 — Class Balance Check

```python
counts = [len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in class_names]
```

**Output:**
```
Min: 746 images  |  Max: 809 images  |  Difference: 63
```

Counts images per class folder directly from disk and plots a bar chart. The goal is to catch **class imbalance** — if one style has 5× more images than another, the model learns to predict it more often just because it sees it more.

A difference of 63 images across 19 classes is nearly perfect balance. The dataset is very well curated.

**What if it was imbalanced?** This is already handled in training via `compute_class_weight('balanced')`, which upweights underrepresented classes in the loss function so mistakes on rare classes are penalised more heavily.

---

## Cell 7 — Sample Batch Display & Deprocess

```python
_IMAGENET_MEAN = np.array([103.939, 116.779, 123.68])

def deprocess(img):
    img = img.copy()
    img += _IMAGENET_MEAN
    img = img[..., ::-1]   # BGR → RGB
    img = np.clip(img, 0, 255).astype('uint8')
    return img

images, labels = next(train_gen)
```

**Why deprocess?** `preprocess_input` transforms images into the range ~[-128, +151] by subtracting ImageNet channel means. `matplotlib.imshow()` expects values in [0, 255] as uint8 — feeding preprocessed values displays completely wrong colours. `deprocess()` reverses the transformation for display only. It does not affect training at all.

**The `[..., ::-1]` line:** `preprocess_input` also converts images from RGB to BGR (the format ResNet50 was originally trained on). `[::-1]` reverses the channel order back to RGB for display.

**`next(train_gen)`:** Generators are lazy — they only load images when asked. `next()` triggers one batch load of 32 images.

**`np.argmax(labels, axis=1)`:** Labels come out as one-hot vectors (e.g. `[0,0,0,0,0,0,0,1,0,...]` for Industrial at index 7). `argmax` finds the position of the 1, giving the integer class index to look up in `class_names`.

**What the output confirms:**
```
Image array shape:        (32, 224, 224, 3)
Preprocessed value range: [-128, +151]
```
32 images, 224×224 pixels, 3 colour channels. The value range confirms `preprocess_input` ran correctly — if `rescale=1./255` had been used instead, values would be in [0, 1].

---

## Cell 9 — Augmentation Preview

```python
aug_preview = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=20,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1
)
img_array = img_to_array(img).reshape((1, 224, 224, 3))
augmented = next(aug_preview.flow(img_array))[0].astype('uint8')
```

Takes one real image and generates 9 augmented versions, displayed alongside the original. This is a **visual sanity check** — confirming the augmentation looks natural and not so aggressive that it distorts style information.

**Why recreate the generator without `preprocess_input`?** For display purposes only — same reason as Cell 7. Natural colours are needed for visual inspection.

**`reshape((1, 224, 224, 3))`** — The generator's `.flow()` method expects a batch dimension. A single image is `(224, 224, 3)`. Reshaping adds the batch dimension of 1.

**`next(...)[0]`** — `.flow()` yields batches. `[0]` gets the first (and only) image from the batch of 1.

---

## Cells 10 & 11 — Environment Checks

```python
print(sys.version)      # 3.13.13
print(tf.__version__)   # 2.21.0
print(tf.config.list_physical_devices('GPU'))  # []
```

Verifies Python and TensorFlow versions and checks whether a GPU was detected.

**The GPU warning and empty list:** TensorFlow ≥ 2.11 dropped native Windows GPU support. `[]` means no GPU was used locally — training was done on **Google Colab** (free T4 GPU) using the separate `02_model_training_colab.ipynb`. This is why there are both local and Colab training notebooks.

---

---

# `02_model_training.ipynb` — Cell by Cell

> **What this notebook does:** This is where the model actually gets trained. It runs in two phases. Phase 4a trains only the custom head while keeping ResNet50 completely frozen — this teaches the head to interpret ResNet's features without disturbing the pretrained weights. Phase 4b unfreezes the top 30 ResNet layers and fine-tunes them at a much lower learning rate, adapting ResNet's deeper features specifically to interior design patterns. Throughout both phases, callbacks automatically stop training when progress stalls, save the best weights to disk, and reduce the learning rate when loss plateaus. The final output is `ambiance_model.h5` — the trained model file that the app loads at runtime.

---

## Cell 1 — Imports & Path Setup

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from src.preprocess import get_generators, TRAIN_DIR
from src.model import build_model

MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODELS_DIR, 'ambiance_model.h5')
HISTORY_PATH = os.path.join(MODELS_DIR, 'training_history.json')
CURVES_PATH  = os.path.join(MODELS_DIR, 'training_curves.png')
```

**`os.makedirs(..., exist_ok=True)`** — Creates the `models/` folder if it doesn't exist. `exist_ok=True` means it won't crash if the folder is already there — safe to run the cell multiple times.

**Three paths defined upfront** — `MODEL_PATH` is where best weights are saved. `HISTORY_PATH` saves training history as JSON. `CURVES_PATH` saves the loss/accuracy chart as PNG. Defining these once at the top means if you need to change a filename, you change it in one place only.

---

## Cell 3 — Load Generators & Assert

```python
train_gen, val_gen, test_gen = get_generators()
NUM_CLASSES = train_gen.num_classes
assert NUM_CLASSES == 19, f"Expected 19 classes, got {NUM_CLASSES}"
```

**The `assert` statement** — Defensive programming. If a folder was accidentally deleted, renamed, or a new folder crept in, Keras would silently find the wrong number of classes and training would proceed with a broken model. The assert catches this immediately with a clear error message.

**Why not just trust Keras?** Silent failures are the hardest bugs to catch. An assertion makes the failure loud and immediate rather than hours into training.

---

## Cell 5 — Class Weights

```python
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(class_weights_array))
```

**What `compute_class_weight('balanced')` does:** Calculates a weight per class using `total_samples / (num_classes × class_count)`. Classes with fewer images get a higher weight so the loss function penalises mistakes on them more heavily.

**`train_gen.classes`** — A flat array of integer labels for every image in the training set (e.g. `[0, 0, 1, 3, 12, ...]`). This is what `compute_class_weight` uses to count images per class.

**`dict(enumerate(...))`** — Converts the array `[0.97, 1.02, 0.98, ...]` into `{0: 0.97, 1: 1.02, 2: 0.98, ...}` — the format Keras `model.fit()` expects for `class_weight`.

**In practice:** The min/max difference was only 63 images, so weights are all close to 1.0. But computing them is correct practice and future-proofs the pipeline if the dataset changes.

---

## Cell 7 — Build the Model

```python
model, base_model = build_model(num_classes=NUM_CLASSES)
model.summary()
```

Calls `build_model()` from `src/model.py`. Both return values are kept — `model` for `.fit()` and `.evaluate()`, `base_model` for accessing ResNet50's layers during fine-tuning in Phase 4b.

`model.summary()` prints every layer, its output shape, and parameter count. Key thing to point out: the frozen ResNet50 layers show as non-trainable (~25M parameters). Only the custom head is trainable (~525K parameters).

---

## Cell 9 — Phase 4a: Head Training (Frozen Base)

```python
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5,
    restore_best_weights=True, verbose=1
)
checkpoint = ModelCheckpoint(
    MODEL_PATH, monitor='val_loss',
    save_best_only=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=3, min_lr=1e-6, verbose=1
)
```

**Three callbacks, each doing a specific job:**

**`EarlyStopping`** — Watches `val_loss` every epoch. If it hasn't improved for 5 consecutive epochs, training stops. `restore_best_weights=True` rewinds the model back to the epoch where val_loss was lowest — so you always end with the best version, not the most recent one.

*Why `val_loss` not `val_accuracy`?* Loss is more sensitive — it captures small improvements that accuracy (which rounds to discrete steps) can miss. Loss is also what the optimiser directly minimises.

**`ModelCheckpoint`** — Every time val_loss hits a new best, the model is saved to `ambiance_model.h5`. `save_best_only=True` means it only overwrites when it's genuinely better — you always have the best checkpoint on disk even if training crashes.

**`ReduceLROnPlateau`** — If val_loss hasn't improved for 3 epochs, the learning rate is halved (`factor=0.5`). Helps escape flat regions in the loss landscape. `min_lr=1e-6` is a floor — below that, updates are too small to be useful.

```python
history_head = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)
```

**`epochs=30`** — The maximum. EarlyStopping triggered at epoch 15 in practice (best was epoch 10, patience=5). The 30 is just a ceiling.

**`validation_data=val_gen`** — The held-out 15% used to compute val_loss and val_accuracy after each epoch. The model never trains on this data — purely for monitoring.

**Result:** Best val accuracy 31.9% at epoch 15.

---

## Cell 11 — Phase 4b: Fine-Tuning

```python
for layer in base_model.layers[-30:]:
    layer.trainable = True
```

Unfreezes the top 30 layers of ResNet50. `[-30:]` is Python negative indexing — the last 30 entries in the list, which are the deepest (most abstract) layers. Because `base_model` shares layer objects with `model`, this change automatically takes effect inside the full model.

**Why only top 30 out of 175 total layers?** Early ResNet layers detect universal features — edges, gradients, basic textures — that transfer perfectly from ImageNet to any visual domain. Only the deeper layers encode abstract, task-specific concepts that benefit from adapting to interior design patterns. Unfreezing all 175 layers with ~780 images per class would cause severe overfitting.

```python
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Must recompile after changing `trainable` flags** — Keras needs to rebuild the computation graph to register which parameters are now trainable. Without recompiling, the unfrozen layers would still not update.

**`learning_rate=1e-5`** — 10× lower than Phase 4a. The ResNet weights already encode good features — tiny nudges to adapt them, not large updates that destroy what they already know.

```python
early_stopping_ft = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint_ft     = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
```

**Why recreate the callbacks?** The original callback objects still have their internal patience counters from Phase 4a. Fresh ones reset those counters so fine-tuning gets a full 5-epoch window to improve before stopping.

**Result:** EarlyStopping triggered at epoch 6 (best was epoch 1, patience exhausted at epoch 6). Best val accuracy 33.35%. Overfitting observed: training accuracy climbed to 62% while val accuracy stayed flat at ~33%.

---

## Cell 13 — Training Curves & Save

```python
acc     = history_head.history['accuracy']     + history_fine.history['accuracy']
val_acc = history_head.history['val_accuracy'] + history_fine.history['val_accuracy']
phase_boundary = len(history_head.history['loss'])
```

**Concatenating the two histories** — Each `model.fit()` returns its own history object covering only its own epochs. Concatenating with `+` creates one continuous list spanning both phases for a single unbroken chart.

**`phase_boundary`** — The epoch index where Phase 4b started. Used to draw a vertical dashed line on the chart, visually separating the two training phases.

```python
axes[0].axvline(x=phase_boundary, color='gray', linestyle='--', label='Fine-tuning starts')
```

Makes it immediately obvious in the chart where frozen training ended and fine-tuning began — important for explaining the two-phase strategy visually.

```python
with open(HISTORY_PATH, 'w') as f:
    json.dump(full_history, f)
```

**Why save history to JSON?** Training on Colab takes time. Saving the numbers means you can re-plot curves in the evaluation notebook — or fix a formatting issue — without retraining from scratch.

```python
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
```

**Final test evaluation** — Run once, at the very end, on the completely separate `dataset_test` folder. This gives the unbiased real-world accuracy. The test set was never seen during training or used to make any decision — it's the final honest measurement.

---

---

# Quick-Reference: Anticipated Teacher Questions

| Question | Answer |
|---|---|
| Why `preprocess_input` not `rescale=1./255`? | ResNet50 was trained with channel-mean subtraction — inputs must match that exact scale or the frozen layers produce garbage |
| Why the same seed for train and val? | Prevents data leakage — different seeds could let the same image appear in both splits |
| Why `shuffle=False` on val/test? | Keeps prediction order aligned with label order for the confusion matrix |
| Why `GlobalAveragePooling2D` not `Flatten`? | Flatten gives 100,352 inputs vs 2048 — far more parameters and overfitting risk |
| Why `Dense → BatchNorm → ReLU → Dropout` in that order? | BatchNorm must see pre-activation values; after Dropout would corrupt its running stats |
| Why return both `model` and `base_model`? | ResNet50 layers are flattened into the combined graph — need `base_model` reference to target the top 30 layers for fine-tuning |
| Why cosine similarity not Euclidean distance? | Cosine only cares about direction (style profile shape), not magnitude — correct for style matching |
| Why average mood board vectors not concatenate? | Keeps result in the same (19,) scale as a product vector so they're directly comparable |
| Why `np.argsort` not `np.argmax` in `top_styles`? | `argmax` gives only the top 1 — `argsort` gives all ranked indices so you can take top N |
| Why `expand_dims` in `_preprocess`? | Model expects batch dimension `(1, 224, 224, 3)` — single image is `(224, 224, 3)` |
| Why `assert NUM_CLASSES == 19`? | Silent data loading failures are hard to catch — assert makes it loud and immediate |
| Why `EarlyStopping` monitors `val_loss` not `val_accuracy`? | Loss is more sensitive to small improvements; accuracy rounds to discrete steps |
| Why recreate callbacks for Phase 4b? | Original callbacks still hold patience counters from Phase 4a — fresh ones reset them |
| Why must you recompile after unfreezing layers? | Keras needs to rebuild the computation graph to register newly trainable parameters |
| Why `save_best_only=True` in ModelCheckpoint? | Ensures the saved file is always the best version, even if later epochs overfit |
| Why save training history to JSON? | Lets you re-plot curves without retraining — Colab sessions reset and lose variables |
| Why only unfreeze top 30 ResNet layers not all 175? | Early layers detect universal features that transfer perfectly; only deep layers need domain adaptation |
| Why `learning_rate=1e-5` in Phase 4b not 1e-4? | ResNet weights are already good — tiny nudges adapt them without destroying ImageNet knowledge |
