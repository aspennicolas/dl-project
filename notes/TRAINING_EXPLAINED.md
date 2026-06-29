# How We Trained the Model — Plain English

## What the model does

It takes a photo of a room and predicts which of 19 interior design styles it belongs to — Scandinavian, Industrial, Victorian, Farmhouse, etc.

---

## The Data

18,605 images total, split into three buckets:

| Split | Images | Purpose |
|---|---|---|
| Training | 12,654 | What the model learns from |
| Validation | 2,222 | Checked after each epoch to catch overfitting |
| Test | 3,729 | Only touched at the very end for the final honest score |

---

## Step 1 — Preprocessing

Before any image goes into the model, it's resized to **224×224 pixels** (the size ResNet was designed for) and its colour values are adjusted to match exactly how ResNet50 was originally trained — not just dividing by 255, but subtracting specific channel means. If you skip this step the model produces garbage even if everything else is right.

For **training images only**, random augmentation is applied: horizontal flips, slight rotations (±20°), brightness variation, and small zooms. This artificially makes the dataset bigger and teaches the model that a "Farmhouse kitchen" looks like a "Farmhouse kitchen" even if it's slightly tilted or a bit darker. Validation and test images get no augmentation — you want a fair, unmodified measurement there.

---

## Step 2 — Handling Class Imbalance

Some styles (like "Modern" or "Traditional") had far more images than others (like "Southwestern"). If you ignore this, the model cheats by predicting the common classes all the time and still looks good on paper.

To fix this, **class weights** were computed — rare classes get a higher penalty when the model gets them wrong. A mistake on a 300-image class hurts the training loss more than a mistake on a 1,000-image class, forcing the model to pay attention to small categories.

---

## Step 3 — The Model Architecture

Instead of training a neural network from scratch (which would need millions of images), the model uses **transfer learning** with ResNet50.

**What is ResNet50?** A deep neural network already trained by researchers on 1.2 million images from ImageNet. It already "knows" how to see — it can detect edges, textures, shapes, and patterns. We borrow those skills.

The model is two pieces bolted together:

**Part 1 — ResNet50 (the eyes):** Takes the 224×224 image and converts it into a 2,048-number "fingerprint" summarising what's visually in the image. In Phase 4a this part is frozen — its weights don't change.

**Part 2 — The custom head (the brain):** Takes those 2,048 numbers and learns to say "this fingerprint looks like Scandinavian." Structure:

```
ResNet50 output (7×7×2048)
    ↓
GlobalAveragePooling → (2048,)    # squashes spatial info, zero extra parameters
    ↓
Dense(512)                         # finds patterns across the 2048 features
    ↓
BatchNorm → ReLU                   # stabilises training, adds non-linearity
    ↓
Dropout(50%)                       # randomly switches off neurons to prevent overfitting
    ↓
Dense(19, softmax)                 # outputs 19 probabilities summing to 1.0
```

The softmax output looks like: `[0.60 Farmhouse, 0.05 Rustic, 0.03 Coastal, ...]`

---

## Step 4a — Training Just the Head (Frozen Base)

In the first training phase, the ResNet50 weights are **locked**. Only the custom head is trained.

Why? With ~780 images per class, there's not nearly enough data to retrain a 50-layer deep network. Doing so would cause catastrophic overfitting.

- **Optimizer:** Adam at learning rate 1e-4
- **Max epochs:** 30
- **EarlyStopping** — stops and rewinds to best weights if validation loss doesn't improve for 5 consecutive epochs
- **ReduceLROnPlateau** — halves the learning rate if loss stalls for 3 epochs
- **ModelCheckpoint** — saves to Drive every time a new best validation loss is hit (survives session crashes)

---

## Step 4b — Fine-Tuning (Unfreezing the Top of ResNet50)

Once the head has learned to classify interior styles, the **top 30 layers of ResNet50 are unfrozen** and training continues.

**Why only the top 30 (out of 175 layers)?**
Early layers detect universal things like edges and gradients — those transfer perfectly to any task. Only the deeper, later layers detect abstract high-level patterns that benefit from adapting to *this specific problem* (colour palettes, materials, architectural features of interior styles).

**Why learning rate 1e-5 (ten times smaller)?**
The ResNet weights are already good. The goal is to nudge them toward interior design features, not overwrite them.

---

## What Actually Happened

Fine-tuning stopped after just **6 epochs** (early stopping kicked in). Training accuracy was climbing (~62%) but validation accuracy plateaued at **~33%** and wasn't improving — the model was memorising the training set but not generalising.

Final test accuracy: **34.1%**

For 19 classes, random guessing is ~5%, so 34% is meaningful learning. The gap between train (~62%) and val (~33%) signals that 19 visually similar interior design categories is a hard problem, and the dataset size limits how far fine-tuning can take us.

---

## Summary

| Phase | What happened |
|---|---|
| Data prep | Images resized to 224×224, colour-normalised, training images augmented |
| Class weights | Rare styles penalised more to prevent majority-class bias |
| Build model | ResNet50 (frozen) + custom 512-neuron head |
| Phase 4a | Trained just the head for up to 30 epochs; ResNet frozen |
| Phase 4b | Unfroze top 30 ResNet layers; continued at 10× lower learning rate |
| Result | **34.1% test accuracy** (vs. ~5% random baseline for 19 classes) |
