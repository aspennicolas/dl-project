# Ambiance — Forward Pass, Backward Pass, and Residual Connections Explained

A step-by-step walkthrough of what happens to one image, from raw pixels to match score — plus how the model learns from its mistakes.

---

## Step 0 — The Raw Image

You upload a photo of a Scandinavian living room. It gets loaded as a grid of pixels:

```
Image shape: (224, 224, 3)

224 rows × 224 columns × 3 colour channels (Red, Green, Blue)

Every pixel is 3 numbers:
  pixel at row 10, col 45:  [R=230, G=215, B=200]  ← a warm beige wall
  pixel at row 80, col 12:  [R=120, G=100, B=80]   ← a dark wooden floor
```

Total numbers in one image: `224 × 224 × 3 = 150,528`

Before entering the model, ResNet50's `preprocess_input` subtracts the ImageNet channel means:

```
R values: subtract 103.9
G values: subtract 116.8
B values: subtract 123.7
```

This puts values in the same scale ResNet50 was trained on. The image is now ready.

---

## Step 1 — Enter ResNet50 (Frozen Layers)

The image flows into ResNet50's first convolutional layer. Here is what a convolution actually does:

```
A filter (small grid of learned weights, e.g. 3×3) slides across the image.
At each position it does: multiply each weight by the pixel underneath, sum them all up.

Filter (3×3):          Image patch:          Result:
┌──────────────┐       ┌──────────────┐
│ -1  -1  -1  │   ×   │ 200 210 195  │
│  0   0   0  │       │ 205 215 200  │   →  sum = 0.8  (edge detected here)
│  1   1   1  │       │ 120 125 118  │
└──────────────┘       └──────────────┘
```

This filter slides across the entire 224×224 image, producing one 2D feature map. ResNet50's first layer has **64 filters**, so it produces 64 feature maps simultaneously — 64 different things being detected at once.

As you go deeper into ResNet50, the filters detect increasingly abstract things:

```
Early layers  (frozen):  edges, corners, colour gradients
Middle layers (frozen):  textures — wood grain, fabric weave, concrete
Deep layers   (frozen):  patterns — geometric shapes, organic curves
Top layers    (fine-tuned): concepts — "this looks like a minimalist room"
```

**The residual connection (skip connection):**

At each residual block, the input is added directly to the output:

```
Input x
   │
   ├──────────────────────────────────┐
   │                                  │
   ▼                                  │
Conv → BN → ReLU → Conv → BN         │  (learn the difference from input)
   │                                  │
   ▼                                  │
   + ◄────────────────────────────────┘  (add original input back)
   │
   ▼
ReLU
```

Why does this matter? In very deep networks, gradients tend to shrink to near-zero as they flow backwards through many layers during training — this is the **vanishing gradient problem**, and it means early layers stop learning entirely. The skip connection gives gradients a shortcut path to flow backwards without passing through all those layers, so even the earliest layers keep learning. This is why ResNet (Residual Network) can be 50 layers deep and still train effectively.

After all the convolutional layers, the output is:

```
(7, 7, 2048)

2048 feature maps, each a 7×7 grid.
Each grid cell = "how strongly was this feature detected at this region of the image"
```

---

## Step 2 — GlobalAveragePooling2D

Each of the 2048 feature maps (channels) is a 7×7 grid. GAP collapses each one into a single number by averaging all 49 values:

```
Feature map 317 — "warm wood texture detector":

┌─────────────────────────────┐
│ 0.8  0.7  0.9  0.8  0.7  │
│ 0.6  0.8  0.9  0.7  0.8  │       Average = 0.71
│ 0.5  0.6  0.7  0.6  0.5  │  ──►
│ 0.4  0.5  0.6  0.5  0.4  │       "Warm wood texture was strongly
│ 0.3  0.4  0.5  0.4  0.3  │        present throughout this image"
└─────────────────────────────┘

Feature map 891 — "concrete/industrial texture detector":

┌─────────────────────────────┐
│ 0.0  0.1  0.0  0.0  0.1  │
│ 0.0  0.0  0.1  0.0  0.0  │       Average = 0.03
│ 0.1  0.0  0.0  0.0  0.1  │  ──►
│ 0.0  0.0  0.0  0.0  0.0  │       "Almost no concrete/industrial
│ 0.0  0.1  0.0  0.0  0.0  │        texture in this image"
└─────────────────────────────┘
```

After GAP, you have one number per channel:

```
(7, 7, 2048)  →  (2048,)

A single vector of 2048 numbers.
Each number = "how present was this feature anywhere in the image overall"
```

This vector is a rich description of what the image contains — 2048 different feature detectors all summarised into a single 1D vector. This is also why GAP is preferred over Flatten: Flatten would produce 100,352 values and inflate the parameter count of the next Dense layer by 50×, causing overfitting on a small dataset.

---

## Step 3 — Dense(512)

Now the vector enters the custom head. The first Dense layer has 512 neurons.

Each neuron looks at **all 2048 values** and computes a weighted sum:

```
Neuron 1:
  output = (0.71 × w₁) + (0.03 × w₂) + (0.55 × w₃) + ... (2048 terms)

  = one number
```

All 512 neurons do this simultaneously, each with their own set of 2048 weights:

```
Input:   (2048,)  — the GAP vector
Weights: (2048 × 512)  — 1,048,576 learned weights
Output:  (512,)  — 512 numbers, one per neuron
```

What are the neurons learning? Each neuron is learning a different combination of the 2048 ResNet features that is useful for distinguishing interior styles. One neuron might learn to fire strongly when both "warm wood" and "minimal clutter" features are high — which together signal Scandinavian. Another might fire when "ornate detail" and "dark wood" are both high — signalling Victorian.

The output at this point is 512 raw numbers, positive and negative, potentially very large.

---

## Step 4 — BatchNormalization

Those 512 raw values could look like this:

```
[-847, 0.003, 1204, -0.1, 562, -23, 4891, ...]
```

Wildly different scales. BatchNorm rescales them to a consistent distribution:

```
Before BN:  [-847,  0.003,  1204,  -0.1,   562,  -23,  4891]
After BN:   [-0.8,  -1.2,    1.4,   0.9,   0.3,  -0.4,  1.8]
```

Now all values are in a similar range, which makes training stable. The next layer (ReLU) can work with these values predictably.

**Why BatchNorm comes before ReLU:** ReLU kills all negative values. If ReLU ran first, BatchNorm would try to normalise a distribution that is half zeros — a distorted picture. BatchNorm before ReLU sees the full distribution (negatives included) and normalises it correctly.

**Why Dropout comes after BatchNorm:** Dropout zeroes 50% of neurons during training but is off at inference. If Dropout ran before BatchNorm, BN would learn running statistics from a half-dead distribution during training, then apply those stats to a fully-active distribution at inference — a mismatch that corrupts normalisation.

Correct order: `Dense → BatchNorm → ReLU → Dropout`

---

## Step 5 — ReLU

ReLU is applied to each of the 512 values individually:

```
Rule: if value > 0, keep it. If value ≤ 0, set it to 0.

Before ReLU:  [-0.8,  -1.2,   1.4,   0.9,   0.3,  -0.4,   1.8]
After ReLU:   [ 0.0,   0.0,   1.4,   0.9,   0.3,   0.0,   1.8]
```

Without a non-linearity like ReLU, stacking Dense layers is mathematically equivalent to just one Dense layer — the whole stack collapses into a single linear transformation and loses the ability to learn complex patterns. ReLU breaks the linearity and lets the network model curved, non-linear decision boundaries between styles.

Output: `(512,)` — same shape, but negative values are gone.

---

## Step 6 — Dropout(0.5)

During **training only**, Dropout randomly picks 50% of the 512 neurons and sets them to zero:

```
Before Dropout:  [0.0,  0.0,  1.4,  0.9,  0.3,  0.0,  1.8,  0.6, ...]
After Dropout:   [0.0,  0.0,  0.0,  0.9,  0.0,  0.0,  1.8,  0.0, ...]
                              ↑ zeroed        ↑ zeroed       ↑ zeroed
```

The zeroed neurons are different every single training step. This forces the network to not rely on any single neuron — if neuron 42 keeps getting switched off, the other neurons have to learn to cover for it. The result is a more robust, distributed representation.

At **inference** (when you upload an image in the app), Dropout is completely switched off and all 512 neurons are active.

---

## Step 7 — Dense(19) + Softmax

The final Dense layer has 19 neurons — one per style class:

```
Input:   (512,)
Weights: (512 × 19)  — 9,728 weights
Output:  (19,)  — 19 raw scores
```

Each output neuron computes a weighted sum over all 512 inputs. Each neuron has learned to fire when the pattern of 512 features matches its assigned style.

Then Softmax converts the 19 raw scores into probabilities that sum to 1.0:

```
Raw scores:    [2.1,  0.3,  0.8,  0.1,  4.7,  0.2, ...]
                Asian Coastal Cont. Craft. Scand. ...

Softmax:       e^each_score / sum(e^all_scores)

Probabilities: [0.02, 0.01, 0.03, 0.01, 0.58, 0.01, ...]
                                          ↑
                                     58% Scandinavian
```

This is the **style vector** — a 19-dimensional probability distribution over all interior design styles.

---

## Step 8 — Mood Board Averaging

The model runs the full pipeline above once per uploaded image. If you upload 4 images:

```
Image 1 (living room):  [0.02, 0.01, 0.08, 0.01, 0.60, 0.01, ...]  → 60% Scandinavian
Image 2 (bedroom):      [0.01, 0.02, 0.10, 0.02, 0.55, 0.02, ...]  → 55% Scandinavian
Image 3 (kitchen):      [0.03, 0.01, 0.12, 0.01, 0.50, 0.03, ...]  → 50% Scandinavian
Image 4 (dining room):  [0.02, 0.03, 0.09, 0.01, 0.53, 0.02, ...]  → 53% Scandinavian

np.mean(axis=0) — average each of the 19 positions across all 4 vectors:

Moodboard vector: [0.02, 0.02, 0.10, 0.01, 0.55, 0.02, ...]
                                            ↑
                                      55% Scandinavian
```

Averaging smooths out noise — if one image was ambiguous, the other three pull the vector back toward the true aesthetic.

---

## Step 9 — Cosine Similarity

The product image goes through the exact same pipeline (Steps 0–7) and produces its own style vector.

Then cosine similarity compares the two vectors:

```
Moodboard vector:  [0.02, 0.02, 0.10, 0.01, 0.55, ...]
Product vector:    [0.01, 0.01, 0.08, 0.02, 0.61, ...]

similarity = dot product / (magnitude of A × magnitude of B)
           = how much do these two vectors point in the same direction?
```

Think of it geometrically — both vectors are points in 19-dimensional space. Cosine similarity measures the angle between them:

```
Angle ≈ 0°   →  similarity = 1.0  →  100% match  (same style distribution)
Angle ≈ 90°  →  similarity = 0.0  →  0% match    (completely different styles)
```

It uses angle, not distance, because what matters is the *shape* of the style distribution — whether both vectors weight the same styles highly — not their absolute magnitudes.

```
similarity = 0.92  →  score = 92%  →  Strong Match
```

---

## The Full Picture

```
224×224×3 image
      ↓
ResNet50 (frozen): pixel patterns → textures → shapes → concepts
      ↓  (7,7,2048)
ResNet50 (fine-tuned): interior design concepts
      ↓  (7,7,2048)
GAP: "how present was each feature overall?"
      ↓  (2048,)
Dense(512): "which combinations of features signal each style?"
      ↓
BN → ReLU → Dropout
      ↓  (512,)
Dense(19) + Softmax: "what are the probabilities across 19 styles?"
      ↓  (19,)
Style vector

× N images → average → moodboard vector
× 1 product image    → product vector

cosine similarity → match score 0–100%
```

Every number in every vector, at every stage, is the result of learned weights being applied to the input. Training is the process of adjusting all those weights — millions of them — until the final 19-dim output consistently matches the correct style label for each training image.

---

---

# Backward Pass — How the Model Learns

## The Core Idea

After the forward pass produces a prediction, the model checks how wrong it was. The backward pass then works backwards through every layer, adjusting every weight slightly to make the prediction less wrong next time.

```
Forward pass:   image → layers → prediction
Backward pass:  prediction error → layers (in reverse) → weight updates
```

---

## Step by Step

**1 — Compute the loss**

After the forward pass you have:

```
Model predicted:  [0.02, 0.01, 0.03, 0.01, 0.58, ...]   (58% Scandinavian)
Correct answer:   [0.00, 0.00, 0.00, 0.00, 1.00, ...]   (100% Scandinavian — one-hot)

Loss (categorical crossentropy) = how far off was the prediction?
→ loss = 0.54   (a single number summarising total error)
```

The lower the loss, the better the prediction.

**2 — Compute gradients (backpropagation)**

The model asks: *if I nudge each weight slightly, does the loss go up or down?*

That rate of change is called a **gradient**. It is computed for every single weight in the network using the chain rule from calculus — working backwards layer by layer:

```
Loss
  ↑
Dense(19) weights — how much did each weight contribute to the error?
  ↑
Dropout
  ↑
ReLU
  ↑
BatchNorm
  ↑
Dense(512) weights — same question
  ↑
GlobalAveragePooling
  ↑
ResNet50 fine-tuned layers — same question
  ↑
ResNet50 frozen layers — gradients computed but NOT applied (weights locked)
```

Each layer passes the gradient signal backwards to the layer before it. This is why the residual (skip) connections matter so much — they give the gradient a shortcut path backwards so it does not vanish before reaching the early layers.

**3 — Update the weights (Adam optimizer)**

Once gradients are computed, Adam uses them to update every trainable weight:

```
new_weight = old_weight − (learning_rate × gradient)

learning_rate = 1e-4   (how big a step to take)
gradient      = which direction to move, and by how much
```

If the gradient is positive (increasing the weight made the loss worse), the weight decreases. If the gradient is negative (increasing the weight helped), the weight increases. Every weight gets nudged in whichever direction reduces the loss.

---

## One Training Step = Forward + Backward

```
Batch of 32 images
      ↓
Forward pass → predictions
      ↓
Loss calculation → one number
      ↓
Backward pass → gradients for every weight
      ↓
Adam → update every trainable weight
      ↓
Repeat with next batch
```

This cycle runs thousands of times across all your training images. Each cycle the weights get slightly better at predicting the correct style. After enough cycles (epochs), the loss plateaus and you stop — that is what early stopping detects.

---

## Why Frozen Layers Still Matter in Backward Pass

Even though the frozen ResNet50 layers do not get their weights updated, the gradient still **flows through them** during backpropagation. This is what allows the fine-tuned top 30 layers to learn — the gradient has to pass through the frozen layers to reach them. The frozen layers just do not apply the update at the end.

```
Gradient flows through ALL layers  ←── backprop needs this
Weights updated in TRAINABLE layers only  ←── frozen layers ignored at update step
```

---

---

# Residual Connections (Skip Connections) — Deep Dive

## The Problem They Solve

Imagine you are passing a message through a chain of 50 people. Each person slightly garbles the message before passing it on. By the time it reaches person 50, the original message is unrecognisable.

That is exactly what happens to gradients during backpropagation in a deep network. The gradient signal starts at the output layer and travels backwards through 50 layers. At each layer it gets multiplied by numbers that are slightly less than 1. After 50 multiplications:

```
0.9 × 0.9 × 0.9 × ... (50 times) = 0.005
```

The gradient arriving at layer 1 is nearly zero. A gradient of zero means "do not change your weights" — so the early layers stop learning entirely. This is the **vanishing gradient problem**.

---

## What the Skip Connection Does

Instead of only passing information through the layer, ResNet adds a **direct shortcut** that bypasses it:

```
Without skip connection:

Input x
   ↓
[Conv → BN → ReLU → Conv → BN]
   ↓
Output
```

```
With skip connection:

Input x
   │
   ├─────────────────────────────┐   ← shortcut, x passes through unchanged
   ↓                             │
[Conv → BN → ReLU → Conv → BN]  │   ← the layer learns the difference
   ↓                             │
   +  ◄──────────────────────────┘   ← add them together
   ↓
ReLU
↓
Output
```

The output is not just what the layer learned — it is **what the layer learned PLUS the original input**.

---

## What the Layer Actually Learns

If the output is:

```
output = layer(x) + x
```

Then the layer is not learning the full output from scratch. It is only learning the **difference** between the input and the ideal output:

```
layer(x) = output − x   ← just the residual (the leftover correction)
```

That is where the name comes from — **Residual** Network. The layer only needs to learn a small correction on top of what was already there, not the entire transformation from scratch.

If the layer does not need to change anything, it can simply learn weights of zero and the input passes through untouched. This means adding more layers cannot hurt the model — worst case they do nothing, best case they add useful information.

**Concrete example:**

```
Input x represents "wooden texture detected":

Input x:                  [0.7, 0.3, 0.8, 0.2, ...]   ← already a good representation
Layer output (residual):  [0.1, 0.0, 0.1, 0.0, ...]   ← small correction: "slightly more wood"
Final output (x + layer): [0.8, 0.3, 0.9, 0.2, ...]   ← refined version
```

The layer did not rebuild the representation from zero — it just nudged it.

---

## How Backpropagation Flows Through a Skip Connection

This is the key reason skip connections exist. During the backward pass, the gradient splits at the `+` point:

**Forward pass** (left to right):

```
Input x  ──────────────────────────────────────┐
   ↓                                           │
Conv → BN → ReLU → Conv → BN                  │
   ↓                                           │
   +  ◄────────────────────────────────────────┘
   ↓
ReLU → Output
```

**Backward pass** (right to left):

```
Gradient from next layer
         ↓
         +  ← gradient SPLITS here into two paths
        / \
       /   \
      ↓     ↓
Path 1:           Path 2:
through           through the shortcut
Conv ← BN ←       (passes through unchanged —
ReLU ← Conv ←     no layers to shrink it)
BN
(gradient shrinks
a little here)
      \           /
       \         /
        ↓       ↓
   Both paths arrive at Input x and are summed
```

Path 2 always delivers a clean, full-strength gradient directly to the input no matter how deep the network is. Even if Path 1 produces a near-zero gradient after passing through many layers, Path 2 guarantees the earlier layer always receives a usable signal and keeps learning.

---

## Input and Output — What These Terms Mean

"Input" and "output" just mean what enters and leaves a specific layer — nothing more:

```
Dense(512)
  input:  the 2048-dim vector coming from GAP
  output: a 512-dim vector going to BatchNorm

BatchNorm
  input:  that 512-dim vector
  output: the same 512-dim vector, rescaled

ReLU
  input:  the rescaled 512-dim vector
  output: the same vector, but negatives set to zero
```

Every layer receives something, does something to it, and hands it to the next layer. "Input" and "output" just label the before and after for that one step.

---

## Summary

| Concept | What it is |
|---|---|
| Forward pass | Image flows through layers left to right → produces a prediction |
| Loss | A single number measuring how wrong the prediction was |
| Backward pass | Error signal flows right to left → computes how much each weight contributed to the error |
| Gradient | The rate of change of the loss with respect to one weight — tells Adam which direction to nudge it |
| Weight update | Adam adjusts every trainable weight slightly in the direction that reduces the loss |
| Vanishing gradient | Gradient shrinks to near-zero in deep networks → early layers stop learning |
| Skip connection | Shortcut that lets both data (forward) and gradient (backward) bypass layers — solves vanishing gradient |
| Residual | What the layer actually learns — just the small correction on top of the input, not the full output |
