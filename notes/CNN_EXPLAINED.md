# How the CNN Works for Ambiance

---

## The Big Picture

```
Raw photo of a room
        ↓
  ResNet50 (feature extractor)
        ↓
  "This image has: warm tones, exposed wood, 
   linen textures, rustic hardware..."
        ↓
  Custom Head (classifier)
        ↓
  [asian: 2%, coastal: 5%, farmhouse: 61%, rustic: 22%, ...]
        ↓
  Predicted style: Farmhouse
```

---

## Step by Step

**1. Preprocessing**
- Image resized to 224×224 pixels (what ResNet50 expects)
- Pixel values adjusted using `preprocess_input` — subtracts ImageNet's channel means so the image "looks like" what ResNet was originally trained on
- Augmentation applied during training only (flips, rotation, brightness) to simulate variety

---

**2. ResNet50 — The Feature Extractor**
```
Input image (224 × 224 × 3 pixels)
        ↓
Early layers → detect edges, corners, gradients
        ↓
Middle layers → detect textures, patterns, materials
        ↓
Deep layers → detect complex concepts (arched windows, 
              exposed brick, shiplap walls)
        ↓
GlobalAveragePooling2D
        ↓
Output: a vector of 2048 numbers
        ("the fingerprint of this image")
```

ResNet50 was pre-trained on ImageNet (1.2 million images, 1000 classes). It already knows how to see. You're just borrowing that ability.

---

**3. Custom Head — The Classifier**
```
2048-number fingerprint
        ↓
Dense(512) → learns which feature combinations 
             signal each interior style
        ↓
BatchNorm → stabilises the numbers
        ↓
ReLU → adds non-linearity (can learn curved patterns, not just lines)
        ↓
Dropout(0.5) → randomly switches off 50% of neurons 
               during training → prevents overfitting
        ↓
Dense(19) + Softmax → converts to 19 probabilities 
                      that sum to 100%
```

---

**4. Two-Phase Training**

```
Phase 4a — Head Training (frozen base)
├── ResNet50 weights: LOCKED (don't change)
├── Custom head weights: TRAINING
├── Learning rate: 1e-4
├── Goal: teach the head to read ResNet's fingerprints
└── Result: val_accuracy ~31.9%

Phase 4b — Fine-Tuning (partial unfreeze)
├── ResNet50 bottom 145 layers: LOCKED
├── ResNet50 top 30 layers: TRAINING ← adapting to interior design
├── Custom head weights: TRAINING
├── Learning rate: 1e-5 (10× smaller — nudge, don't overwrite)
└── Goal: make ResNet's deep layers care about interior design 
          specifically (shiplap, rattan, marble, etc.)
```

Why two phases? If you unfreeze ResNet50 straight away with a fresh head and a high learning rate, the random head weights send chaotic gradients backwards and destroy ResNet's ImageNet knowledge. Phase 4a stabilises the head first, then Phase 4b carefully adapts the base.

---

**5. What a Prediction Looks Like**

```
User uploads a photo
        ↓
preprocess_input()
        ↓
ResNet50 → 2048 fingerprint
        ↓
Custom head → [0.02, 0.05, 0.03, 0.61, 0.22, ...]
                                       ↑
                               Farmhouse: 61%
        ↓
For mood board: average the vectors from 3-5 photos
        ↓
Cosine similarity vs product image vector → match score
```

---

**Why accuracy is hard for your problem**

Many styles genuinely overlap visually:
```
Farmhouse ←→ Rustic (both: wood, neutral tones, textures)
Contemporary ←→ Transitional (both: clean lines, neutral palette)
Coastal ←→ Tropical (both: light colours, natural materials)
```

So 55-65% val_accuracy is actually a strong result — the model is distinguishing 19 styles that even humans sometimes disagree on.

---

---

# The Layers Explained

---

## What is a Layer?

A layer is just a mathematical transformation. It takes numbers in, does some math, and outputs new numbers. Stack enough of them and the network learns to see.

```
Numbers in → [do some math] → Numbers out
                  ↑
             this is a layer
```

---

## 1. The Input Layer

```
Your photo = 224 × 224 pixels × 3 colour channels (R, G, B)
           = 150,528 numbers, each between 0-255
```

That's all an image is to the network — a big grid of numbers.

---

## 2. Convolutional Layers (the core of ResNet50)

This is where the magic happens. A convolutional layer slides a small filter (like a 3×3 window) across the entire image, looking for a specific pattern.

```
Filter looking for vertical edges:

[ -1  0  1 ]
[ -1  0  1 ]   slides across image →  produces a "map" of 
[ -1  0  1 ]                          where vertical edges are
```

ResNet50 has many of these filters stacked together. Early ones find simple things, later ones find complex things:

```
Layer 1-10:   edges, corners, colour gradients
                    ↓
Layer 11-30:  textures (wood grain, fabric weave, marble veins)
                    ↓
Layer 31-50:  parts (window frames, furniture shapes, wall panels)
                    ↓
Layer 143-175: complex concepts (exposed brick wall, 
                shiplap panelling, rattan furniture)
```

**In your project:** The bottom 145 layers stay frozen — they already know how to detect textures and materials from ImageNet. The top 30 layers get fine-tuned because those are the ones that need to learn interior-design-specific concepts.

---

## 3. Residual Connections (what makes ResNet special)

The "Res" in ResNet stands for residual. This is the key innovation.

Normal deep network:
```
Input → Layer 1 → Layer 2 → Layer 3 → Output
```
Problem: in very deep networks, gradients (the learning signal) shrink as they travel backwards through 50 layers and eventually vanish — early layers stop learning.

ResNet's solution — skip connections:
```
Input ──────────────────────────┐
  ↓                             ↓
Layer 1 → Layer 2 → Layer 3 → ADD → Output
```

The input is added directly to the output, creating a shortcut. Gradients can flow directly through the shortcut without shrinking. This is why ResNet can be 50 layers deep and still train effectively. Without this, 50 layers would be nearly impossible to train.

---

## 4. GlobalAveragePooling2D

After ResNet50's convolutional layers, the output is a 7×7×2048 tensor — 7×7 is a spatial grid, 2048 is the number of feature maps.

```
7×7×2048 tensor
(think: 2048 small 7×7 "activation maps", 
 each one lighting up where a feature was found)
        ↓
GlobalAveragePooling2D
(average each 7×7 map down to a single number)
        ↓
2048 numbers
("how much of each feature was present overall")
```

Alternative would be Flatten: 7×7×2048 = 100,352 numbers fed into the head — way too many parameters, would overfit badly.

---

## 5. Dense(512) — the first head layer

```
2048 numbers in
        ↓
512 neurons, each connected to all 2048 inputs
Each neuron = weighted sum of all 2048 inputs + bias
        ↓
512 numbers out
```

Each of the 512 neurons learns to detect a combination of features. For example one neuron might learn:

```
"high wood texture + high warm tones + high rustic hardware 
 + low marble + low chrome = farmhouse signal"
```

You can't read this directly from the weights — the network figures it out on its own during training.

---

## 6. BatchNormalization

```
512 numbers come in — some might be very large (500), 
some very small (0.001), hard to train consistently
        ↓
BatchNorm: subtract mean, divide by std → all ~centred around 0
Then apply learnable scale + shift → network adjusts as needed
        ↓
512 normalised numbers
```

Without this, large numbers dominate and training becomes unstable. Think of it like making sure everyone on a team is operating on the same scale before combining their inputs.

---

## 7. ReLU (Activation Function)

```
For each of the 512 numbers:
  if number < 0 → output 0
  if number ≥ 0 → output the number unchanged
```

```
-3.2 → 0
-0.1 → 0
 0.0 → 0
 1.4 → 1.4
 5.7 → 5.7
```

Why? Without an activation function, stacking layers is pointless — linear math on top of linear math is still just linear math. ReLU introduces non-linearity, letting the network learn curved decision boundaries. ReLU specifically is used because it's fast and doesn't suffer from the vanishing gradient problem that older activations like sigmoid did.

---

## 8. Dropout(0.5)

```
During training only:
512 neurons → randomly zero out 50% each step

Step 1: neurons 3, 7, 12, 45... switched off
Step 2: neurons 1, 8, 19, 33... switched off (different ones)
Step 3: neurons 2, 5, 11, 40... switched off
```

Forces the network to learn redundant representations — no single neuron can be relied upon, so many neurons learn to recognise the same feature independently. At inference time, all neurons are active. This is the main thing preventing overfitting in your head.

---

## 9. Dense(19) + Softmax — the output layer

```
512 numbers in
        ↓
19 neurons (one per interior style)
Each neuron = weighted sum of all 512 inputs
        ↓
19 raw scores (called logits), e.g.:
  farmhouse:    8.3
  rustic:       6.1
  scandinavian: 2.4
  ...
        ↓
Softmax: converts to probabilities summing to 100%
  farmhouse:    61%
  rustic:       22%
  scandinavian:  4%
  ...
```

Softmax formula for each class: e^score / sum of all e^scores. The exponential makes the biggest score dominate, which is what you want — the model should be confident about its top prediction.

---

## The Full Stack Together

```
Photo (150,528 numbers)
        ↓
[ResNet50 — 175 convolutional layers]
  edges → textures → parts → concepts
        ↓
GlobalAveragePooling2D → 2048 numbers
        ↓
Dense(512) → learns style-relevant combinations
        ↓
BatchNorm → stabilises scale
        ↓
ReLU → adds non-linearity
        ↓
Dropout(0.5) → prevents overfitting
        ↓
Dense(19) + Softmax → 19 style probabilities
```
