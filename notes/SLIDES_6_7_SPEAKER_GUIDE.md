# Slides 6 & 7 — Speaker Guide
**What you need to understand + what to say**

---

## Slide 6 — Technical Architecture

### What you need to know

**Why CNN and not ANN or RNN?**

A CNN applies small filters that slide across an image detecting local patterns. The key is that it preserves *where* things are — a pixel's neighbourhood matters. A plain ANN flattens the image into a 1D list of pixels first, destroying all spatial structure. An RNN processes sequences one step at a time — it's for text and time series, not images. CNN is the only architecture that keeps the 2D structure of an image intact while learning from it.

**How CNNs build understanding layer by layer:**

- Layers 1–10: detects edges, corners, colour gradients
- Layers 10–30: detects textures — wood grain, fabric weave, concrete
- Layers 30–50: detects object parts — a chair leg, a cushion shape, an arched window

This is what "spatial hierarchy" means — simple features combine into complex ones as you go deeper.

**Why ResNet50 specifically — residual connections:**

Training a 50-layer network used to be nearly impossible. As you backpropagate the error signal through 50 layers, it gets multiplied by numbers less than 1 at each step — it vanishes to near-zero before it reaches the early layers. Early layers stop learning entirely. This is the **vanishing gradient problem**.

ResNet's fix: skip connections that jump over 1–2 layers. Instead of a layer learning a new function from scratch, it learns a *correction* on top of its input. Gradients can flow directly through the skip path without shrinking. That is why you can go 50 layers deep and still have the whole network learning.

**Transfer learning — frozen vs fine-tuned:**

ResNet50 was trained on 1.2 million ImageNet images across 1000 categories. It already knows how to detect edges, textures, shapes, materials. We exploit that.

- We **freeze the bottom 145 layers** because those general visual features are exactly what we need — a wood texture looks like a wood texture whether it's in a Scandinavian room or an ImageNet photo.
- We **fine-tune only the top 30 layers** because those encode higher-level concepts we want to shift toward interior design.

We have ~780 images per class — far too few to learn visual features from scratch. Transfer learning lets us stand on the shoulders of 1.2 million training images we never had to collect.

**GlobalAveragePooling2D:**

Before this layer, ResNet outputs a 7×7×2048 tensor. If we flattened that, we'd get 7×7×2048 = 100,352 numbers going into our dense layer — huge number of parameters, high overfitting risk, slow to train. GlobalAveragePooling instead takes the average of each 7×7 feature map, reducing the whole thing down to just 2048 numbers — one summary value per feature. Much more efficient, and forces the network to be spatially invariant (it doesn't care *where* in the image the feature appeared, only *whether* it appeared).

**The custom head — each component explained:**

| Component | What it does | Why it's there |
|---|---|---|
| Dense(512) | Reduces 2048 → 512 | Compresses representation, forces the model to extract only what matters for our task |
| BatchNormalization | Normalises activations to mean ≈ 0, std ≈ 1 | Stabilises training, placed *before* ReLU so half the inputs are positive and active |
| ReLU | Zeros out negative values | Adds non-linearity — without it, stacking layers is mathematically equivalent to one single linear layer |
| Dropout(0.5) | Randomly switches off 50% of neurons during training | Forces the network to learn redundant representations, prevents overfitting |
| Dense(19) + Softmax | 19 outputs, probabilities summing to 1 | One probability per style class — this is the style fingerprint |

**BatchNorm before ReLU — why the order matters:**

BatchNorm normalises *before* the activation so ReLU sees a distribution centred around zero. This means roughly half of the inputs will be positive and pass through — ReLU stays active and useful. If you BatchNorm after ReLU, you're normalising an already-clipped distribution and lose some of the benefit.

---

### What to say (Slide 6)

> "Our model is a Convolutional Neural Network. CNNs are the right tool here because they preserve spatial structure — they know that a pixel's neighbours matter. An ANN would flatten the image into a list of numbers and lose all of that. An RNN processes sequences — it's for text and time series. CNN is the only architecture that actually reads an image the way we do.
>
> CNNs build understanding layer by layer. Early layers detect edges and colour gradients. Middle layers detect textures — wood grain, concrete, fabric. Deep layers detect complex visual concepts — the silhouette of a mid-century chair, the geometry of an arched window. That is the spatial hierarchy.
>
> We chose ResNet50 because it solves a problem that stopped earlier deep networks from working: vanishing gradients. In a 50-layer network, the error signal gets smaller with every layer during training — by the time it reaches the early layers it's essentially zero, and those layers stop learning. ResNet adds shortcut connections that let gradients flow directly, bypassing this problem. That's why 50 layers actually work.
>
> And we use transfer learning. ResNet50 was pretrained on 1.2 million ImageNet images — it already knows how to detect edges, textures, materials. We freeze the bottom 145 layers because those universal visual features are exactly what we need. We only fine-tune the top 30 layers — shifting the high-level representations toward interior design. This lets us build a powerful model with 780 images per class instead of millions."

---

---

## Slide 7 — Prediction & Matching Logic

### What you need to know

**The style vector is everything:**

Every image that passes through our model produces a 19-dimensional probability vector — 19 numbers, each representing how strongly the image matches one style, all summing to 1. This is the image's style fingerprint. Everything the app does is built on top of this vector.

Example output for a Scandinavian room:
```
[0.60 Scandinavian, 0.20 Modern, 0.10 Industrial, 0.03 Minimalist, ...]
```

**Why we average the mood board vectors (np.mean):**

If image 1 gives `[0.80 Scandi, 0.05 Modern, ...]` and image 2 gives `[0.40 Scandi, 0.40 Industrial, ...]`, the element-wise mean gives `[0.60 Scandi, 0.22 Industrial, ...]`. The result is a blended profile that represents the user's aesthetic across all their reference images rather than any single one.

It's also cheap — no extra model training required, just arithmetic. The limitation (which we acknowledge in Slide 11) is that all images are weighted equally — a future improvement could weight by the model's confidence on each image.

**Why cosine similarity and not Euclidean distance:**

Euclidean distance measures how far apart two points are in space. Cosine similarity measures the *angle* between them — direction, not distance.

For softmax vectors, this distinction matters enormously. Imagine two images that are both clearly Scandinavian:
- Image A (very confident): `[0.95, 0.02, 0.02, ...]`
- Image B (less confident): `[0.50, 0.10, 0.10, ...]`

Euclidean distance would say these are quite different — the numbers are far apart. But cosine similarity sees that both vectors point in essentially the same direction. The *confidence* is different but the *style direction* is the same. Since we care about style direction, cosine similarity is the correct measure.

**The formula:**
```
similarity = dot(A, B) / (‖A‖ × ‖B‖)
```
The dot product captures how much the two vectors agree element-by-element. Dividing by the magnitudes normalises out the confidence difference. Since all softmax values are positive (0 to 1), the result is always between 0 and 1 — we multiply by 100 to get a percentage.

**Why this justifies Top-5 as the headline metric:**

The cosine similarity uses the *entire* 19-dimensional vector — all 19 probabilities contribute to the score. The exact top-1 label doesn't matter much. What matters is whether the overall shape of the vector is pointing in the right stylistic direction. Top-5 accuracy measures whether the correct style is anywhere in the model's most confident predictions — which reflects this much better than Top-1.

---

### What to say (Slide 7)

> "This slide is where the app's unique logic lives. The model itself is standard — what makes Ambiance work is what we do with the model's output.
>
> Every image that passes through our CNN produces a 19-dimensional probability vector. Think of it as a style fingerprint — 19 numbers representing how strongly that image matches each style, all summing to 1. For a Scandinavian room: 0.6 Scandinavian, 0.2 Modern, 0.1 Industrial, and so on.
>
> For the mood board, we run this on each of the 3 to 5 uploaded images and take the element-wise average. Image 1 might be strongly Scandinavian, image 2 might blend Scandinavian with Industrial — the average captures that mix. The result is a single composite profile representing the user's aesthetic across all their references, not just one image.
>
> The product photo goes through the exact same pipeline. We then ask: how similar are these two vectors?
>
> We use cosine similarity — and the choice matters. Cosine similarity measures the *angle* between two vectors, not the distance between them. For softmax outputs, magnitude encodes confidence, not style. A very confident Scandinavian vector and a less confident one are both pointing in the same direction — cosine similarity correctly identifies them as similar. Euclidean distance would wrongly say they're far apart.
>
> The result is a match score from 0 to 100 percent. And this is also why our headline metric is Top-5 accuracy, not Top-1 — the cosine similarity uses all 19 probabilities, so the full shape of the vector matters far more than which single class came first."

---

## Q&A — Questions likely to come at you on these two slides

**"Why not just match on the top-1 class instead of cosine similarity?"**
> Because cosine similarity captures style blends. A user's mood board might be 60% Scandinavian and 40% Industrial. Matching on top-1 only would miss that nuance entirely — a product that's 50/50 Scandinavian-Industrial would score as a mismatch even though it fits the mood board perfectly.

**"Why freeze 145 layers and not all of them?"**
> Freezing everything means the model can only use ImageNet features exactly as learned, with no adaptation to interior design. The top 30 layers learn to reuse lower-level features in ways more relevant to our specific domain — materials, room layouts, stylistic patterns that ImageNet never optimised for.

**"What's the difference between BatchNorm and Dropout?"**
> BatchNorm normalises activation values during training to stabilise learning. Dropout randomly deactivates neurons to prevent overfitting. Different problems, complementary solutions — BatchNorm stops the signal from exploding or shrinking, Dropout stops the model from memorising.

**"Why 19 classes specifically?"**
> That's what our dataset had — 19 interior design style categories. It's a recognised taxonomy in the field. More classes would require more data per class to learn from; fewer would lose important stylistic distinctions.
