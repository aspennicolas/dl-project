# Ambiance — Presentation Script
**IE University Deep Learning Final Project**
**Total time: 15 minutes including Q&A**

> Rubric reminder: Business (20%) | Technical Depth (25%) | MVP Integration (25%) | Presentation (20%) | Live Demo (10%)
> Every slide should serve at least one of these pillars. Business metrics MUST be quantified. All team members MUST speak.

---

## Slide 1 — Title (30 sec)
**Speaker: Person 1**

**On slide:**
- Ambiance
- Interior Design Style Matcher
- Team names | IE University | Deep Learning Final Project

**Say:**
> "Good [morning/afternoon]. We're [Team Name] and today we're presenting Ambiance — a deep learning application that decodes your interior design aesthetic and tells you whether a piece of furniture actually belongs in your home."

---

## Slide 2 — Hook (1 min)
**Speaker: Person 1**

**On slide:**
- Large image: a beautiful mood board next to a mismatched room
- Stat in big font: **"1 in 4 furniture purchases is returned due to style mismatch"**

**Say:**
> "Has this ever happened to you? You spend hours building a Pinterest board. You find what looks like the perfect sofa. It arrives — and something is just off. The colour, the lines, the vibe. It doesn't fit.
>
> You're not alone. Industry data shows 15 to 30 percent of furniture purchases are returned because of style mismatch — not damage, not defects, just: this doesn't look right in my home.
>
> That's the problem we set out to solve."

---

## Slide 3 — The Problem (1 min)
**Speaker: Person 2**

**On slide:**
- Three columns:
  - **For consumers:** Hours wasted comparing, still getting it wrong
  - **For retailers:** £50B+ in annual return costs globally
  - **The gap:** No tool translates a mood board into purchase validation

**Say:**
> "The root cause is simple: people know what they like when they see it, but they can't articulate it — and they definitely can't check whether a random product matches it.
>
> Current tools don't help. Pinterest lets you save images. IKEA Kreativ lets you visualise furniture in a room. But nothing decodes your personal aesthetic from a mood board and validates a purchase against it automatically.
>
> That's the gap Ambiance fills."

---

## Slide 4 — The Solution (1 min)
**Speaker: Person 2**

**On slide:**
- Simple 3-step visual flow:
  1. Upload 3–5 inspiration images → **Your style profile**
  2. Upload a product photo → **Match score 0–100%**
  3. ✅ Buy with confidence / ❌ Keep looking

**Say:**
> "Ambiance works in two steps.
>
> First, you upload 3 to 5 images of spaces you love — your mood board. Our CNN analyses each image, extracts its style fingerprint, and averages them into your personal aesthetic profile: for example, 60% Scandinavian, 25% Modern, 15% Industrial.
>
> Second, you upload a product photo. The model extracts the product's style fingerprint and computes how closely it matches your profile. You get a match score from 0 to 100 percent — and a clear yes or no."

---

## Slide 5 — Market Context (30 sec)
**Speaker: Person 2**

**On slide:**
- Competitor table:
  | Tool | What it does | What it doesn't do |
  |---|---|---|
  | Pinterest | Saves inspiration images | No style decoding, no product validation |
  | IKEA Kreativ | Visualises furniture in your room | No mood board analysis, no match scoring |
  | **Ambiance** | Decodes aesthetic + validates products | — |

**Say:**
> "IKEA has already invested in AI for interior design — IKEA Kreativ, their room visualiser. But even they don't do what we do. No existing tool decodes your aesthetic DNA from a raw mood board and validates arbitrary product photos against it. This is a clear white space."

---

## Slide 6 — Technical Architecture (1.5 min)
**Speaker: Person 3**

**On slide:**
```
Input Image (224×224)
      ↓
ResNet50 — pretrained on ImageNet
  Bottom 145 layers: FROZEN (edges, textures, patterns)
  Top 30 layers: FINE-TUNED (interior design concepts)
      ↓
GlobalAveragePooling2D → 2048-dim vector
      ↓
Custom Head:
  Dense(512) → BatchNorm → ReLU → Dropout(0.5)
      ↓
Dense(19) + Softmax → style probabilities
```

**Say:**
> "Our model is a Convolutional Neural Network — the correct architecture for image classification. CNNs learn spatial hierarchies: early layers detect edges and textures, deeper layers detect complex visual concepts. An ANN would ignore spatial relationships between pixels. An RNN is for sequential data. CNN is the only sensible choice here.
>
> We chose ResNet50 specifically because it uses residual connections — shortcuts that let gradients flow through 50 layers without vanishing. This solves a problem that prevented earlier deep networks from actually learning.
>
> We use transfer learning. ResNet50 was pretrained on 1.2 million ImageNet images — it already knows how to see. We freeze the bottom 145 layers (universal features) and fine-tune only the top 30 (domain-specific features). This lets us train a powerful model with ~780 images per class instead of millions."

---

## Slide 7 — Data Pipeline & Hyperparameters (1 min)
**Speaker: Person 3**

**On slide:**
- Dataset: 18,876 images | 19 classes | ~780 train / ~195 test per class
- Preprocessing choices (justified):
  - `preprocess_input` not `rescale=1/255` — must match ImageNet training distribution
  - `horizontal_flip`, `rotation_range=20`, `brightness_range=[0.8,1.2]`, `zoom_range=0.1`
  - `validation_split=0.15` with fixed seed — prevents data leakage
  - Class weights: `compute_class_weight('balanced')` — upweights rare classes

**Say:**
> "A critical preprocessing decision: we use ResNet50's `preprocess_input` function, which subtracts ImageNet's channel means from each pixel. If we had used the common `rescale=1/255` instead, the frozen base would receive inputs it was never trained on and produce meaningless features. We caught this bug early — it was keeping accuracy at random-chance levels.
>
> For augmentation, every choice is justified: horizontal flips because a Scandinavian room is still Scandinavian when mirrored. Brightness variation to simulate different lighting conditions. Zoom to simulate different shooting distances. These increase effective training data variety without introducing unrealistic transformations.
>
> We computed class weights using sklearn's balanced mode — rare classes get upweighted so the model penalises mistakes on them proportionally."

---

## Slide 8 — Training Strategy & Overfitting Mitigation (1 min)
**Speaker: Person 3**

**On slide:**
- Two-phase training diagram:
  ```
  Phase 4a — Head training
  lr = 1e-4 | Base FROZEN | 20 epochs
  Result: val_accuracy 31.9%

  Phase 4b — Fine-tuning
  lr = 1e-5 | Top 30 layers unfrozen | 6 epochs
  Result: val_accuracy 33.4%
  ```
- Overfitting prevention checklist:
  - ✅ Dropout(0.5)
  - ✅ Data augmentation
  - ✅ EarlyStopping (patience=5)
  - ✅ ReduceLROnPlateau (factor=0.5)
  - ✅ BatchNormalization
  - ✅ Class weights

**Say:**
> "We train in two phases — this is critical. Phase 4a trains only the custom head with the ResNet base frozen. If we unfroze the base immediately with a random head, the chaotic gradients would destroy ResNet's pretrained knowledge. Only once the head stabilises do we enter Phase 4b — fine-tuning the top 30 ResNet layers at a learning rate 10x lower than the head. We nudge the pretrained weights, we don't overwrite them.
>
> Overfitting prevention is layered: Dropout randomly deactivates 50% of neurons per training step. EarlyStopping halts training when validation loss stops improving. ReduceLROnPlateau halves the learning rate when progress stalls. Together, these keep the model generalising rather than memorising."

---

## Slide 9 — Evaluation Results (1 min)
**Speaker: Person 4**

**On slide:**
- Big headline numbers:
  - **Top-5 Accuracy: 73.1%** ← headline
  - Top-1 Accuracy: 34.1% (6× better than random chance of 5.3%)
  - Macro F1: 0.341
- F1 bar chart image (from `f1_per_class.png`)
- Two-column insight:
  - **Model does well:** Industrial (0.447), Eclectic (0.441), Rustic (0.431) — visually distinctive
  - **Model struggles:** French-Country (0.223), Contemporary (0.227), Transitional (0.234) — visually ambiguous

**Say:**
> "Our headline metric is Top-5 accuracy: 73.1%. This means the correct interior style appears in the model's top 5 predictions 3 out of every 4 times. For 19 categories, that's a strong result.
>
> The per-class F1 chart tells the real story. The model excels at visually distinctive styles — Industrial, with its exposed metal and concrete, scores highest. It struggles with styles that genuinely overlap — Contemporary, Transitional, and Modern all share clean lines and neutral palettes. Even professional interior designers disagree on these labels.
>
> This is not a model failure — it reflects the inherent ambiguity in interior design classification. And critically, our app averages predictions across 3 to 5 images, which smooths out this individual image uncertainty."

---

## Slide 10 — Live Demo (3 min)
**Speaker: Person 4 — runs the demo**

**On slide:**
- Just the Ambiance app open on screen
- Have these ready before the presentation:
  - 4–5 mood board images (pre-selected, clear style — e.g. Scandinavian)
  - 2 product images: one strong match, one clear mismatch

**Demo script:**
> "Let me show you this working in real time — no pre-computed results.
>
> [Upload 4 mood board images]
> The model is running inference on each image right now, extracting a style vector, and averaging them. You can see our style profile — dominant style is [X] at [Y]%.
>
> [Upload matching product]
> Here's a [product type] — [X]% match. Strong fit.
>
> [Upload mismatching product]
> And here's [different product] — [X]% match. The model correctly identifies the style clash.
>
> This runs in real time, on any image, with no pre-loading of results."

**Backup plan if internet/app fails:**
- Have screenshots of a successful run ready as a final slide

---

## Slide 11 — Business Model & ROI (1 min)
**Speaker: Person 1**

**On slide:**
- ROI calculation:
  ```
  Industry average furniture return rate:  15–30%
  Average order value:                     £500
  Target: reduce returns by 50% with Ambiance

  For a retailer with 10,000 orders/month:
    Returns saved:    750–1,500 orders/month
    Cost saved:       £375,000–£750,000/month
    Annual saving:    £4.5M–£9M
  ```
- Revenue model:
  - B2B API subscription for retailers (IKEA, Wayfair, H&M Home)
  - Consumer app: freemium

**Say:**
> "The business case is clear. Furniture returns cost retailers 15 to 30 percent of revenue — not just the refund, but reverse logistics, restocking, and lost customer trust.
>
> A retailer processing 10,000 orders a month, with Ambiance reducing returns by 50%, saves between £375,000 and £750,000 per month. That's £4.5 to 9 million annually. The API subscription price is a rounding error by comparison.
>
> Our target customers are furniture platforms — IKEA, Wayfair, H&M Home — who embed Ambiance as a 'check your match' feature at the point of purchase. The consumer version follows as a freemium app."

---

## Slide 12 — Future Roadmap (30 sec)
**Speaker: Person 1**

**On slide:**
- V2: Siamese Network — end-to-end similarity learning, removes classification step
- V2: Pinterest API — import boards directly
- V3: "Shop the match" — when score is low, suggest products that do match
- V3: Room photo input — photograph your actual room, not inspo images

**Say:**
> "Version 2 replaces our classification-plus-cosine-similarity pipeline with a Siamese network — a single model trained directly on image pairs to learn similarity end-to-end. We also plan Pinterest API integration so users don't need to download images manually.
>
> Version 3 adds 'Shop the match' — when a product scores low, we don't just say no, we suggest alternatives that do match. That's the full retail integration."

---

## Slide 13 — Summary & Thank You (30 sec)
**Speaker: Person 2**

**On slide:**
- Three bullets:
  - **The problem:** 15–30% furniture returns due to style mismatch
  - **The solution:** CNN-powered aesthetic decoding + real-time product validation
  - **The result:** 73.1% top-5 accuracy across 19 interior design styles, working live demo
- "Thank you — questions?"

**Say:**
> "To summarise: furniture style mismatch is a multi-billion pound problem. Ambiance solves it with a ResNet50 CNN that decodes your aesthetic from a mood board and validates products against it in real time. 73% top-5 accuracy across 19 style categories, running live.
>
> Thank you. We're happy to take questions."

---

## Q&A Prep (2 min buffer)

**Q: Why CNN and not a simpler model?**
> CNNs learn spatial hierarchies — edges, textures, patterns, styles. A Random Forest or plain ANN treats each pixel independently and loses all spatial context. For image classification, CNN is the only sensible architecture.

**Q: Why ResNet50 specifically?**
> Residual connections solve the vanishing gradient problem — gradients flow through shortcut paths so early layers still learn in a 50-layer network. VGG16 is simpler but slower and lacks residuals. EfficientNet is more parameter-efficient but harder to explain.

**Q: Why not train from scratch?**
> We have ~780 images per class. Training ResNet50 from scratch requires millions of images and weeks of GPU time. Transfer learning lets us leverage 1.2M ImageNet images already learned. Fine-tuning the top 30 layers adapts it to interior design without destroying that base knowledge.

**Q: Why is accuracy only 34%?**
> 34% across 19 classes is 6× better than random chance. More importantly, our Top-5 accuracy is 73% — the correct style is in our top 5 predictions 3 out of 4 times. The classes that are hardest — Contemporary, Transitional, Modern — are also the ones that human designers argue about. The confusion matrix shows the model understands style families; it just struggles at fine-grained separation within them.

**Q: How do you prevent overfitting?**
> Five layers: Dropout(0.5), data augmentation, EarlyStopping, ReduceLROnPlateau, and BatchNormalization. We also use class weights to prevent the model from biasing toward majority classes.

**Q: Couldn't someone just use IKEA Kreativ?**
> IKEA Kreativ visualises furniture in your room — it doesn't decode your aesthetic profile from a mood board or validate arbitrary product photos against it. These tools solve different problems and are complementary.

---

## Timing Summary

| Slide | Speaker | Time |
|---|---|---|
| 1. Title | Person 1 | 0:30 |
| 2. Hook | Person 1 | 1:00 |
| 3. Problem | Person 2 | 1:00 |
| 4. Solution | Person 2 | 1:00 |
| 5. Market context | Person 2 | 0:30 |
| 6. Architecture | Person 3 | 1:30 |
| 7. Data pipeline | Person 3 | 1:00 |
| 8. Training strategy | Person 3 | 1:00 |
| 9. Evaluation | Person 4 | 1:00 |
| 10. Live demo | Person 4 | 3:00 |
| 11. Business ROI | Person 1 | 1:00 |
| 12. Roadmap | Person 1 | 0:30 |
| 13. Summary | Person 2 | 0:30 |
| Q&A | All | 2:00 |
| **Total** | | **15:30** |

> Aim for 13 minutes of content so Q&A has breathing room. If you're running long, cut Slide 12 (roadmap) — it's the least rubric-critical.

---

## Rubric Coverage Check

| Pillar | Weight | Covered by |
|---|---|---|
| Business Use Case | 20% | Slides 3, 4, 5, 11 — quantified ROI, clear market gap, named customers |
| Technical Depth | 25% | Slides 6, 7, 8, 9 — justified architecture, hyperparameters, preprocessing, eval metrics |
| MVP Integration | 25% | Slide 10 — live real-time demo, no pre-computed results |
| Presentation | 20% | All slides — each person speaks, clear structure, executive pitch tone |
| Live Demo | 10% | Slide 10 — working app, backup screenshots ready |
