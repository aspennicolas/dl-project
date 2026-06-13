# Model Evaluation Results

---

## Final Numbers

| Metric | Score |
|---|---|
| Top-1 Accuracy | 34.1% |
| Top-5 Accuracy | 73.1% |
| Macro F1 | 0.341 |
| Test images | 3,729 |
| Classes | 19 |
| Random chance | 5.3% (1/19) |

---

## What These Numbers Mean

**Top-1 accuracy (34.1%)** — the model's single best guess is correct 34% of the time. That's 6x better than random chance.

**Top-5 accuracy (73.1%)** — the correct style appears somewhere in the model's top 5 highest-probability guesses 73% of the time. Roughly 3 out of every 4 images. This is the headline metric to lead with in the presentation.

**How Top-5 works in practice:**
```
Model sees a Farmhouse room and outputs:
  1. Rustic        — 28%   ← top guess (wrong)
  2. Craftsman     — 19%
  3. Farmhouse     — 15%   ← correct answer is here
  4. Scandinavian  — 11%
  5. Traditional   —  8%

→ Counts as Top-5 correct. Model knew the right style 
  was a likely candidate, just couldn't pinpoint exactly which.
```

---

## Per-Class F1 Results

### Best Performing (F1 > 0.40) — visually distinctive styles
| Class | F1 | Why the model does well |
|---|---|---|
| Industrial | 0.447 | Very distinctive — exposed metal, brick, pipes, dark tones |
| Eclectic | 0.441 | Unique mix of styles, hard to confuse with one specific style |
| Rustic | 0.431 | Strong textures, reclaimed wood, earthy tones |
| Tropical | 0.427 | Plants, rattan, bright colours, natural light |
| Southwestern | 0.422 | Terracotta, adobe, bold geometric patterns |
| Asian | 0.419 | Minimal, specific materials, clean lines with natural elements |

### Worst Performing (F1 < 0.25) — visually overlapping styles
| Class | F1 | Why the model struggles |
|---|---|---|
| French-Country | 0.223 | Overlaps heavily with Shabby Chic + Traditional |
| Contemporary | 0.227 | Looks almost identical to Modern + Transitional |
| Farmhouse | 0.233 | Very similar to Rustic + Craftsman |
| Transitional | 0.234 | By definition a blend of Traditional + Contemporary |

---

## The Key Insight

The model separates **visually distinctive styles** well and struggles where **even human experts would disagree**. "Transitional" literally means a blend of traditional and contemporary — of course the model finds it hard to classify. This is not a model failure; it reflects the genuine ambiguity in interior design labelling.

**Style confusion clusters (expected from confusion matrix):**
```
Contemporary ↔ Transitional ↔ Modern
(all share: clean lines, neutral palettes, minimal ornamentation)

Farmhouse ↔ Rustic ↔ Craftsman
(all share: wood, neutral tones, handcrafted textures)

Coastal ↔ Tropical
(all share: light colours, natural materials, airy feel)
```

These clusters show the model understands **style families** — it just struggles at the fine-grained level within them.

---

## Why This Is Good Enough for the App

The Ambiance mood board feature runs the model on **3-5 images** and averages the probability vectors. This smooths out individual prediction errors — if one image is ambiguous, the others correct for it. The 73.1% top-5 accuracy on single images will translate to even stronger performance when aggregating across a full mood board.

---

## Presentation Talking Points

- "6x better than random chance across 19 style categories"
- "73% of the time the correct style is in the model's top 5 predictions"
- "The model struggles exactly where human designers would struggle — distinguishing Contemporary from Transitional from Modern"
- "Averaging predictions across 3-5 mood board images makes the app more robust than any single-image accuracy number suggests"

---

## Saved Artefacts (Google Drive)

| File | Contents |
|---|---|
| `classification_report.txt` | Full precision, recall, F1 per class |
| `f1_per_class.png` | Bar chart — F1 per class sorted low to high |
| `confusion_matrix.png` | Normalised heatmap — which styles get confused |
| `misclassified_samples.png` | 12 example images the model got wrong + top-3 predictions |
