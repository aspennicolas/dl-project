"""
model.py — Builds and compiles the Ambiance classification model.

Architecture: ResNet50 (frozen, pretrained on ImageNet) + custom ANN head
  Input: 224×224×3 image
  Output: probability vector over 19 interior design style classes

Call build_model() to get a compiled, ready-to-train Keras model.
"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_CLASSES   = 19            # One output neuron per interior design style category
IMG_SHAPE     = (224, 224, 3) # ResNet50 was designed for 224×224 RGB images
LEARNING_RATE = 1e-4          # Conservative rate for fine-tuning — see justification below


def build_model(num_classes: int = NUM_CLASSES, learning_rate: float = LEARNING_RATE):
    """
    Build and compile the Ambiance model.

    Returns (model, base_model) — model is the full compiled model ready for .fit();
    base_model is the ResNet50 reference needed for fine-tuning later.
    """

    # ── Step 1: Load the pretrained ResNet50 base ──────────────────────────────
    # ResNet50 was pretrained on ImageNet (1.2M images, 1000 classes) — it already knows edges, textures, and patterns.
    # include_top=False removes the original 1000-class output layer so we can attach our own 19-class head.
    # input_shape must match the 224×224 we set in preprocess.py.
    # Why ResNet50? It uses residual connections — shortcuts that let gradients flow through deep layers without vanishing,
    # solving a problem that caused earlier networks (VGG, AlexNet) to stop learning in their early layers.
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SHAPE
    )

    # ── Step 2: Freeze the base model ─────────────────────────────────────────
    # Locks all ~25M ResNet50 weights so they don't update during training.
    # We have ~780 images per class — too few to retrain a 50-layer CNN from scratch without catastrophic overfitting.
    # Freezing preserves the rich ImageNet features and lets us train only the small custom head.
    base_model.trainable = False

    # ── Step 3: Build the custom ANN head ──────────────────────────────────────
    # ResNet50 outputs a (7, 7, 2048) feature map. We need to reduce this to a 1D vector before Dense layers.
    x = base_model.output

    # GlobalAveragePooling2D averages each of the 2048 channels across the 7×7 grid: (7, 7, 2048) → (2048,).
    # Preferred over Flatten because Flatten produces 100,352 inputs (7×7×2048), inflating parameter count and overfitting risk.
    # GAP has zero trainable parameters of its own.
    x = GlobalAveragePooling2D()(x)

    # Dense(512) learns which combinations of the 2048 ResNet features predict each interior style.
    # 512 sits between the 2048 input and 19 output — large enough for complexity, small enough to train fast.
    # ReLU outputs max(0, x), adding non-linearity so the layer can learn patterns beyond simple linear combinations.
    # Dense without inline activation so BatchNorm sees the raw pre-activation values —
    # the standard order is Dense (linear) → BatchNorm → activation → Dropout.
    # Placing BN before activation means it normalises inputs into the non-linearity,
    # which stabilises training. Placing it after Dropout would corrupt BN's running
    # statistics because Dropout zeros 50% of values during training but not at inference,
    # causing a train/test mismatch in the normalisation.
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Output layer: 19 neurons (one per style class). Softmax converts raw scores to probabilities summing to 1.0.
    # e.g. [0.60, 0.05, 0.10, ...] = "60% Scandinavian, 10% Minimalist, ..."
    # This probability vector is also what feeds the cosine similarity calculation in Phase 5.
    output = Dense(num_classes, activation='softmax')(x)

    # ── Step 4: Assemble the Model ─────────────────────────────────────────────
    # Wires the frozen ResNet50 base and trainable head into one Keras Model object.
    model = Model(inputs=base_model.input, outputs=output)

    # ── Step 5: Compile ────────────────────────────────────────────────────────
    # Adam adapts the learning rate per parameter based on gradient history — better than plain SGD for fine-tuning.
    # 1e-4 is 10× lower than Adam's default (1e-3): smaller updates preserve the pretrained ResNet weights.
    # categorical_crossentropy is the standard loss for multi-class classification with one-hot labels.
    # accuracy tracks the fraction of correctly classified images per epoch.
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def model_summary():
    """Print a summary of the model architecture — useful for sanity-checking layer shapes."""
    model, _ = build_model()
    model.summary()
    trainable     = sum(p.numpy().size for p in model.trainable_weights)
    non_trainable = sum(p.numpy().size for p in model.non_trainable_weights)
    print(f"\nTrainable parameters:     {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
    print(f"(Only the {trainable:,} trainable params update during training — the frozen ResNet50 base does not.)")


if __name__ == "__main__":
    model_summary()
