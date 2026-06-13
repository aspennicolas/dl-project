"""
preprocess.py — Data loading and augmentation pipeline.

Builds three Keras ImageDataGenerators:
  - train_gen : augmented images from dataset_train (85% of that folder)
  - val_gen   : clean images from dataset_train (held-out 15%)
  - test_gen  : clean images from dataset_test  (never seen during training)
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# ── Paths ──────────────────────────────────────────────────────────────────────
# os.path.abspath(__file__) gives the full path to this file (src/preprocess.py).
# We go up two levels (..) to reach the project root, then descend into data/raw.
# This means the script works correctly no matter what folder you run it from.
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR  = os.path.join(BASE_DIR, "data", "raw", "dataset_train")
TEST_DIR   = os.path.join(BASE_DIR, "data", "raw", "dataset_test")

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)  # ResNet50 was designed for 224×224 — all images are resized to this
BATCH_SIZE = 32          # 32 images processed per gradient update; standard starting point
VAL_SPLIT  = 0.15        # 15% of dataset_train held out for validation (never trained on)
SEED       = 42          # Fixed seed so the train/val split is the same every run


def get_generators(batch_size: int = BATCH_SIZE):
    """
    Create and return (train_gen, val_gen, test_gen).

    Call this from your training notebook or script:
        train_gen, val_gen, test_gen = get_generators()
    """

    # ── Training generator — WITH augmentation ─────────────────────────────────
    # Augmentation creates modified copies of existing images on-the-fly during
    # training. The model never sees the exact same image twice, which forces it
    # to learn the actual features of a style rather than memorising specific photos.
    # Each parameter below mimics a realistic variation in interior photography:
    # ResNet50 was trained with channel-mean subtraction (not simple /255 scaling).
    # preprocess_input must match what was used at training time — the frozen base
    # produces meaningful features only when it sees inputs in the same scale.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,         # Mirror the image left-right.
                                      # A minimalist room flipped is still minimalist.
        rotation_range=20,            # Rotate up to ±20 degrees.
                                      # Simulates a photographer standing at a slight angle.
        brightness_range=[0.8, 1.2],  # Randomly darken or brighten by up to 20%.
                                      # Accounts for different lighting conditions per photo.
        zoom_range=0.1,               # Zoom in/out up to 10%.
                                      # Simulates different distances from the room.
        validation_split=VAL_SPLIT    # Tell Keras to reserve 15% of files for validation.
                                      # The split is file-order based (not random), so we
                                      # use a fixed seed below to keep it reproducible.
    )

    # ── Evaluation generator — NO augmentation ─────────────────────────────────
    # Validation and test images must NOT be augmented.
    # We evaluate the model on clean, unmodified images so the metrics reflect
    # how the model actually performs on real-world inputs.
    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # ── Training split ─────────────────────────────────────────────────────────
    # flow_from_directory scans subfolders inside TRAIN_DIR.
    # Each subfolder name becomes a class label (e.g. /industrial → "industrial").
    # subset="training" tells it to use the 85% portion defined by VAL_SPLIT above.
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,         # Resize every image to 224×224
        batch_size=batch_size,
        class_mode="categorical",     # Returns one-hot encoded labels, e.g. [0,0,1,0,...]
                                      # Required because we have 19 classes + softmax output
        subset="training",
        seed=SEED,
        shuffle=True                  # Shuffle order each epoch so the model doesn't learn
                                      # patterns based on which class comes first in a batch
    )

    # ── Validation split ───────────────────────────────────────────────────────
    # Uses the SAME datagen (train_datagen) and SAME seed as train_gen so Keras
    # assigns exactly the same files to each split every time.
    # If we used a different seed, images could appear in BOTH splits — data leakage.
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",          # The held-out 15%
        seed=SEED,
        shuffle=False                 # Keep validation order fixed — not needed for metrics
    )

    # ── Test generator ─────────────────────────────────────────────────────────
    # Completely separate folder — these images are never seen during training or
    # validation. This gives us an unbiased final performance measurement.
    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False                 # Order must stay fixed so predictions align with
                                      # true labels when we compute the confusion matrix
    )

    return train_gen, val_gen, test_gen
