import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

CLASS_NAMES = [
    'asian', 'coastal', 'contemporary', 'craftsman', 'eclectic',
    'farmhouse', 'french-country', 'industrial', 'mediterranean',
    'mid-century-modern', 'modern', 'rustic', 'scandinavian',
    'shabby-chic-style', 'southwestern', 'traditional', 'transitional',
    'tropical', 'victorian'
]

IMG_SIZE = (224, 224)


def load_model(model_path: str):
    """Load the trained model from a .h5 file."""
    return tf.keras.models.load_model(model_path)


def _preprocess(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a preprocessed array ready for ResNet50."""
    img = image.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)


def get_style_vector(model, image: Image.Image) -> np.ndarray:
    """Run the model on a single PIL image. Returns a (19,) probability vector."""
    x = _preprocess(image)
    return model.predict(x, verbose=0)[0]


def get_moodboard_vector(model, images: list) -> np.ndarray:
    """
    Average style vectors from 3-5 images to build a mood board profile.
    Averaging smooths out noise from any single ambiguous image.
    Returns a (19,) vector representing the overall aesthetic.
    """
    vectors = [get_style_vector(model, img) for img in images]
    return np.mean(vectors, axis=0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def match_score(moodboard_vector: np.ndarray, product_vector: np.ndarray) -> float:
    """
    Compute how well a product matches a mood board.
    Uses cosine similarity between the two style vectors, scaled to 0-100%.
    """
    similarity = _cosine_similarity(moodboard_vector, product_vector)
    return round(similarity * 100, 1)


def top_styles(vector: np.ndarray, n: int = 3) -> list:
    """
    Return the top-N styles and their probabilities from a style vector.
    Returns a list of (style_name, probability_percent) tuples.
    """
    top_indices = np.argsort(vector)[-n:][::-1]
    return [(CLASS_NAMES[i], round(float(vector[i]) * 100, 1)) for i in top_indices]
