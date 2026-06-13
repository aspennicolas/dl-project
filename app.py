import os
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from src.predictor import load_model, get_moodboard_vector, get_style_vector, match_score, top_styles

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ambiance_model.h5')

st.set_page_config(page_title="Ambiance", layout="wide")

# ── Load model (cached — only runs once per session) ──────────────────────────
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Download ambiance_model.h5 from Google Drive and place it in the models/ folder.")
        st.stop()
    return load_model(MODEL_PATH)

model = get_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Ambiance")
st.caption("Upload inspiration images to discover your interior style, then check how well a product matches your aesthetic.")
st.divider()

# ── Step 1: Mood board ────────────────────────────────────────────────────────
st.header("Step 1 — Upload your mood board")
st.caption("Choose 3–5 images of rooms or spaces that match the aesthetic you want.")

uploaded_mood = st.file_uploader(
    "Upload 3–5 inspiration images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="moodboard"
)

moodboard_vector = None

if uploaded_mood:
    if len(uploaded_mood) < 3:
        st.warning(f"Please upload at least 3 images ({len(uploaded_mood)} uploaded so far).")
    elif len(uploaded_mood) > 5:
        st.warning("Please upload no more than 5 images.")
    else:
        # Display uploaded images
        cols = st.columns(len(uploaded_mood))
        images = []
        for col, f in zip(cols, uploaded_mood):
            img = Image.open(f)
            images.append(img)
            col.image(img, use_container_width=True)

        with st.spinner("Analysing your style..."):
            moodboard_vector = get_moodboard_vector(model, images)

        # Style profile
        st.subheader("Your style profile")
        top5 = top_styles(moodboard_vector, n=5)
        names  = [t[0].replace('-', ' ').title() for t in top5]
        scores = [t[1] for t in top5]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(names[::-1], scores[::-1], color='steelblue')
        ax.set_xlabel('Probability (%)')
        ax.set_xlim(0, 100)
        ax.set_title('Top 5 detected styles')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        dominant_name  = names[0]
        dominant_score = scores[0]
        st.markdown(f"**Dominant style:** {dominant_name} ({dominant_score:.0f}%)")

# ── Step 2: Product match ─────────────────────────────────────────────────────
if moodboard_vector is not None:
    st.divider()
    st.header("Step 2 — Check a product")
    st.caption("Upload an image of a furniture or decor item to see how well it fits your aesthetic.")

    uploaded_product = st.file_uploader(
        "Upload a product image",
        type=["jpg", "jpeg", "png"],
        key="product"
    )

    if uploaded_product:
        product_img = Image.open(uploaded_product)

        col_img, col_result = st.columns([1, 2])
        col_img.image(product_img, caption="Product", use_container_width=True)

        with st.spinner("Checking match..."):
            product_vector = get_style_vector(model, product_img)
            score = match_score(moodboard_vector, product_vector)
            product_top = top_styles(product_vector, n=3)

        with col_result:
            st.metric("Match Score", f"{score}%")

            if score >= 80:
                st.success("Strong match ✅ — This product fits your aesthetic well.")
            elif score >= 65:
                st.warning("Good match — Some style differences but generally compatible.")
            else:
                st.error("Poor match ❌ — This product may clash with your interior style.")

            st.markdown(f"**Product style:** {product_top[0][0].replace('-', ' ').title()} ({product_top[0][1]:.0f}%)")
            st.markdown(f"**Your dominant style:** {names[0]}")

            # Top 3 product styles
            st.markdown("**Product style breakdown:**")
            for style, prob in product_top:
                st.markdown(f"- {style.replace('-', ' ').title()}: {prob:.0f}%")
