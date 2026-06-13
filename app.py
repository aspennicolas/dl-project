import os
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from src.predictor import load_model, get_moodboard_vector, get_style_vector, match_score, top_styles

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ambiance_model.h5')

st.set_page_config(
    page_title="Ambiance",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Inter:wght@300;400;500&display=swap');

/* Background */
.stApp { background-color: #F9F7F5; }

/* Remove default top padding */
.block-container {
    padding-top: 1rem !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #2C2C2C;
}

/* Hero */
.hero {
    text-align: center;
    padding: 1rem 0 1rem 0;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem;
    font-weight: 500;
    color: #1A1A1A;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #888888;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Section headers */
.section-tag {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #A8987A;
    margin-bottom: 0.3rem;
}
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 500;
    color: #1A1A1A;
    margin-bottom: 0.3rem;
}
.section-desc {
    font-size: 0.88rem;
    color: #888888;
    font-weight: 300;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

/* Thin divider */
.thin-divider {
    border: none;
    border-top: 1px solid #E8E4E0;
    margin: 2.5rem 0;
}

/* Score card */
.score-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.score-number {
    font-family: 'Cormorant Garamond', serif;
    font-size: 5rem;
    font-weight: 500;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.score-verdict {
    font-size: 0.75rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 0.5rem;
    font-weight: 500;
}
.score-green { color: #5C8C6A; }
.score-amber { color: #B8924A; }
.score-red   { color: #A85C5C; }

/* Style pill */
.style-pill {
    display: inline-block;
    background: #EAF0EA;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    color: #5A7A5A;
    margin: 0.2rem;
    font-weight: 400;
}

/* Instruction box */
.instruction-box {
    background: #F4F8F4;
    border-left: 3px solid #A8C4A2;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1.5rem;
    font-size: 0.875rem;
    font-family: 'Inter', sans-serif;
    color: #666666;
    line-height: 1.7;
}

/* Dominant style callout */
.dominant-callout {
    background: #FFFFFF;
    border-left: 3px solid #C4B4D4;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.5rem;
    margin-top: 1.2rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.dominant-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #AAAAAA;
    margin-bottom: 0.2rem;
}
.dominant-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    color: #1A1A1A;
}
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Place ambiance_model.h5 in the models/ folder.")
        st.stop()
    return load_model(MODEL_PATH)

model = get_model()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Ambiance</div>
    <div class="hero-sub">Decode your interior aesthetic. Validate every purchase.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)


# ── Step 1 ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-tag">Step 01</div>
<div class="section-title">Build your mood board</div>
<div class="section-desc">Upload 3 to 5 images of interiors that reflect the aesthetic you are drawn to.<br>
These can be rooms, spaces, or any image that captures the feeling you want in your home.</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="instruction-box">
    <strong>Tips for best results:</strong><br>
    • Use clear room photos, not product shots<br>
    • Mix different angles of the same style<br>
    • 3–5 images gives the most accurate profile
</div>
""", unsafe_allow_html=True)

uploaded_mood = st.file_uploader(
    "Choose your inspiration images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="moodboard"
)

moodboard_vector = None
names = []

if uploaded_mood:
    if len(uploaded_mood) < 3:
        st.info(f"{len(uploaded_mood)} image{'s' if len(uploaded_mood) > 1 else ''} uploaded — please add at least {3 - len(uploaded_mood)} more.")
    elif len(uploaded_mood) > 5:
        st.warning("Please upload no more than 5 images for the most accurate results.")
    else:
        # Image grid
        cols = st.columns(len(uploaded_mood))
        images = []
        for col, f in zip(cols, uploaded_mood):
            img = Image.open(f)
            images.append(img)
            col.image(img, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner("Analysing your aesthetic..."):
            moodboard_vector = get_moodboard_vector(model, images)

        # Style profile chart
        top5  = top_styles(moodboard_vector, n=5)
        names  = [t[0].replace('-', ' ').title() for t in top5]
        scores = [t[1] for t in top5]

        fig, ax = plt.subplots(figsize=(9, 3))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')

        bar_color = '#C4D4BC'
        bars = ax.barh(names[::-1], scores[::-1], color=bar_color, height=0.5)
        bars[-1].set_color('#7A9E72')  # Highlight top style

        ax.set_xlim(0, max(scores) * 1.3)
        ax.set_xlabel('Probability (%)', fontsize=9, color='#999999', labelpad=8)
        ax.tick_params(axis='y', labelsize=10, colors='#333333')
        ax.tick_params(axis='x', labelsize=8, colors='#AAAAAA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#EEEEEE')
        ax.xaxis.grid(True, color='#F0F0F0', linewidth=0.5)
        ax.set_axisbelow(True)

        for bar, score in zip(bars[::-1], scores):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{score:.0f}%', va='center', ha='left', fontsize=9, color='#888888')

        plt.title('Your Style Profile', fontsize=11, color='#333333',
                  fontweight='normal', pad=12, loc='left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Dominant style callout
        st.markdown(f"""
        <div class="dominant-callout">
            <div class="dominant-label">Dominant Style</div>
            <div class="dominant-value">{names[0]}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Step 2 ────────────────────────────────────────────────────────────────────
if moodboard_vector is not None:
    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-tag">Step 02</div>
    <div class="section-title">Check a product</div>
    <div class="section-desc">Upload an image of a furniture or décor item you are considering.<br>
    Ambiance will tell you how well it fits the aesthetic you defined above.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="instruction-box">
        <strong>Works best with:</strong><br>
        • Product photos on a plain or room background<br>
        • Single items (sofa, lamp, rug, artwork)<br>
        • Standard product images from retailer websites
    </div>
    """, unsafe_allow_html=True)

    uploaded_product = st.file_uploader(
        "Choose a product image",
        type=["jpg", "jpeg", "png"],
        key="product"
    )

    if uploaded_product:
        product_img = Image.open(uploaded_product)

        with st.spinner("Computing match score..."):
            product_vector = get_style_vector(model, product_img)
            score          = match_score(moodboard_vector, product_vector)
            product_top    = top_styles(product_vector, n=3)

        col_img, col_gap, col_result = st.columns([2, 0.2, 3])

        with col_img:
            st.image(product_img, use_container_width=True)
            st.markdown(f"""
            <div style="text-align:center; margin-top:0.5rem;">
                {'  '.join([f'<span class="style-pill">{s.replace("-"," ").title()}</span>'
                            for s, _ in product_top])}
            </div>
            """, unsafe_allow_html=True)

        with col_result:
            if score >= 80:
                color_class = "score-green"
                verdict     = "Strong Match"
                desc        = f"This piece aligns closely with your {names[0]} aesthetic. A confident buy."
            elif score >= 65:
                color_class = "score-amber"
                verdict     = "Good Match"
                desc        = f"This piece shares some qualities with your style. Minor differences may be noticeable."
            else:
                color_class = "score-red"
                verdict     = "Poor Match"
                desc        = f"This piece may clash with your {names[0]} aesthetic. Consider alternatives."

            st.markdown(f"""
            <div class="score-card">
                <div class="score-number {color_class}">{score:.0f}%</div>
                <div class="score-verdict {color_class}">{verdict}</div>
                <div style="margin-top:1.2rem; font-size:0.88rem; color:#666666; line-height:1.6;">
                    {desc}
                </div>
                <div style="margin-top:1.5rem; border-top:1px solid #F0F0F0; padding-top:1.2rem;">
                    <div style="font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
                                color:#AAAAAA; margin-bottom:0.5rem;">Style Comparison</div>
                    <div style="font-size:0.88rem; color:#555555;">
                        <span style="color:#8A7A6E; font-weight:500;">Your style</span>
                        &nbsp;·&nbsp; {names[0]}
                    </div>
                    <div style="font-size:0.88rem; color:#555555; margin-top:0.3rem;">
                        <span style="color:#8A7A6E; font-weight:500;">Product style</span>
                        &nbsp;·&nbsp; {product_top[0][0].replace('-', ' ').title()}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#BBBBBB; font-size:0.78rem; letter-spacing:1px;
                padding-bottom:2rem;">
        AMBIANCE &nbsp;·&nbsp; IE University Deep Learning Project
    </div>
    """, unsafe_allow_html=True)
