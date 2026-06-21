
# =============================================================
#  AgroScan — Agricultural Pest Classification
#  Streamlit Application  |  5 Pages
#  Graduation Project 2026
# =============================================================

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import urllib.request

# ─────────────────────────────────────────────────────────────
#  CONFIG  —  edit these if needed
# ─────────────────────────────────────────────────────────────
# Local path inside the repo. Works on Streamlit Community Cloud,
# Hugging Face Spaces, Render, or any container deployment.
MODEL_PATH  = os.environ.get("MODEL_PATH", "models/best_wavelet_resnet_model.pth")

# Optional: if the .pth file is too large to commit to GitHub (>100MB),
# host it elsewhere (Hugging Face Hub, S3, a GitHub Release asset, etc.)
# and set MODEL_URL as a Streamlit secret or environment variable.
# The app will download it once into MODEL_PATH on first run.
MODEL_URL   = os.environ.get("MODEL_URL", st.secrets.get("MODEL_URL", "") if hasattr(st, "secrets") else "")

NUM_CLASSES = 72
IMG_SIZE    = 380                          # EfficientNet-B4 native
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_model_downloaded():
    """Download the model file from MODEL_URL if it isn't already on disk."""
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL:
        return
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    with st.spinner(f"Downloading model weights from {MODEL_URL} …"):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroScan | Pest Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&
family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f0a 0%, #162b10 60%, #1a3314 100%);
    border-right: 1px solid #2d5a1e;
}
[data-testid="stSidebar"] * { color: #e8f5e0 !important; }

/* Main background */
.main { background-color: #f8faf6; }

/* Cards */
.ag-card {
    background: white;
    border: 1px solid #e0ead8;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.ag-card-green {
    border-left: 5px solid #4caf50;
}
.ag-card-amber {
    border-left: 5px solid #ff9800;
}
.ag-card-blue {
    border-left: 5px solid #2196f3;
}

/* Hero section */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #1b5e20;
    line-height: 1.15;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #4a7c59;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

/* Result badge */
.result-box {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border: 2px solid #4caf50;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}
.pred-name {
    font-size: 1.9rem;
    font-weight: 700;
    color: #1b5e20;
}
.pred-conf {
    font-size: 1.1rem;
    color: #388e3c;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.3rem;
}

/* Class pill */
.pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem;
}

/* Team card */
.team-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    border: 1px solid #e0ead8;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    transition: transform 0.2s;
}
.team-card:hover { transform: translateY(-3px); }
.avatar {
    width: 72px; height: 72px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem;
    margin: 0 auto 0.8rem;
}
.member-name { font-size: 1rem; font-weight: 700; color: #1b5e20; }
.member-role { font-size: 0.8rem; color: #666; margin-top: 0.2rem; }

/* Stat chip */
.stat-box {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border: 1px solid #c8e6c9;
    border-radius: 12px;
    padding: 1.1rem;
    text-align: center;
}
.stat-val { font-size: 1.8rem; font-weight: 800; color: #2e7d32; }
.stat-lbl { font-size: 0.72rem; color: #555; text-transform: uppercase;
             letter-spacing: 0.07em; margin-top: 0.2rem; }

/* Section header */
.sec-hdr {
    font-size: 1.5rem; font-weight: 700; color: #1b5e20;
    border-bottom: 3px solid #4caf50;
    padding-bottom: 0.4rem; margin-bottom: 1.2rem; display: inline-block;
}

/* Buttons */
div.stButton > button {
    background: #2e7d32; color: white; border: none;
    border-radius: 8px; font-weight: 600;
    padding: 0.5rem 1.6rem; width: 100%;
}
div.stButton > button:hover { background: #388e3c; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #4caf50;
    border-radius: 12px;
    background: #f1f8f1;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  CLASS DATA  (72 classes with order and risk)
# ─────────────────────────────────────────────────────────────
CLASS_DATA = [
    ("Acalymma vittatum",              "Coleoptera",   "High"),
    ("Achatina fulica",                "Gastropoda",   "High"),
    ("Alticini",                       "Coleoptera",   "Medium"),
    ("Ampelophaga",                    "Lepidoptera",  "High"),
    ("Anasa tristis",                  "Hemiptera",    "High"),
    ("Aphids",                         "Hemiptera",    "Very High"),
    ("Armyworm butterfly",             "Lepidoptera",  "High"),
    ("Armyworm caterpillar",           "Lepidoptera",  "Very High"),
    ("Aulacophora similis",            "Coleoptera",   "Medium"),
    ("Beet army worm butterfly",       "Lepidoptera",  "High"),
    ("Beet army worm caterpillar",     "Lepidoptera",  "High"),
    ("Beet spot flies butterfly",      "Diptera",      "Medium"),
    ("Beet spot flies caterpillar",    "Diptera",      "Medium"),
    ("Beetle",                         "Coleoptera",   "High"),
    ("Black cutworm butterfly",        "Lepidoptera",  "Medium"),
    ("Black cutworm caterpillar",      "Lepidoptera",  "Very High"),
    ("Blister beetle",                 "Coleoptera",   "Medium"),
    ("Bollworm butterfly",             "Lepidoptera",  "High"),
    ("Bollworm caterpillar",           "Lepidoptera",  "Very High"),
    ("Cabbage army worm",              "Lepidoptera",  "High"),
    ("Cerodonta denticornis butterfly","Diptera",      "Low"),
    ("Cerodonta denticornis caterpillar","Diptera",    "Medium"),
    ("Cicadella viridis",              "Hemiptera",    "Medium"),
    ("Cicadellidae",                   "Hemiptera",    "High"),
    ("Corn borer butterfly",           "Lepidoptera",  "High"),
    ("Corn borer caterpillar",         "Lepidoptera",  "Very High"),
    ("Dermaptera",                     "Dermaptera",   "Low"),
    ("Grasshopper",                    "Orthoptera",   "Very High"),
    ("Grub",                           "Coleoptera",   "High"),
    ("Icerya purchasi Maskell",        "Hemiptera",    "High"),
    ("Large cutworm butterfly",        "Lepidoptera",  "Medium"),
    ("Large cutworm caterpillar",      "Lepidoptera",  "High"),
    ("Legume blister beetle",          "Coleoptera",   "High"),
    ("Leptinotarsa decemlineata",      "Coleoptera",   "Very High"),
    ("Locustoidea",                    "Orthoptera",   "Very High"),
    ("Lycaena delicatula",             "Lepidoptera",  "Low"),
    ("Lytta polita",                   "Coleoptera",   "Medium"),
    ("Mantodea",                       "Mantodea",     "Low"),
    ("Miridae",                        "Hemiptera",    "High"),
    ("Mites",                          "Arachnida",    "Very High"),
    ("Mole cricket",                   "Orthoptera",   "High"),
    ("Mosquito",                       "Diptera",      "Medium"),
    ("Peach borer butterfly",          "Lepidoptera",  "High"),
    ("Peach borer caterpillar",        "Lepidoptera",  "Very High"),
    ("Prodenia litura butterfly",      "Lepidoptera",  "High"),
    ("Prodenia litura caterpillar",    "Lepidoptera",  "Very High"),
    ("Red spider",                     "Arachnida",    "Very High"),
    ("Rhytidodera bowrinii white",     "Coleoptera",   "High"),
    ("Rice leaf roller butterfly",     "Lepidoptera",  "High"),
    ("Rice leaf roller caterpillar",   "Lepidoptera",  "Very High"),
    ("Rice water weevil",              "Coleoptera",   "Very High"),
    ("Salurnis marginella Guerr",      "Hemiptera",    "Medium"),
    ("Sawfly butterfly",               "Hymenoptera",  "Medium"),
    ("Sawfly caterpillar",             "Hymenoptera",  "High"),
    ("Sericaorient alismots chulsky",  "Coleoptera",   "Medium"),
    ("Stem borer butterfly",           "Lepidoptera",  "High"),
    ("Stem borer caterpillar",         "Lepidoptera",  "Very High"),
    ("Tarnished plant bug",            "Hemiptera",    "High"),
    ("Therioaphis maculata Buckton",   "Hemiptera",    "High"),
    ("Thrips",                         "Thysanoptera", "Very High"),
    ("Wireworm butterfly",             "Coleoptera",   "Medium"),
    ("Wireworm caterpillar",           "Coleoptera",   "Very High"),
    ("Xylotrectus",                    "Coleoptera",   "High"),
    ("Yellow cutworm butterfly",       "Lepidoptera",  "Medium"),
    ("Yellow cutworm caterpillar",     "Lepidoptera",  "High"),
    ("Alfalfa plant bug butterfly",    "Hemiptera",    "High"),
    ("Alfalfa plant bug caterpillar",  "Hemiptera",    "High"),
    ("Army worm butterfly",            "Lepidoptera",  "High"),
    ("Army worm caterpillar",          "Lepidoptera",  "Very High"),
    ("Asiatic rice borer butterfly",   "Lepidoptera",  "High"),
    ("Asiatic rice borer caterpillar", "Lepidoptera",  "Very High"),
]
CLASS_NAMES = [c[0] for c in CLASS_DATA]

ORDER_COLORS = {
    "Lepidoptera": "#4caf50",  "Coleoptera":  "#ff9800",
    "Hemiptera":   "#2196f3",  "Orthoptera":  "#9c27b0",
    "Arachnida":   "#f44336",  "Diptera":     "#009688",
    "Hymenoptera": "#cddc39",  "Dermaptera":  "#795548",
    "Gastropoda":  "#00bcd4",  "Thysanoptera":"#e91e63",
    "Mantodea":    "#8bc34a",
}
RISK_COLORS = {
    "Very High": "#f44336", "High": "#ff9800",
    "Medium":    "#4caf50", "Low":  "#2196f3",
}

# ─────────────────────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    """Load EfficientNet-B4 from MODEL_PATH (auto-downloads from MODEL_URL if set)."""
    ensure_model_downloaded()
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at {MODEL_PATH}"
    try:
        model = models.efficientnet_b4(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        )
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle different checkpoint formats
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state:
                    state = state[key]
                    break
        model.load_state_dict(state, strict=False)
        model.to(DEVICE).eval()
        return model, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
#  PREPROCESSING & INFERENCE
# ─────────────────────────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict(model, pil_img, top_k=5):
    """Return top-k (class_name, probability) predictions."""
    tensor = TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs  = torch.softmax(model(tensor), dim=1)[0]
    vals, idxs = probs.topk(top_k)
    return [(CLASS_NAMES[i], float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]


# ─────────────────────────────────────────────────────────────
#  SIDEBAR  NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '''<div style="text-align:center;padding:1.2rem 0 1rem;">'''  +
        '''<div style="font-size:2.6rem;">🌿</div>'''  +
        '''<div style="font-size:1.3rem;font-weight:800;color:#9DC640;">AgroScan</div>'''  +
        '''<div style="font-size:0.72rem;color:#7abd5c;margin-top:0.2rem;">Pest Intelligence System</div>'''  +
        '''</div>''', unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔬 Pest Prediction", "📋 Pest Classes",
         "📖 About Project", "👥 Team"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    # Model status indicator in sidebar
    model_exists = os.path.exists(MODEL_PATH)
    status_color = "#9DC640" if model_exists else "#f44336"
    status_text  = "Model Ready" if model_exists else "Model Not Found"
    st.markdown(
        f'''<div style="font-size:0.78rem;color:{status_color};font-weight:600;">''' +
        f'''● {status_text}</div>''',
        unsafe_allow_html=True
    )
    st.markdown(
        '''<div style="font-size:0.7rem;color:#7abd5c;margin-top:0.3rem;">''' +
        f'''Path: <code>{MODEL_PATH}</code></div>''',
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════
if page == "🏠 Home":

    # Hero
    st.markdown(
        '''<div style="padding:0.5rem 0 1rem;">'''  +
        '''<p class="hero-title">🌿 Agricultural Pest<br>Classification</p>'''  +
        '''<p class="hero-sub">AI-powered pest identification for smarter crop protection</p>'''  +
        '''</div>''', unsafe_allow_html=True
    )

    # Stat chips row
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("72",      "Pest Classes"),
        ("118K+",   "Training Images"),
        ("EffB4",   "Backbone"),
        ("Top-5",   "Accuracy Mode"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
        col.markdown(
            f'''<div class="stat-box">'''  +
            f'''<div class="stat-val">{val}</div>'''  +
            f'''<div class="stat-lbl">{lbl}</div>'''  +
            '''</div>''', unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown('''<div class="sec-hdr">About This System</div>''', unsafe_allow_html=True)
        st.markdown('''<div class="ag-card ag-card-green">'''  +
            '''<p style="font-size:0.95rem;color:#333;line-height:1.8;margin:0;">'''  +
            '''<b>AgroScan</b> is a deep-learning system that automatically identifies '''  +
            '''agricultural pest insects from field photographs. It was trained on a custom '''  +
            '''dataset of <b>118,664 images</b> spanning <b>72 pest species</b>, covering '''  +
            '''the most economically significant insects across major crop systems worldwide.'''  +
            '''</p></div>''', unsafe_allow_html=True)

        st.markdown('''<div class="sec-hdr">How It Works</div>''', unsafe_allow_html=True)
        steps = [
            ("📸", "Upload",    "Take or upload a photo of the suspected pest."),
            ("⚙️", "Preprocess","Image is resized, normalised, and standardised to 380×380."),
            ("🧠", "Inference", "EfficientNet-B4 runs forward pass on GPU/CPU."),
            ("📊", "Result",    "Top predicted species and confidence score are shown."),
        ]
        for icon, title, desc in steps:
            st.markdown(
                f'''<div style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:0.8rem;">'''  +
                f'''<div style="font-size:1.5rem;">{icon}</div>'''  +
                f'''<div><b style="color:#1b5e20;">{title}</b>'''  +
                f'''<p style="font-size:0.85rem;color:#555;margin:0;">{desc}</p></div>'''  +
                '''</div>''', unsafe_allow_html=True
            )

    with col_right:
        st.markdown('''<div class="sec-hdr">Project Objectives</div>''', unsafe_allow_html=True)
        objectives = [
            "Develop an accurate AI model for 72-class agricultural pest identification",
            "Build a clean, deduplicated training dataset from 135K+ raw images",
            "Apply transfer learning (EfficientNet-B4 + ImageNet weights)",
            "Handle class imbalance with weighted loss and stratified sampling",
            "Deploy a user-friendly web interface for real-world field use",
            "Contribute to integrated pest management (IPM) through AI automation",
        ]
        for i, obj in enumerate(objectives, 1):
            st.markdown(
                f'''<div style="display:flex;gap:0.7rem;align-items:flex-start;'''  +
                f'''margin-bottom:0.7rem;'''  +
                f'''background:white;border:1px solid #e0ead8;border-radius:10px;'''  +
                f'''padding:0.7rem 1rem;">'''  +
                f'''<span style="background:#e8f5e9;color:#1b5e20;font-weight:700;'''  +
                f'''border-radius:50%;width:24px;height:24px;display:flex;'''  +
                f'''align-items:center;justify-content:center;font-size:0.75rem;'''  +
                f'''flex-shrink:0;">{i}</span>'''  +
                f'''<span style="font-size:0.88rem;color:#333;">{obj}</span>'''  +
                '''</div>''', unsafe_allow_html=True
            )


# ═════════════════════════════════════════════════════════════
#  PAGE 2 — PEST PREDICTION
# ═════════════════════════════════════════════════════════════
elif page == "🔬 Pest Prediction":

    st.markdown('''<p class="hero-title" style="font-size:2rem;">🔬 Pest Prediction</p>''',
                unsafe_allow_html=True)
    st.markdown('''<p class="hero-sub">Upload a clear pest image to get an instant classification.</p>''',
                unsafe_allow_html=True)

    # Load model once
    model, err = load_model()
    if err:
        st.error(f"⚠️  Could not load model: {err}")
        st.info(f"Make sure your trained model is at: `{MODEL_PATH}`")
        st.stop()

    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown('''<div class="sec-hdr">Upload Image</div>''', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop an image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            st.markdown(
                '''<div class="ag-card ag-card-amber" style="font-size:0.8rem;color:#555;">'''  +
                '''<b>💡 Tips for best results:</b><ul style="margin:0.3rem 0 0;padding-left:1.2rem;">'''  +
                '''<li>Clear, well-lit photograph</li>'''  +
                '''<li>Pest fills most of the frame</li>'''  +
                '''<li>Both larvae and adults supported</li>'''  +
                '''</ul></div>''', unsafe_allow_html=True
            )

    with col_res:
        if uploaded:
            st.markdown('''<div class="sec-hdr">Prediction Result</div>''', unsafe_allow_html=True)
            with st.spinner("Analysing image…"):
                top5 = predict(model, img)

            best_name, best_conf = top5[0]
            meta = next((c for c in CLASS_DATA if c[0] == best_name), None)

            # Result hero card
            order = meta[1] if meta else ""
            risk  = meta[2] if meta else ""
            o_col = ORDER_COLORS.get(order, "#4caf50")
            r_col = RISK_COLORS.get(risk, "#ff9800")

            st.markdown(
                '''<div class="result-box">'''  +
                '''<div style="font-size:2.8rem;margin-bottom:0.2rem;">🦟</div>'''  +
                f'''<div class="pred-name">{best_name}</div>'''  +
                f'''<div class="pred-conf">Confidence: {best_conf*100:.2f}%</div>'''  +
                f'''<div style="margin-top:0.8rem;">'''  +
                f'''<span class="pill" style="background:{o_col}22;color:{o_col};'''  +
                f'''border:1px solid {o_col}44;">{order}</span>'''  +
                f'''<span class="pill" style="background:{r_col}22;color:{r_col};'''  +
                f'''border:1px solid {r_col}44;">⚡ {risk} Risk</span>'''  +
                '''</div></div>''', unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Top-5 bar chart via Plotly
            names  = [r[0] for r in top5]
            probs  = [r[1]*100 for r in top5]
            colors = ["#2e7d32" if i == 0 else "#81c784" for i in range(5)]

            fig = go.Figure(go.Bar(
                x=probs[::-1], y=names[::-1],
                orientation="h",
                marker_color=colors[::-1],
                text=[f"{p:.1f}%" for p in probs[::-1]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Top-5 Predictions",
                xaxis_title="Confidence (%)",
                xaxis=dict(range=[0, 110]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(248,250,246,1)",
                height=260,
                margin=dict(l=10, r=10, t=40, b=20),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown(
                '''<div style="height:360px;display:flex;flex-direction:column;'''  +
                '''align-items:center;justify-content:center;'''  +
                '''border:2px dashed #4caf50;border-radius:16px;text-align:center;padding:2rem;">'''  +
                '''<div style="font-size:3rem;margin-bottom:1rem;">🔬</div>'''  +
                '''<div style="font-size:1.1rem;font-weight:700;color:#1b5e20;">Upload an image to begin</div>'''  +
                '''<div style="font-size:0.85rem;color:#555;margin-top:0.5rem;">Supports JPG, JPEG, PNG</div>'''  +
                '''</div>''', unsafe_allow_html=True
            )


# ═════════════════════════════════════════════════════════════
#  PAGE 3 — PEST CLASSES
# ═════════════════════════════════════════════════════════════
elif page == "📋 Pest Classes":

    st.markdown('''<p class="hero-title" style="font-size:2rem;">📋 Pest Classes</p>''',
                unsafe_allow_html=True)

    # Stats
    import collections
    order_counts = collections.Counter(c[1] for c in CLASS_DATA)
    risk_counts  = collections.Counter(c[2] for c in CLASS_DATA)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [72, len(order_counts), risk_counts["Very High"], risk_counts["High"]],
        ["Total Classes", "Insect Orders", "Very High Risk", "High Risk"]
    ):
        col.markdown(
            f'''<div class="stat-box">'''  +
            f'''<div class="stat-val">{val}</div>'''  +
            f'''<div class="stat-lbl">{lbl}</div>'''  +
            '''</div>''', unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    f1, f2 = st.columns([2, 2])
    with f1:
        sel_order = st.selectbox("Filter by Order", ["All"] + sorted(order_counts.keys()))
    with f2:
        sel_risk  = st.selectbox("Filter by Risk",  ["All", "Very High", "High", "Medium", "Low"])

    filtered = CLASS_DATA
    if sel_order != "All":
        filtered = [c for c in filtered if c[1] == sel_order]
    if sel_risk != "All":
        filtered = [c for c in filtered if c[2] == sel_risk]

    st.markdown(
        f'''<div style="font-size:0.85rem;color:#555;margin:0.5rem 0 1rem;">'''  +
        f'''Showing <b style="color:#1b5e20;">{len(filtered)}</b> of 72 classes</div>''',
        unsafe_allow_html=True
    )

    # Class grid — 3 columns
    cols = st.columns(3)
    for i, (name, order, risk) in enumerate(filtered):
        o_col = ORDER_COLORS.get(order, "#4caf50")
        r_col = RISK_COLORS.get(risk, "#ff9800")
        with cols[i % 3]:
            st.markdown(
                '''<div style="background:white;border:1px solid #e0ead8;border-radius:10px;'''  +
                '''padding:0.8rem 1rem;margin-bottom:0.6rem;'''  +
                '''box-shadow:0 1px 6px rgba(0,0,0,0.05);">'''  +
                f'''<div style="font-size:0.9rem;font-weight:600;color:#1b5e20;">{name}</div>'''  +
                '''<div style="margin-top:0.4rem;">'''  +
                f'''<span class="pill" style="background:{o_col}18;color:{o_col};'''  +
                f'''border:1px solid {o_col}33;font-size:0.68rem;">{order}</span>'''  +
                f'''<span class="pill" style="background:{r_col}18;color:{r_col};'''  +
                f'''border:1px solid {r_col}33;font-size:0.68rem;">⚡ {risk}</span>'''  +
                '''</div></div>''', unsafe_allow_html=True
            )


# ═════════════════════════════════════════════════════════════
#  PAGE 4 — ABOUT PROJECT
# ═════════════════════════════════════════════════════════════
elif page == "📖 About Project":

    st.markdown('''<p class="hero-title" style="font-size:2rem;">📖 About the Project</p>''',
                unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('''<div class="sec-hdr">Project Overview</div>''', unsafe_allow_html=True)
        st.markdown(
            '''<div class="ag-card ag-card-green">'''  +
            '''<p style="font-size:0.9rem;color:#333;line-height:1.8;margin:0;">'''  +
            '''This graduation project develops a deep learning system for <b>automatic '''  +
            '''agricultural pest recognition</b>. Farmers and agronomists can photograph '''  +
            '''a pest and receive instant species identification, enabling faster and more '''  +
            '''targeted pest management decisions that reduce crop loss and pesticide use.'''  +
            '''</p></div>''', unsafe_allow_html=True
        )

        st.markdown('''<div class="sec-hdr">Dataset</div>''', unsafe_allow_html=True)
        dataset_info = [
            ("📦", "Source",           "Custom web-scraped dataset"),
            ("🗂️", "Raw images",       "135,190 images"),
            ("✅", "Cleaned images",   "118,664 images (after dedup)"),
            ("🏷️", "Classes",          "72 pest species"),
            ("🔍", "Deduplication",    "pHash · Hamming threshold τ=10"),
            ("⚖️", "Imbalance ratio",  "17.9:1 (max/min class)"),
            ("📐", "Train/Val/Test",   "70% / 15% / 15% stratified"),
        ]
        for icon, key, val in dataset_info:
            st.markdown(
                f'''<div style="display:flex;gap:0.8rem;align-items:center;'''  +
                '''background:white;border:1px solid #e0ead8;border-radius:8px;'''  +
                '''padding:0.5rem 0.9rem;margin-bottom:0.4rem;">'''  +
                f'''<span style="font-size:1.1rem;">{icon}</span>'''  +
                f'''<span style="font-size:0.82rem;color:#333;"><b>{key}:</b> {val}</span>'''  +
                '''</div>''', unsafe_allow_html=True
            )

    with col_r:
        st.markdown('''<div class="sec-hdr">Deep Learning Model</div>''', unsafe_allow_html=True)
        model_info = [
            ("🧠", "Architecture",    "EfficientNet-B4"),
            ("📦", "Pretrained on",   "ImageNet-1K (1.2M images)"),
            ("📐", "Input size",      "380 × 380 × 3 (RGB)"),
            ("🔢", "Parameters",      "≈ 19 million"),
            ("🎯", "Output",          "72 class probabilities"),
            ("📉", "Loss function",   "Class-weighted Cross-Entropy"),
            ("⚡", "Optimizer",       "AdamW · Cosine LR annealing"),
            ("🔄", "Augmentations",   "Flip, rotate, jitter, erasing"),
            ("💻", "Hardware",        "GPU (T4 / A100)"),
        ]
        for icon, key, val in model_info:
            st.markdown(
                f'''<div style="display:flex;gap:0.8rem;align-items:center;'''  +
                '''background:white;border:1px solid #e0ead8;border-radius:8px;'''  +
                '''padding:0.5rem 0.9rem;margin-bottom:0.4rem;">'''  +
                f'''<span style="font-size:1.1rem;">{icon}</span>'''  +
                f'''<span style="font-size:0.82rem;color:#333;"><b>{key}:</b> {val}</span>'''  +
                '''</div>''', unsafe_allow_html=True
            )

        st.markdown('''<div class="sec-hdr" style="margin-top:1rem;">Training Results</div>''',
                    unsafe_allow_html=True)
        metrics = [("Top-1 Accuracy", "TBD"), ("Top-5 Accuracy", "TBD"),
                   ("Best Val Loss", "TBD"), ("Epochs Trained", "30")]
        mc1, mc2 = st.columns(2)
        for i, (k, v) in enumerate(metrics):
            col = mc1 if i % 2 == 0 else mc2
            col.markdown(
                f'''<div class="stat-box" style="margin-bottom:0.5rem;">'''  +
                f'''<div class="stat-val" style="font-size:1.3rem;">{v}</div>'''  +
                f'''<div class="stat-lbl">{k}</div>'''  +
                '''</div>''', unsafe_allow_html=True
            )


# ═════════════════════════════════════════════════════════════
#  PAGE 5 — TEAM
# ═════════════════════════════════════════════════════════════
elif page == "👥 Team":

    # Department header
    st.markdown(
        '''<div style="text-align:center;padding:1rem 0 0.5rem;">'''  +
        '''<div style="font-size:2rem;margin-bottom:0.4rem;">🎓</div>'''  +
        '''<div style="font-size:1rem;color:#4a7c59;font-weight:600;'''  +
        '''text-transform:uppercase;letter-spacing:0.1em;">Department of Computer Science Engineering</div>'''  +
        '''<div style="font-size:2rem;font-weight:800;color:#1b5e20;margin:0.4rem 0;">Our Team</div>'''  +
        '''<div style="font-size:0.9rem;color:#4a7c59;">Graduation Project 1 · 2026</div>'''  +
        '''</div>''', unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Supervisor ─────────────────────────────────────────
    st.markdown(
        '''<div style="background:linear-gradient(135deg,#e8f5e9,#f1f8e9);'''  +
        '''border:2px solid #4caf50;border-radius:16px;padding:1.8rem;text-align:center;'''  +
        '''margin-bottom:2rem;">'''  +
        '''<div style="font-size:0.75rem;font-weight:700;color:#4a7c59;'''  +
        '''text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.8rem;">Project Supervisor</div>'''  +
        '''<div style="font-size:3rem;margin-bottom:0.5rem;">👨‍🏫</div>'''  +
        '''<div style="font-size:1.5rem;font-weight:800;color:#1b5e20;">Dr. XXXXX</div>'''  +
        '''<div style="font-size:0.9rem;color:#4a7c59;margin-top:0.3rem;">Associate Professor · Computer Science Engineering</div>'''  +
        '''</div>''', unsafe_allow_html=True
    )

    # ── Team Members ───────────────────────────────────────
    st.markdown(
        '''<div style="font-size:1.1rem;font-weight:700;color:#1b5e20;'''  +
        '''text-align:center;margin-bottom:1.2rem;">Team Members</div>''',
        unsafe_allow_html=True
    )

    members = [
        ("👩‍💻", "Member 1", "Deep Learning Engineer",  "#e8f5e9", "#4caf50"),
        ("👨‍💻", "Member 2", "Data Pipeline Engineer",   "#e3f2fd", "#2196f3"),
        ("👩‍🔬", "Member 3", "Model Training & Eval",    "#fff3e0", "#ff9800"),
        ("👨‍🎨", "Member 4", "UI / Deployment Engineer", "#f3e5f5", "#9c27b0"),
    ]

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    for col, (icon, name, role, bg, border) in zip([c1,c2,c3,c4], members):
        col.markdown(
            '''<div class="team-card">'''  +
            f'''<div class="avatar" style="background:{bg};border:2px solid {border};">{icon}</div>'''  +
            f'''<div class="member-name">{name}</div>'''  +
            f'''<div class="member-role">{role}</div>'''  +
            '''</div>''', unsafe_allow_html=True
        )

    # ── Footer banner ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '''<div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);'''  +
        '''border-radius:14px;padding:1.5rem 2rem;text-align:center;">'''  +
        '''<div style="font-size:1.2rem;font-weight:700;color:white;margin-bottom:0.4rem;">AgroScan</div>'''  +
        '''<div style="font-size:0.85rem;color:#a5d6a7;">Agricultural Pest Classification System</div>'''  +
        '''<div style="font-size:0.78rem;color:#81c784;margin-top:0.5rem;">Graduation Project 1 · 2026</div>'''  +
        '''</div>''', unsafe_allow_html=True
    )
