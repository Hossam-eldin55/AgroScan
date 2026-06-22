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
NUM_CLASSES  = 67                          # 72 originally collected − 5 excluded during training
IMG_SIZE     = 224                          # WaveletEnhancedResNet / ResNet50 native input
BACKBONE_NAME = os.environ.get("BACKBONE_NAME", "resnet50")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_url():
    """Read MODEL_URL from env var first, then Streamlit secrets."""
    url = os.environ.get("MODEL_URL", "")
    if url:
        return url
    try:
        return st.secrets["MODEL_URL"]
    except Exception:
        return ""


def ensure_model_downloaded():
    """Download the model file from MODEL_URL if it isn't already on disk."""
    if os.path.exists(MODEL_PATH):
        return
    url = get_model_url()
    if not url:
        return
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    with st.spinner("Downloading model weights from Google Drive … (~91 MB, first run only)"):
        try:
            import gdown
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return

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
#  CLASS DATA  (67 classes — order matches the trained model's
#  output indices exactly, verified against the test-set
#  classification_report from the training notebook)
# ─────────────────────────────────────────────────────────────
CLASS_DATA = [
    ("Acalymma vittatum",               "Coleoptera",   "High"),
    ("Achatina fulica",                 "Gastropoda",   "High"),
    ("Alticini",                        "Coleoptera",   "Medium"),
    ("Ampelophaga",                     "Lepidoptera",  "High"),
    ("Anasa tristis",                   "Hemiptera",    "High"),
    ("Aphids",                          "Hemiptera",    "Very High"),
    ("Armyworm caterpillar",            "Lepidoptera",  "Very High"),
    ("Aulacophora similis",             "Coleoptera",   "Medium"),
    ("Beet spot flies butterfly",       "Diptera",      "Medium"),
    ("Beet spot flies caterpillar",     "Diptera",      "Medium"),
    ("Cicadella viridis",               "Hemiptera",    "Medium"),
    ("Cicadellidae",                    "Hemiptera",    "High"),
    ("Dermaptera",                      "Dermaptera",   "Low"),
    ("Icerya purchasi Maskell",         "Hemiptera",    "High"),
    ("Leptinotarsa decemlineata",       "Coleoptera",   "Very High"),
    ("Locustoidea",                     "Orthoptera",   "Very High"),
    ("Lycorma delicatula",              "Hemiptera",    "High"),
    ("Mantodea",                        "Mantodea",     "Low"),
    ("Miridae",                         "Hemiptera",    "High"),
    ("Potosiabre vitarsis",             "Coleoptera",   "Medium"),
    ("Prodenia litura butterfly",       "Lepidoptera",  "High"),
    ("Prodenia litura caterpillar",     "Lepidoptera",  "Very High"),
    ("Rhytidodera bowrinii white",      "Coleoptera",   "High"),
    ("Salurnis marginella Guerr",       "Hemiptera",    "Medium"),
    ("Thrips",                          "Thysanoptera", "Very High"),
    ("Xylotrechus",                     "Coleoptera",   "High"),
    ("Alfalfa plant bug butterfly",     "Hemiptera",    "High"),
    ("Alfalfa plant bug caterpillar",   "Hemiptera",    "High"),
    ("Army worm butterfly",             "Lepidoptera",  "High"),
    ("Army worm caterpillar",           "Lepidoptera",  "Very High"),
    ("Asiatic rice borer butterfly",    "Lepidoptera",  "High"),
    ("Asiatic rice borer caterpillar",  "Lepidoptera",  "Very High"),
    ("Beet army worm butterfly",        "Lepidoptera",  "High"),
    ("Beet army worm caterpillar",      "Lepidoptera",  "High"),
    ("Beetle",                          "Coleoptera",   "High"),
    ("Black cutworm butterfly",         "Lepidoptera",  "Medium"),
    ("Black cutworm caterpillar",       "Lepidoptera",  "Very High"),
    ("Blister beetle",                  "Coleoptera",   "Medium"),
    ("Bollworm butterfly",              "Lepidoptera",  "High"),
    ("Bollworm caterpillar",            "Lepidoptera",  "Very High"),
    ("Cabbage army worm",               "Lepidoptera",  "High"),
    ("Cerodonta denticornis caterpillar","Diptera",     "Medium"),
    ("Corn borer butterfly",            "Lepidoptera",  "High"),
    ("Corn borer caterpillar",          "Lepidoptera",  "Very High"),
    ("Grasshopper",                     "Orthoptera",   "Very High"),
    ("Grub",                            "Coleoptera",   "High"),
    ("Large cutworm butterfly",         "Lepidoptera",  "Medium"),
    ("Legume blister beetle",           "Coleoptera",   "High"),
    ("Lytta polita",                    "Coleoptera",   "Medium"),
    ("Mites",                           "Arachnida",    "Very High"),
    ("Mole cricket",                    "Orthoptera",   "High"),
    ("Mosquito",                        "Diptera",      "Medium"),
    ("Peach borer butterfly",           "Lepidoptera",  "High"),
    ("Peach borer caterpillar",         "Lepidoptera",  "Very High"),
    ("Red spider",                      "Arachnida",    "Very High"),
    ("Rice leaf roller butterfly",      "Lepidoptera",  "High"),
    ("Rice leaf roller caterpillar",    "Lepidoptera",  "Very High"),
    ("Rice water weevil",               "Coleoptera",   "Very High"),
    ("Sawfly butterfly",                "Hymenoptera",  "Medium"),
    ("Sericaorient alismots chulsky",   "Coleoptera",   "Medium"),
    ("Stem borer butterfly",            "Lepidoptera",  "High"),
    ("Stem borer caterpillar",          "Lepidoptera",  "Very High"),
    ("Tarnished plant bug",             "Hemiptera",    "High"),
    ("Therioaphis maculata Buckton",    "Hemiptera",    "High"),
    ("Wireworm butterfly",              "Coleoptera",   "Medium"),
    ("Wireworm caterpillar",            "Coleoptera",   "Very High"),
    ("Yellow cutworm butterfly",        "Lepidoptera",  "Medium"),
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
#  MODEL ARCHITECTURE — must match the training notebook exactly
#  (Wavelet_ResNet_on_Cleaned_Dataset_without_conflicts_.ipynb)
# ─────────────────────────────────────────────────────────────
class HaarDWT2D(nn.Module):
    """2D Haar Wavelet Decomposition.

    Input:  RGB image tensor [B, 3, H, W] (H, W must be even)
    Output: Wavelet tensor   [B, 12, H/2, W/2]  (4 sub-bands per RGB channel)
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("pL", torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.register_buffer("pH", torch.tensor([0.5, -0.5], dtype=torch.float32))

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError("Input height and width must be even for HaarDWT2D.")

        x_grouped = x.view(B * C, 1, H, W)
        k_LL = torch.outer(self.pL, self.pL).view(1, 1, 2, 2)
        k_LH = torch.outer(self.pL, self.pH).view(1, 1, 2, 2)
        k_HL = torch.outer(self.pH, self.pL).view(1, 1, 2, 2)
        k_HH = torch.outer(self.pH, self.pH).view(1, 1, 2, 2)
        kernels = torch.cat([k_LL, k_LH, k_HL, k_HH], dim=0).to(x.device)

        sub_bands = nn.functional.conv2d(x_grouped, kernels, stride=2)
        sub_bands = sub_bands.view(B, C, 4, H // 2, W // 2)
        out = sub_bands.reshape(B, C * 4, H // 2, W // 2)
        return out


class WaveletEnhancedResNet(nn.Module):
    def __init__(self, num_classes, backbone_name="resnet50"):
        super().__init__()
        self.dwt = HaarDWT2D()

        # Convert 12 wavelet channels back to 3 channels so a
        # pretrained ResNet can process them like RGB-like features.
        self.frequency_projector = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=None)
        elif backbone_name == "resnet34":
            self.backbone = models.resnet34(weights=None)
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=None)
        elif backbone_name == "resnet101":
            self.backbone = models.resnet101(weights=None)
        else:
            raise ValueError("Choose one of: resnet18, resnet34, resnet50, resnet101")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x_freq = self.dwt(x)
        x_encoded = self.frequency_projector(x_freq)
        return self.backbone(x_encoded)


# ─────────────────────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    """Load WaveletEnhancedResNet from MODEL_PATH (auto-downloads from MODEL_URL if set)."""
    ensure_model_downloaded()
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at {MODEL_PATH}"
    try:
        model = WaveletEnhancedResNet(num_classes=NUM_CLASSES, backbone_name=BACKBONE_NAME)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle different checkpoint formats
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state:
                    state = state[key]
                    break
        # strict=True on purpose: if the architecture doesn't match the
        # checkpoint exactly, fail loudly instead of silently loading
        # an untrained model.
        model.load_state_dict(state, strict=True)
        model.to(DEVICE).eval()
        return model, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
#  PREPROCESSING & INFERENCE
# ─────────────────────────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
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
        ["🏠 Home", "🔬 Pest Prediction", "📋 Pest Classes", "👥 Team"],
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
        ("67",    "Pest Species"),
        ("📸",    "Photo Upload"),
        ("⚡",    "Instant Results"),
        ("🌾",    "Crop Protection"),
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
        st.markdown('''<div class="sec-hdr">What is AgroScan?</div>''', unsafe_allow_html=True)
        st.markdown('''<div class="ag-card ag-card-green">'''  +
            '''<p style="font-size:0.95rem;color:#333;line-height:1.8;margin:0;">'''  +
            '''<b>AgroScan</b> is an intelligent pest identification tool designed to help '''  +
            '''farmers, agronomists, and agricultural professionals quickly identify harmful '''  +
            '''insects and pests from field photographs. Simply upload a photo of a suspected '''  +
            '''pest and AgroScan will instantly tell you the species name, its insect order, '''  +
            '''and its risk level to your crops — enabling faster, more targeted pest '''  +
            '''management decisions that reduce crop loss.'''  +
            '''</p></div>''', unsafe_allow_html=True)

        st.markdown('''<div class="sec-hdr">How to Use</div>''', unsafe_allow_html=True)
        steps = [
            ("📸", "Take a Photo",      "Photograph the pest you found in the field. Make sure it is clear, well-lit, and fills most of the frame."),
            ("📂", "Upload the Image",  "Go to the Pest Prediction page and upload your photo (JPG or PNG)."),
            ("⏱️", "Get Instant Results","AgroScan will analyse your image and return the most likely species match along with a confidence score."),
            ("📋", "Explore Classes",   "Browse all 67 supported pest species under the Pest Classes page to learn more about risk levels and categories."),
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
        st.markdown('''<div class="sec-hdr">Key Features</div>''', unsafe_allow_html=True)
        features = [
            "Identifies 67 agriculturally significant pest species from a single photo",
            "Covers both adult and larval stages for many common pests",
            "Provides a confidence score so you know how certain the result is",
            "Displays risk level (Very High / High / Medium / Low) for each identified pest",
            "Shows insect order classification alongside the species name",
            "Browse the full pest catalogue with filters by order and risk level",
        ]
        for i, feat in enumerate(features, 1):
            st.markdown(
                f'''<div style="display:flex;gap:0.7rem;align-items:flex-start;'''  +
                f'''margin-bottom:0.7rem;'''  +
                f'''background:white;border:1px solid #e0ead8;border-radius:10px;'''  +
                f'''padding:0.7rem 1rem;">'''  +
                f'''<span style="background:#e8f5e9;color:#1b5e20;font-weight:700;'''  +
                f'''border-radius:50%;width:24px;height:24px;display:flex;'''  +
                f'''align-items:center;justify-content:center;font-size:0.75rem;'''  +
                f'''flex-shrink:0;">{i}</span>'''  +
                f'''<span style="font-size:0.88rem;color:#333;">{feat}</span>'''  +
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

        # ── Debug panel ──────────────────────────────────────────
        with st.expander("🔍 Debug info (share this if asking for help)"):
            url = get_model_url()
            st.write("**MODEL_PATH:**", MODEL_PATH)
            st.write("**MODEL_URL detected:**", url if url else "❌ Empty — secret not found")
            st.write("**File exists on disk:**", os.path.exists(MODEL_PATH))
            if url and not os.path.exists(MODEL_PATH):
                if st.button("⬇️ Retry download now"):
                    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
                    try:
                        import gdown
                        with st.spinner("Downloading …"):
                            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
                        if os.path.exists(MODEL_PATH):
                            st.success("✅ Downloaded! Please reload the page.")
                        else:
                            st.error("gdown ran but file still not found — Drive link may be restricted.")
                    except Exception as ex:
                        st.error(f"gdown error: {ex}")
        # ─────────────────────────────────────────────────────────
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
        f'''Showing <b style="color:#1b5e20;">{len(filtered)}</b> of {len(CLASS_DATA)} classes</div>''',
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
#  PAGE 4 — TEAM
# ═════════════════════════════════════════════════════════════
elif page == "👥 Team":

    # University / department header
    st.markdown(
        '''<div style="text-align:center;padding:1rem 0 0.5rem;">'''  +
        '''<div style="font-size:2.5rem;margin-bottom:0.6rem;">🎓</div>'''  +
        '''<div style="font-size:1.15rem;font-weight:800;color:#1b5e20;margin-bottom:0.2rem;">'''  +
        '''Egypt Japan University of Science and Technology</div>'''  +
        '''<div style="font-size:0.95rem;color:#4a7c59;font-weight:600;margin-bottom:0.1rem;">'''  +
        '''Faculty of Engineering</div>'''  +
        '''<div style="font-size:0.88rem;color:#4a7c59;margin-bottom:0.8rem;">'''  +
        '''Computer Science and Engineering Department</div>'''  +
        '''<div style="width:60px;height:3px;background:#4caf50;margin:0 auto 1rem;border-radius:2px;"></div>'''  +
        '''<div style="font-size:1.8rem;font-weight:800;color:#1b5e20;margin:0.4rem 0;">Our Team</div>'''  +
        '''<div style="font-size:0.9rem;color:#4a7c59;">Graduation Project · 2026</div>'''  +
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
        '''<div style="font-size:0.9rem;color:#4a7c59;margin-top:0.3rem;">Associate Professor · Computer Science and Engineering</div>'''  +
        '''</div>''', unsafe_allow_html=True
    )

    # ── Team Members ───────────────────────────────────────
    st.markdown(
        '''<div style="font-size:1.1rem;font-weight:700;color:#1b5e20;'''  +
        '''text-align:center;margin-bottom:1.4rem;">Team Members</div>''',
        unsafe_allow_html=True
    )

    avatars = [
        ("👩‍💻", "#e8f5e9", "#4caf50"),
        ("👨‍💻", "#e3f2fd", "#2196f3"),
        ("👩‍🔬", "#fff3e0", "#ff9800"),
        ("👨‍🎨", "#f3e5f5", "#9c27b0"),
        ("👩‍💼", "#fce4ec", "#e91e63"),
        ("👨‍🔧", "#e0f7fa", "#00bcd4"),
    ]

    # Row 1: 3 members
    r1 = st.columns(3, gap="medium")
    for col, (icon, bg, border) in zip(r1, avatars[:3]):
        col.markdown(
            '''<div class="team-card">'''  +
            f'''<div class="avatar" style="background:{bg};border:2px solid {border};">{icon}</div>'''  +
            '''<div class="member-name" style="color:#888;font-style:italic;">Name</div>'''  +
            '''<div class="member-role" style="color:#aaa;font-size:0.78rem;margin-top:0.3rem;">ID: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</div>'''  +
            '''</div>''', unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: 3 members
    r2 = st.columns(3, gap="medium")
    for col, (icon, bg, border) in zip(r2, avatars[3:]):
        col.markdown(
            '''<div class="team-card">'''  +
            f'''<div class="avatar" style="background:{bg};border:2px solid {border};">{icon}</div>'''  +
            '''<div class="member-name" style="color:#888;font-style:italic;">Name</div>'''  +
            '''<div class="member-role" style="color:#aaa;font-size:0.78rem;margin-top:0.3rem;">ID: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</div>'''  +
            '''</div>''', unsafe_allow_html=True
        )

    # ── Footer banner ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '''<div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);'''  +
        '''border-radius:14px;padding:1.5rem 2rem;text-align:center;">'''  +
        '''<div style="font-size:1.2rem;font-weight:700;color:white;margin-bottom:0.4rem;">AgroScan</div>'''  +
        '''<div style="font-size:0.85rem;color:#a5d6a7;">Agricultural Pest Classification System</div>'''  +
        '''<div style="font-size:0.78rem;color:#81c784;margin-top:0.5rem;">'''  +
        '''Egypt Japan University of Science and Technology · Faculty of Engineering · 2026</div>'''  +
        '''</div>''', unsafe_allow_html=True
    )
