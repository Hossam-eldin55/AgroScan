# 🌿 AgroScan — Agricultural Pest Classification

A Streamlit app that identifies agricultural pests from photos using an
EfficientNet-B4 model trained on 72 pest species. Originally built as a
Colab + ngrok notebook; this repo is the same app restructured for
GitHub → Streamlit Community Cloud deployment.

## Repo structure

```
agroscan-streamlit/
├── app.py                   # Main Streamlit app (5 pages)
├── requirements.txt         # Python dependencies (CPU-only torch)
├── .gitignore                # Excludes model weights, caches, secrets
├── .streamlit/
│   └── config.toml          # Theme + server settings
└── README.md
```
