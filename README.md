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

## 1. Get the code onto GitHub

```bash
git init
git add .
git commit -m "Initial commit: AgroScan Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/agroscan-streamlit.git
git push -u origin main
```

## 2. Provide the trained model

`best_wavelet_resnet_model.pth` is a binary weights file. At ~91MB it fits
under GitHub's 100MB hard limit, so you can commit it directly — no Git LFS
needed. (GitHub will show a one-time warning above 50MB recommending LFS;
the push still completes fine.)

**Option A — Commit it directly (recommended for your file size)**
```bash
git add models/best_wavelet_resnet_model.pth
git commit -m "Add trained model weights"
git push
```
Make sure the `models/*.pth` line in `.gitignore` is removed or commented
out first, or git will silently skip the file.

**Option B — Host externally, auto-download on startup**
Use this if the file grows past 100MB later, or you'd rather not bloat
the git history. Upload it to Hugging Face Hub, a GitHub Release asset,
or any direct-download URL, then set a `MODEL_URL` secret (see step 4) —
the app downloads it into `models/best_wavelet_resnet_model.pth` the
first time it runs and reuses it afterward.

## 3. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**, select this repo, branch `main`, and file path `app.py`.
3. Click **Deploy**.

## 4. (If using Option B) Set the model URL as a secret

In the app's dashboard on Streamlit Cloud: **Settings → Secrets**, add:
```toml
MODEL_URL = "https://your-host.com/best_wavelet_resnet_model.pth"
```
Locally, you'd instead create `.streamlit/secrets.toml` with the same line
(this file is gitignored and should never be committed).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
Place `best_wavelet_resnet_model.pth` in `models/` first, or set `MODEL_URL`/`MODEL_PATH`
as environment variables.

## Notes

- **Checkpoint formats supported**: a raw `state_dict()`, or a checkpoint
  dict containing `model_state_dict` / `state_dict` / `model`.
- **Changing the backbone**: edit `load_model()` in `app.py` if your
  trained model isn't EfficientNet-B4.
- **CPU only on Streamlit Cloud**: there's no GPU on the free tier;
  inference will be slower than on a Colab GPU runtime but works fine for
  single-image predictions. `requirements.txt` pulls CPU-only torch
  wheels to keep build times reasonable.
- The original Colab notebook (`AgroScan_Complete.ipynb`) used `ngrok`
  for a temporary public URL — that's no longer needed since Streamlit
  Cloud hosts the app directly.
