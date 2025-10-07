# Crate Digger (Local Music Recommender)

A tiny Streamlit app to explore a local music collection (~10k tracks) by **similarity, clusters, and personal ratings**.

## What it does
- Extracts lightweight audio features (tempo, MFCCs, chroma, timbre).
- (Optional) Loads learned embeddings if you add them later.
- Builds a FAISS HNSW index for fast k-NN.
- Lets you:
  - Pick a track and find **similar** ones.
  - See a **2-D UMAP map** of your library and click to preview neighbors.
  - **Rate** tracks (1–5⭐) and save to a local SQLite DB.
  - Filter neighbors by **BPM** and **Camelot key** for DJ-friendly transitions.
  - **Train a model (LightGBM)** to re-rank by your predicted stars.

## Quickstart
```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml   # edit music_dir
python build_index.py                # catalog → features → FAISS
streamlit run app.py

Outputs go under data/:

data/tracks.sqlite

data/features/*.npy

data/index/faiss_hnsw.index, row_ids.json

data/umap/umap_2d.npy

data/model/lgbm_stars.pkl (after training))

```

## Dependencies

Listening to song previews depends on having ffmpeg installed in your OS.

