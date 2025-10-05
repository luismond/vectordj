import os, pandas as pd
import streamlit as st

st.set_page_config(page_title="Crate Digger", page_icon="🎧", layout="wide")
st.title("🎧 Crate Digger – Local Music Recommender")

st.markdown('''Use the pages on the left:
🔎 Similar to…: pick a seed track and get neighbors.
🗺️ Map: visualize your library in 2-D (UMAP).
⭐ Rate tracks: quickly add 1–5⭐ ratings.
⚙️ Train model: fit a LightGBM regressor for your stars.
Run python build_index.py first to create DB/features/index.
''')


data_dir = "data"

index_ok = os.path.exists(os.path.join(data_dir, "index", "faiss_hnsw.index"))

db_ok = os.path.exists(os.path.join(data_dir, "tracks.sqlite"))

feat_ok = os.path.exists(os.path.join(data_dir, "features")) \
    and os.path.isdir(os.path.join(data_dir, "features")) \
        and len(os.listdir(os.path.join(data_dir, "features"))) > 0

st.subheader("Status")
st.write({"DB": db_ok, "Features": feat_ok, "Index": index_ok})
st.info("Tip: re-run python build_index.py after adding new music. It's resumable.")
