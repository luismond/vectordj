import os, json, numpy as np, pandas as pd
import streamlit as st
from recutils.indexer import query_index, query_index_filtered, id_to_track, load_feature_matrix, lookup_by_path
from recutils.model import has_model

st.title("🔎 Similar to…")

index_path = os.path.join("data", "index", "faiss_hnsw.index")
ids_path = os.path.join("data", "index", "row_ids.json")
feat_dir = os.path.join("data", "features")

if not (os.path.exists(index_path) and os.path.exists(ids_path) and os.path.isdir(feat_dir)):
    st.error("Index or features not found. Please run `python build_index.py` first.")
    st.stop()

ids = json.load(open(ids_path))
feat_files = [os.path.join(feat_dir, f"{tid}.npy") for tid in ids]

# --- Pick seed track ---
seed_mode = st.radio("Choose seed:", ["Pick by file path", "Pick by track ID"])
seed = None

if seed_mode == "Pick by file path":
    seed_path = st.text_input("Full path to a known track on disk:")
    if seed_path:
        tid = lookup_by_path(seed_path)
        if tid:
            seed = tid
        else:
            st.warning("That path is not in the catalog.")
else:
    seed = st.text_input("Track ID (16-hex)")

# --- Options ---
k = st.slider("Neighbors (k)", 5, 50, 25)
bpm_filter = st.checkbox("Filter by BPM", value=False)
bpm_center = st.number_input("Target BPM", value=128.0) if bpm_filter else None
bpm_tol = st.number_input("BPM tolerance (±)", value=6.0) if bpm_filter else None

camel_filter = st.checkbox("Use Camelot key mixing", value=False)
camel_mode = st.selectbox("Camelot mode", ["compatible", "same"]) if camel_filter else None
camel_seed = st.text_input("Seed Camelot key (e.g., 8A, 9B). Leave blank to use seed track's key.") if camel_filter else ""

rerank = st.checkbox("Re-rank by my predicted stars (if model trained)", value=False)

# --- Main logic ---
if seed:
    seed_feat_path = os.path.join(feat_dir, f"{seed}.npy")
    if not os.path.exists(seed_feat_path):
        st.error("No features for that track. Re-run build to include it.")
        st.stop()

    v = np.load(seed_feat_path)

    # Camelot seed handling
    camel_seed_val = camel_seed.strip().upper() if camel_filter and camel_seed.strip() else None
    if camel_filter and camel_seed_val is None:
        row = id_to_track(seed)
        # row = (id, path, title, artist, album, genre, duration, stars, bpm, key, camelot)
        camel_seed_val = row[10] if row and len(row) > 10 else None

    # Neighbor query
    if camel_filter or bpm_filter:
        neighbors = query_index_filtered(
            v,
            k=k,
            bpm_center=bpm_center if bpm_filter else None,
            bpm_tolerance=bpm_tol if bpm_filter else 6.0,
            camelot=camel_seed_val if camel_filter else None,
            camelot_mode=camel_mode if camel_filter else "compatible",
        )
    else:
        neighbors = query_index(v, k=k)

    # Build dataframe
    rows = []
    for tid, sim in neighbors:
        id_, path, title, artist, album, genre, duration, stars, bpm, key, camelot = id_to_track(tid)
        rows.append(
            dict(
                similarity=round(sim, 4),
                id=id_,
                title=title,
                artist=artist,
                album=album,
                genre=genre,
                duration=duration,
                stars=stars,
                bpm=bpm,
                key=key,
                camelot=camelot,
                path=path,
            )
        )
    df = pd.DataFrame(rows)

    # Optional re-ranking
    if rerank and has_model():
        import joblib
        from recutils.model import MODEL_PATH
        blob = joblib.load(MODEL_PATH)
        model = blob["model"]
        feat_vecs = []
        order_ids = []
        for tid, _ in neighbors:
            fp = os.path.join(feat_dir, f"{tid}.npy")
            feat_vecs.append(np.load(fp))
            order_ids.append(tid)
        X = np.stack(feat_vecs)
        preds = model.predict(X)
        df["pred_stars"] = preds
        df = df.sort_values(by=["pred_stars", "similarity"], ascending=[False, False]).head(k)

    st.dataframe(df, use_container_width=True)
