import os, json, numpy as np, pandas as pd
import streamlit as st
from recutils.indexer import load_feature_matrix, ids_to_meta

st.title("🗺️ Map (UMAP)")

umap_path = os.path.join("data","umap","umap_2d.npy")
ids_path  = os.path.join("data","index","row_ids.json")
feat_dir  = os.path.join("data","features")
badrows_path = os.path.join("data","umap","umap_dropped_rows.csv")

os.makedirs(os.path.dirname(umap_path), exist_ok=True)

if not (os.path.exists(ids_path) and os.path.isdir(feat_dir)):
    st.error("Features not found. Please run `python build_index.py` first.")
    st.stop()

seed        = st.number_input("Random seed", value=42)
n_neighbors = st.slider("n_neighbors", 5, 50, 15)
min_dist    = st.slider("min_dist", 0.0, 1.0, 0.1, 0.05)
standardize = st.checkbox("Standardize features before UMAP (recommended)", value=True)
recompute   = st.button("(Re)compute map")

ids = json.load(open(ids_path))

def clean_features(X: np.ndarray):
    X = np.asarray(X, dtype=np.float32)
    # Treat +/-inf as NaN so we can impute
    X[~np.isfinite(X)] = np.nan

    # Drop columns that are entirely NaN
    col_all_nan = np.isnan(X).all(axis=0)
    if col_all_nan.any():
        X = X[:, ~col_all_nan]
        st.warning(f"Dropped {int(col_all_nan.sum())} all-NaN feature columns.")

    # Impute remaining NaNs with column-wise median
    if np.isnan(X).any():
        col_median = np.nanmedian(X, axis=0)
        # If a column is still NaN median (e.g., all NaN), set to zero
        nan_meds = ~np.isfinite(col_median)
        if nan_meds.any():
            col_median[nan_meds] = 0.0
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_median, inds[1])

    # After imputation, mark any still-nonfinite rows (extremely rare)
    good_mask = np.isfinite(X).all(axis=1)
    return X, good_mask

def standardize_features(X):
    # Zero-variance columns will produce NaNs; guard them
    std = X.std(axis=0)
    zero_var = std == 0
    if zero_var.any():
        st.info(f"Found {int(zero_var.sum())} zero-variance columns; skipping standardization for them.")
        std = std.copy()
        std[zero_var] = 1.0
    Xz = (X - X.mean(axis=0)) / std
    return Xz.astype(np.float32)

if recompute or not os.path.exists(umap_path):
    st.write("Loading features…")
    # Adjust this call if your load_feature_matrix returns (X, ids_from_features)
    X, ids2 = load_feature_matrix()  # expected shape: (n_items, n_features)

    if X.shape[0] != len(ids):
        st.warning(f"Row count mismatch: features={X.shape[0]}, ids={len(ids)}. Will align by index; check your index build.")
        # Align to min length
        n = min(X.shape[0], len(ids))
        X = X[:n]
        ids = ids[:n]

    st.write("Cleaning features (handling NaN/inf, imputing)…")
    X_clean, good_mask = clean_features(X)

    dropped = (~good_mask).sum()
    if dropped:
        # Save info about dropped rows
        dropped_idx = np.where(~good_mask)[0].tolist()
        dropped_ids = [ids[i] for i in dropped_idx]
        pd.DataFrame({"row_index": dropped_idx, "row_id": dropped_ids}).to_csv(badrows_path, index=False)
        st.warning(f"Dropped {dropped} rows with non-finite values after imputation. Saved details to {badrows_path}.")
        # Keep only good rows (and matching ids)
        X_clean = X_clean[good_mask]
        ids     = [ids[i] for i, ok in enumerate(good_mask) if ok]

    if standardize:
        st.write("Standardizing…")
        X_clean = standardize_features(X_clean)

    st.write("Running UMAP… (metric=cosine)")
    import umap
    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="cosine",
        random_state=int(seed),
        verbose=True,
    )

    XY = reducer.fit_transform(X_clean).astype("float32")
    np.save(umap_path, XY)
    st.success(f"Saved UMAP to {umap_path} with {XY.shape[0]} points.")

else:
    st.info("Loading precomputed UMAP…")
    XY = np.load(umap_path)

# --- Simple preview scatter (optional) ---
try:
    import pandas as pd
    import altair as alt
    df = pd.DataFrame({"x": XY[:,0], "y": XY[:,1], "id": ids})
    chart = alt.Chart(df).mark_circle(opacity=0.6).encode(
        x="x", y="y",
        tooltip=["id"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
except Exception as e:
    st.write("Preview not available:", e)

# Optional: show link to dropped rows file if it exists
if os.path.exists(badrows_path):
    st.download_button("Download dropped-rows CSV", data=open(badrows_path, "rb"), file_name="umap_dropped_rows.csv")
