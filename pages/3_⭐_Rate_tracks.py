import os, pandas as pd, sqlite3, subprocess, shutil, random
import streamlit as st
from recutils.indexer import DB_PATH

if not os.path.exists(DB_PATH):
    st.error("Database not found. Run `python build_index.py` first.")
    st.stop()

with sqlite3.connect(DB_PATH) as conn:
    rated = conn.execute("SELECT COUNT(*) FROM tracks WHERE stars IS NOT NULL").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
st.metric("Rated tracks", f"{rated} / {total}", f"{(rated/total if total else 0):.1%}")

st.title("⭐ Rate tracks")

if "rating_saved" in st.session_state:
    st.success(st.session_state.pop("rating_saved"))

def get_random_unrated_ids(limit=30):
    conn = sqlite3.connect(DB_PATH)
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM tracks WHERE stars IS NULL"
    ).fetchall()]
    conn.close()
    #random.seed(42)  # or
    st.session_state.get("batch_seed", 42)
    return random.sample(ids, min(limit, len(ids)))

def get_df_for_ids(ids):
    # if not ids:
    #    return pd.DataFrame(columns=["id","path","title","artist","album","genre","duration","stars"])
    conn = sqlite3.connect(DB_PATH)
    ph = ",".join("?"*len(ids))
    # Use pandas to read SQL query, passing column names, connection and track IDs as parameters
    df = pd.read_sql_query(
        f"SELECT id, path, title, artist, album, genre, duration, stars FROM tracks WHERE id IN ({ph})",
        conn, params=ids
    )
    # Close DB connection
    conn.close()
    # Reverse dict to generate track order
    order = {tid:i for i, tid in enumerate(ids)}
    df["__order"] = df["id"].map(order)
    return df.sort_values("__order")

# Create batch once
if "batch_ids" not in st.session_state:
    st.session_state.batch_ids = get_random_unrated_ids(limit=50)

df = get_df_for_ids(st.session_state.batch_ids)

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

if df.empty:
    st.success("All tracks are rated!")
else:
    st.caption("Tip: click the preview button to hear ~30s before rating.")
    for _, row in df.iterrows():
        with st.expander(f"{row['artist']} – {row['title']} | {row['album']}"):
            st.text(row["path"])

            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                start_sec = st.selectbox("Start (s)", [0, 30, 60, 90], index=0, key=row["id"]+"_start")
            with col2:
                dur_sec = st.selectbox("Dur (s)", [15, 30, 45], index=1, key=row["id"]+"_dur")

            try:
                with open(row["path"], "rb") as fh:
                    st.audio(fh.read())
                    st.caption("Playing original file (no preview). If it fails, install ffmpeg for MP3 previews.")
            except Exception:
                if not has_ffmpeg():
                    st.warning("ffmpeg not found. Install ffmpeg to enable reliable MP3 previews.")
                else:
                    st.warning("Could not generate or play a preview for this file.")

            stars = st.slider("Stars", 1, 5, 1, key=row["id"])
            if st.button("Save", key=row["id"] + "_save"):
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("UPDATE tracks SET stars=? WHERE id=?", (int(stars), row["id"]))
                st.session_state["rating_saved"] = f"Saved {int(stars)}⭐ for {row['artist']} – {row['title']}."


st.caption("Tip: use the sidebar rerun button any time you want a fresh unrated batch.")
