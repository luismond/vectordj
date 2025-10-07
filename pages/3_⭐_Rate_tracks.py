import os, pandas as pd, sqlite3, subprocess, shutil
import streamlit as st

import sqlite3
from recutils.indexer import DB_PATH
conn = sqlite3.connect(DB_PATH)
rated = conn.execute("SELECT COUNT(*) FROM tracks WHERE stars IS NOT NULL").fetchone()[0]
total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
conn.close()
st.metric("Rated tracks", f"{rated} / {total}", f"{(rated/total if total else 0):.1%}")


st.title("⭐ Rate tracks")

DB_PATH = os.path.join("data", "tracks.sqlite")
PREVIEW_DIR = os.path.join("data", "previews")
os.makedirs(PREVIEW_DIR, exist_ok=True)

if not os.path.exists(DB_PATH):
    st.error("Database not found. Run `python build_index.py` first.")
    st.stop()

def random_unrated(limit=30):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id, path, title, artist, album, genre, duration FROM tracks "
        "WHERE stars IS NULL ORDER BY RANDOM() LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df

def preview_path_for(tid: str) -> str:
    return os.path.join(PREVIEW_DIR, f"{tid}.mp3")

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

@st.cache_data(show_spinner=False)
def ensure_preview(tid: str, src_path: str, start_sec: int = 0, dur_sec: int = 30) -> str | None:
    """
    Create a 30s MP3 preview with ffmpeg if not present; return preview file path.
    Cached by (tid, src_path, start_sec, dur_sec).
    """
    out_path = preview_path_for(tid)
    if os.path.exists(out_path):
        return out_path
    if not has_ffmpeg():
        return None
    try:
        # Build ffmpeg command: 30s stereo, 44.1kHz, 160 kbps MP3
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-t", str(dur_sec),
            "-i", src_path,
            "-ac", "2",
            "-ar", "44100",
            "-b:a", "160k",
            out_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path if os.path.exists(out_path) else None
    except Exception:
        return None

df = random_unrated(limit=30)

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

            # Try to create/play preview
            preview_btn = st.button("🎧 Generate/Play 30s preview", key=row["id"]+"_preview")
            preview_file = None
            if os.path.exists(preview_path_for(row["id"])) and not preview_btn:
                preview_file = preview_path_for(row["id"])
            elif preview_btn:
                preview_file = ensure_preview(row["id"], row["path"], start_sec=start_sec, dur_sec=dur_sec)

            if preview_file and os.path.exists(preview_file):
                with open(preview_file, "rb") as fh:
                    st.audio(fh.read(), format="audio/mp3")
            else:
                # Fallback: try playing original file directly (works if browser supports codec)
                try:
                    with open(row["path"], "rb") as fh:
                        st.audio(fh.read())
                        st.caption("Playing original file (no preview). If it fails, install ffmpeg for MP3 previews.")
                except Exception:
                    if not has_ffmpeg():
                        st.warning("ffmpeg not found. Install ffmpeg to enable reliable MP3 previews.")
                    else:
                        st.warning("Could not generate or play a preview for this file.")

            stars = st.slider("Stars", 1, 5, 3, key=row["id"])
            if st.button("Save", key=row["id"] + "_save"):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("UPDATE tracks SET stars=? WHERE id=?", (int(stars), row["id"]))
                conn.commit()
                conn.close()
                st.success("Saved!")

st.caption("Tip: click 'Rerun' from Streamlit for a fresh unrated batch.")
