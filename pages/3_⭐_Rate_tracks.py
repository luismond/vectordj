import os, pandas as pd, sqlite3
import streamlit as st

st.title("⭐ Rate tracks")

DB_PATH = os.path.join("data", "tracks.sqlite")
if not os.path.exists(DB_PATH):
    st.error("Database not found. Run `python build_index.py` first.")
    st.stop()

def random_unrated(limit=50):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id, path, title, artist, album, genre, duration "
        "FROM tracks WHERE stars IS NULL ORDER BY RANDOM() LIMIT ?",
        conn,
        params=(limit,),
    )
    conn.close()
    return df

df = random_unrated(limit=30)

if df.empty:
    st.success("All tracks are rated!")
else:
    for _, row in df.iterrows():
        with st.expander(f"{row['artist']} – {row['title']} | {row['album']}"):
            st.text(row["path"])
            stars = st.slider("Stars", 1, 5, 3, key=row["id"])
            if st.button("Save", key=row["id"] + "_save"):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("UPDATE tracks SET stars=? WHERE id=?", (int(stars), row["id"]))
                conn.commit()
                conn.close()
                st.success("Saved!")

st.caption("Tip: click 'Rerun' from Streamlit for a fresh unrated batch.")
