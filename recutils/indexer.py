import os, json, sqlite3, numpy as np
from typing import List, Tuple
from tqdm import tqdm
import faiss
from .features import track_id, extract_features, read_tags, walk_music_dir, quick_bpm_key

DATA_DIR = "data"
FEAT_DIR = os.path.join(DATA_DIR, "features")
INDEX_DIR = os.path.join(DATA_DIR, "index")
UMAP_DIR = os.path.join(DATA_DIR, "umap")
DB_PATH = os.path.join(DATA_DIR, "tracks.sqlite")

os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(UMAP_DIR, exist_ok=True)


def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS tracks(
    id TEXT PRIMARY KEY,
    path TEXT UNIQUE,
    title TEXT, artist TEXT, album TEXT, year INT, genre TEXT,
    duration REAL, stars INT,
    bpm REAL, key TEXT, camelot TEXT
    )""")
    conn.commit(); conn.close()
    _migrate_add_columns()


def _migrate_add_columns():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    try: c.execute("ALTER TABLE tracks ADD COLUMN bpm REAL")
    except Exception: pass
    try: c.execute("ALTER TABLE tracks ADD COLUMN key TEXT")
    except Exception: pass
    try: c.execute("ALTER TABLE tracks ADD COLUMN camelot TEXT")
    except Exception: pass
    conn.commit(); conn.close()


def ingest(music_dir: str):
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    paths = list(walk_music_dir(music_dir))
    print(f'Walked through {len(paths)} paths')
    for p in tqdm(paths, desc="Cataloging"):
        tid = track_id(p)
        tags = read_tags(p)
        c.execute("""INSERT OR IGNORE INTO tracks(id,path,title,artist,album,year,genre,duration,stars,bpm,key,camelot)
        VALUES(?,?,?,?,?,?,?,?,NULL,NULL,NULL,NULL)""",
        (tid, p, tags["title"], tags["artist"], tags["album"], tags["year"], \
         tags["genre"], tags["duration"]))
    conn.commit(); conn.close()


def build_features(sr: int=22050, sec: int=30):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute("SELECT id, path FROM tracks").fetchall()
    conn.close()
    for tid, path in tqdm(rows, desc="Extracting features"):
        out = os.path.join(FEAT_DIR, f"{tid}.npy")
        if os.path.exists(out):
            continue
        feat = extract_features(path, sr_target=sr, sec=sec)
        if feat is not None:
            np.save(out, feat)
        bpm, key, camel = quick_bpm_key(path, sr_target=sr, sec=sec)
        conn2 = sqlite3.connect(DB_PATH); c2 = conn2.cursor()
        c2.execute("UPDATE tracks SET bpm=?, key=?, camelot=? WHERE id=?", (bpm, key, camel, tid))
        conn2.commit(); conn2.close()


def load_feature_matrix() -> Tuple[np.ndarray, List[str]]:
    files = [f for f in os.listdir(FEAT_DIR) if f.endswith(".npy")]
    ids = [os.path.splitext(f)[0] for f in files]
    X = np.stack([np.load(os.path.join(FEAT_DIR, f)) for f in files], axis=0)
    return X.astype("float32"), ids


def build_faiss_index(hnsw_m: int=32, ef_c: int=200):
    X, ids = load_feature_matrix()
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, hnsw_m)
    index.hnsw.efConstruction = ef_c
    index.add(X)
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss_hnsw.index"))
    json.dump(ids, open(os.path.join(INDEX_DIR, "row_ids.json"), "w"))
    return len(ids)


def query_index(vec, k=25):
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss_hnsw.index"))
    ids = json.load(open(os.path.join(INDEX_DIR, "row_ids.json")))
    v = vec / (np.linalg.norm(vec)+1e-9)
    D, I = index.search(v[None,:].astype("float32"), k)
    return [(ids[i], float(1 - D[0, j])) for j, i in enumerate(I[0])]


def id_to_track(tid: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    row = c.execute("SELECT id, path, title, artist, album, genre, duration, stars, bpm, key, camelot FROM tracks WHERE id=?", (tid,)).fetchone()
    conn.close()
    return row


def lookup_by_path(path: str) -> str:
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    row = c.execute("SELECT id FROM tracks WHERE path=?", (path,)).fetchone()
    conn.close()
    return row[0] if row else None


def ids_to_meta(ids: List[str]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    placeholders = ",".join("?"*len(ids))
    rows = c.execute(f"SELECT id, path, title, artist, album, genre, duration, stars, bpm, key, camelot FROM tracks WHERE id IN ({placeholders})", ids).fetchall()
    conn.close()
    x = {
        r[0]: dict(id=r[0], path=r[1], title=r[2], artist=r[3], album=r[4], genre=r[5], \
                   duration=r[6], stars=r[7], bpm=r[8], key=r[9], camelot=r[10]) for r in rows
        }
    return x


def query_index_filtered(vec, k=50, bpm_center=None, bpm_tolerance=6.0,
                         camelot=None, camelot_mode="compatible"):
    base = query_index(vec, k=k*3)
    ids = [tid for tid, sim in base]
    meta = ids_to_meta(ids)

    def camel_ok(cand):
        if camelot is None: return True
        c = meta[cand].get("camelot")
        if not c: return False
        if camelot_mode == "same": return c == camelot
        from .theory import camelot_neighbors
        return c in camelot_neighbors(camelot)

    def bpm_ok(cand):
        if bpm_center is None: return True
        b = meta[cand].get("bpm")
        if b is None: return False
        return abs(float(b) - float(bpm_center)) <= float(bpm_tolerance)

    filtered = [(tid, sim) for tid, sim in base if camel_ok(tid) and bpm_ok(tid)]
    return filtered[:k]
