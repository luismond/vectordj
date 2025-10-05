# pip install mutagen librosa pyloudnorm faiss-cpu sqlite-utils tqdm
import os, json, hashlib, sqlite3, numpy as np
from tqdm import tqdm
from mutagen import File as MFile
import librosa, pyloudnorm as pyln
import faiss


#%%
MUSIC_DIR = "/home/user/Music"
DB_PATH = "tracks.sqlite"
FEAT_DIR = "features"
os.makedirs(FEAT_DIR, exist_ok=True)

#%%

def track_id(path):
    return hashlib.md5(path.encode("utf-8")).hexdigest()[:16]

def extract_features(path, sr_target=22050, sec=30):
    y, sr = librosa.load(path, sr=sr_target, mono=True, duration=sec)
    # basic guards
    if len(y) < sr * 5:
        return None
    # loudness (approx via RMS→LUFS)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y))
    # tempo
    tempo = float(librosa.beat.tempo(y=y, sr=sr, aggregate=np.median))
    # chroma & MFCC summary
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    feat = np.concatenate([
        chroma.mean(1), chroma.std(1),
        mfcc.mean(1), mfcc.std(1),
        spec_cent.mean(1), spec_cent.std(1),
        np.array([tempo, lufs])
    ]).astype("float32")
    return feat

# %%
# 1) Build catalog + features
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS tracks(
  id TEXT PRIMARY KEY, path TEXT, title TEXT, artist TEXT, album TEXT,
  year INT, genre TEXT, duration REAL, stars INT DEFAULT NULL
)""")

paths = []
for root, _, files in os.walk(MUSIC_DIR):
    for f in files:
        if f.lower().endswith((".mp3",".flac",".m4a",".wav",".ogg",".aiff",".aif")):
            paths.append(os.path.join(root,f))
#%%

for p in tqdm(paths):
    try:
        tid = track_id(p)
        audio = MFile(p)
        title = getattr(audio.tags, "get", lambda k, d=None: d)("TIT2", None) if hasattr(audio, "tags") else None
        artist = audio.tags.get("TPE1", None) if hasattr(audio, "tags") else None
        album = audio.tags.get("TALB", None) if hasattr(audio, "tags") else None
        year = None
        genre = audio.tags.get("TCON", None) if hasattr(audio, "tags") else None
        duration = getattr(audio.info, "length", None) if hasattr(audio, "info") else None
    
        cur.execute("INSERT OR IGNORE INTO tracks(id,path,title,artist,album,year,genre,duration) VALUES(?,?,?,?,?,?,?,?)",
                    (tid,p,str(title) if title else None,str(artist) if artist else None,
                     str(album) if album else None,year,str(genre) if genre else None,duration))
    
        feat_path = os.path.join(FEAT_DIR, f"{tid}.npy")
        if not os.path.exists(feat_path):
            try:
                feat = extract_features(p)
                if feat is not None:
                    np.save(feat_path, feat)
            except Exception:
                pass
    except:
        print(f'error in {p}')

conn.commit()

# %%
# 2) Build FAISS index (cosine via L2 on normalized vectors)
feat_files = [os.path.join(FEAT_DIR,f) for f in os.listdir(FEAT_DIR) if f.endswith(".npy")]
X = np.stack([np.load(f) for f in feat_files])
# map row -> track id
ids = [os.path.splitext(os.path.basename(f))[0] for f in feat_files]

# normalize
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
d = X.shape[1]
index = faiss.IndexHNSWFlat(d, 32)
index.hnsw.efConstruction = 200
index.add(X)

faiss.write_index(index, "index_timbre.faiss")
json.dump(ids, open("index_ids.json","w"))
print("Indexed", len(ids), "tracks.")

#%%

import json, faiss, numpy as np
index = faiss.read_index("index_timbre.faiss")
ids = json.load(open("index_ids.json"))

def neighbors(vector, k=20):
    v = vector / (np.linalg.norm(vector)+1e-9)
    D,I = index.search(v[None,:].astype("float32"), k)
    return [(ids[i], float(1-D[0,idx])) for idx,i in enumerate(I[0])]

# %%



