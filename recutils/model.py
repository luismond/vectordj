import os, sqlite3, joblib, numpy as np
from typing import List
from .indexer import DB_PATH, load_feature_matrix
import lightgbm as lgb

MODEL_DIR = "data/model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_stars.pkl")


def _load_labels(ids: List[str]) -> np.ndarray:
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    placeholders = ",".join("?"*len(ids))
    rows = c.execute(f"SELECT stars FROM tracks WHERE id IN ({placeholders})", ids).fetchall()
    conn.close()
    y = np.array([r[0] if r and r[0] is not None else np.nan for r in rows], dtype=float)
    return y


def train_model() -> int:
    X, ids = load_feature_matrix()
    y = _load_labels(ids)
    mask = ~np.isnan(y)
    if mask.sum() < 50:
        return 0

    Xtr = X[mask]; ytr = y[mask]
    params = dict(objective="regression", n_estimators=600, learning_rate=0.05, num_leaves=63)
    model = lgb.LGBMRegressor(**params)
    model.fit(Xtr, ytr)
    joblib.dump({"model": model, "ids": ids}, MODEL_PATH)
    return int(mask.sum())

def has_model() -> bool:
    return os.path.exists(MODEL_PATH)


def predict_scores(vecs: np.ndarray) -> np.ndarray:
    if not has_model():
        raise RuntimeError("Model not trained")
    blob = joblib.load(MODEL_PATH)
    model = blob["model"]
    return model.predict(vecs)
