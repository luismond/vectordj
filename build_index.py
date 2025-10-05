import argparse, yaml, os
from recutils.indexer import ingest, build_features, build_faiss_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if os.path.exists(args.config) \
        else yaml.safe_load(open("config.example.yaml"))

    music_dir = cfg["music_dir"]
    sr = int(cfg.get("sample_rate", 22050))
    sec = int(cfg.get("duration_sec", 30))
    m = int(cfg.get("hnsw_m", 32))
    ef = int(cfg.get("hnsw_ef_construction", 200))

    print("== Ingesting catalog ==")
    ingest(music_dir)
    print("== Extracting features ==")
    build_features(sr=sr, sec=sec)
    print("== Building FAISS index ==")
    n = build_faiss_index(hnsw_m=m, ef_c=ef)
    print(f"Done. Indexed {n} tracks.")


if __name__ == "main":
    main()
