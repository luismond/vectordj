import os, hashlib, numpy as np
from typing import Optional, Dict, Any
import librosa, pyloudnorm as pyln
from mutagen import File as MFile
from .theory import estimate_key_from_chroma

AUDIO_EXTS = (".mp3",".flac",".m4a",".wav",".ogg",".aiff",".aif",".wma",".aac")

def track_id(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()[:16]

def extract_features(path: str, sr_target: int = 22050, sec: int = 30) -> Optional[np.ndarray]:
    try:
        y, sr = librosa.load(path, sr=sr_target, mono=True, duration=sec)
        if len(y) < sr * 5:
            return None
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(y))
        tempo = float(librosa.beat.tempo(y=y, sr=sr, aggregate=np.median))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        _key, _mode, _camel = estimate_key_from_chroma(chroma)
        feat = np.concatenate([
        chroma.mean(1), chroma.std(1),
        mfcc.mean(1), mfcc.std(1),
        spec_cent.mean(1), spec_cent.std(1),
        spec_bw.mean(1), spec_bw.std(1),
        roll.mean(1), roll.std(1),
        zcr.mean(1), zcr.std(1),
        np.array([tempo, lufs])
        ]).astype("float32")
        return feat
    except Exception:
        return None


def read_tags(path: str) -> Dict[str, Any]:
    try:
        a = MFile(path)
        title = artist = album = genre = year = duration = None
        if hasattr(a, "tags") and a.tags:
            title = str(a.tags.get("TIT2", a.tags.get("title", [None])[0] if isinstance(a.tags.get("title"), list) else a.tags.get("title")))
            artist = str(a.tags.get("TPE1", a.tags.get("artist", [None])[0] if isinstance(a.tags.get("artist"), list) else a.tags.get("artist")))
            album = str(a.tags.get("TALB", a.tags.get("album", [None])[0] if isinstance(a.tags.get("album"), list) else a.tags.get("album")))
            genre = str(a.tags.get("TCON", a.tags.get("genre", [None])[0] if isinstance(a.tags.get("genre"), list) else a.tags.get("genre")))
        if hasattr(a, "info"):
            duration = getattr(a.info, "length", None)
        return dict(title=title, artist=artist, album=album, genre=genre, year=year, duration=duration)
    except Exception:
        return dict(title=None, artist=None, album=None, genre=None, year=None, duration=None)


def walk_music_dir(music_dir: str):
    for root, _, files in os.walk(music_dir):
        for f in files:
            if f.lower().endswith(AUDIO_EXTS):
                yield os.path.join(root, f)


def quick_bpm_key(path: str, sr_target: int = 22050, sec: int = 30):
    try:
        y, sr = librosa.load(path, sr=sr_target, mono=True, duration=sec)
        if len(y) < sr * 5:
            return None, None, None
        tempo = float(librosa.beat.tempo(y=y, sr=sr, aggregate=np.median))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key, mode, camel = estimate_key_from_chroma(chroma)
        return tempo, key, camel
    except Exception:
        return None, None, None
