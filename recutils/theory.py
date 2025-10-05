import numpy as np

MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
PITCHES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

CAMEL0T_MAJOR = ["8B","3B","10B","5B","12B","7B","2B","9B","4B","11B","6B","1B"]
CAMEL0T_MINOR = ["5A","12A","7A","2A","9A","4A","11A","6A","1A","8A","3A","10A"]

def estimate_key_from_chroma(chroma: np.ndarray):
    v = chroma.mean(axis=1)
    v = v / (v.sum()+1e-9)
    scores_maj = []
    scores_min = []
    for i in range(12):
        scores_maj.append((i, np.dot(np.roll(v, -i), MAJOR_PROFILE)))
        scores_min.append((i, np.dot(np.roll(v, -i), MINOR_PROFILE)))
        best_maj = max(scores_maj, key=lambda x: x[1])
        best_min = max(scores_min, key=lambda x: x[1])
        if best_maj[1] >= best_min[1]:
            pc = best_maj[0]; key = PITCHES[pc]; mode = "major"; camel = CAMEL0T_MAJOR[pc]
        else:
            pc = best_min[0]; key = PITCHES[pc]+"m"; mode = "minor"; camel = CAMEL0T_MINOR[pc]
    return key, mode, camel


def camelot_neighbors(camel: str):
    num = int(''.join([c for c in camel if c.isdigit()]))
    side = 'A' if 'A' in camel else 'B'
    same = camel
    plus = f"{(num % 12) + 1}{side}"
    minus = f"{(num - 2) % 12 + 1}{side}"
    swap = f"{num}{'B' if side=='A' else 'A'}"
    return {same, plus, minus, swap}
