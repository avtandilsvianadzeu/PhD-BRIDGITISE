from pathlib import Path
import numpy as np
import pandas as pd
import dwdatareader as dw

RAW_DIR = Path(r"C:\Users\svianadze\OneDrive - zag.si\Desktop\Task for WIM\03_Terenske meritve\surovi podatki")

# 1. Check Specific Anchor Values (Section 5.1)
anchors = [
    {"file": "dilatacije_2026_09_03_124033.dxd", "win": (446.0, 447.0), "ch": "ch 7", "expected_min": -205.6,
     "expected_max": 52.1},
    {"file": "dilatacije_2026_09_03_124033.dxd", "win": (446.0, 447.0), "ch": "ch 8", "expected_min": -90.1,
     "expected_max": 46.9},
    {"file": "dilatacije_2026_09_03_120120.dxd", "win": (116.5, 117.6), "ch": "ch 7", "expected_min": -140.3,
     "expected_max": 37.9},
    {"file": "dilatacije_2026_09_03_115328.dxd", "win": (26.0, 27.5), "ch": "ch 1", "expected_min": -98.7,
     "expected_max": 68.6},
]

print("=== TASK 2: ANCHOR VALUE VERIFICATION ===")
for a in anchors:
    fpath = RAW_DIR / a["file"]
    with dw.DWFile(str(fpath)) as d:
        ch_data = d[a["ch"]].series()
        # Time window slicing
        mask = (ch_data.index >= a["win"][0]) & (ch_data.index <= a["win"][1])
        segment = ch_data.values[mask]

        c_min, c_max = np.min(segment), np.max(segment)
        print(f"File: {a['file']} | {a['ch']} @ {a['win']}s")
        print(f"  Measured: Min = {c_min:.1f}, Max = {c_max:.1f} µm/m")
        print(f"  Expected: Min = {a['expected_min']}, Max = {a['expected_max']} µm/m\n")


# 2. Noise Check via Robust MAD: 1.4826 * median(|x - median(x)|)
def get_channel_noise(signal):
    med = np.median(signal)
    return 1.4826 * np.median(np.abs(signal - med))