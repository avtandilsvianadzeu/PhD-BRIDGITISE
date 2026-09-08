from pathlib import Path
import dwdatareader as dw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

RAW_DIR = Path(
    r"C:\Users\svianadze\OneDrive - zag.si\Desktop\Task for WIM\03_Terenske meritve\surovi podatki"
)
OUTPUT_DIR = RAW_DIR / "processed_output"
OUTPUT_DIR.mkdir(exist_ok=True)

dxd_files = sorted(list(RAW_DIR.glob("*.dxd")))
all_event_stats = []

print(f"Processing {len(dxd_files)} raw DXD files for vehicle crossings...\n")

for fpath in dxd_files:
    print(f"Processing {fpath.name}...")

    with dw.DWFile(str(fpath)) as d:
        data_20k = {}
        for i in range(1, 10):
            ch_key = f"ch {i}"
            try:
                data_20k[ch_key] = d[ch_key].series().values
            except Exception:
                pass

        if not data_20k:
            print(f"  -> Warning: No channels found in {fpath.name}")
            continue

        df = pd.DataFrame(data_20k)
        dt = 1.0 / 20000.0

        # Mean-centered detrending per 10-second chunk to preserve dynamic peaks
        detrended = pd.DataFrame(index=df.index)
        chunk_size = 200000  # 10 seconds at 20 kHz
        for col in df.columns:
            sig = df[col].values
            # Fast baseline subtraction using lowpass filter
            b, a = signal.butter(2, 0.5, btype="lowpass", fs=20000)
            baseline = signal.filtfilt(b, a, sig)
            detrended[col] = sig - baseline

        # Vehicle envelope across key web/flange channels
        avail_cols = [
            c for c in ["ch 1", "ch 2", "ch 7", "ch 8"] if c in detrended.columns
        ]
        envelope = np.abs(detrended[avail_cols]).max(axis=1).values

        # 5.0 microstrain threshold for robust vehicle peak detection
        is_event = (envelope > 5.0).astype(int)

        # Merge close events within 1 second (20,000 samples)
        kernel = np.ones(20000, dtype=int)
        is_event_merged = (
            np.convolve(is_event, kernel, mode="same") > 0
        ).astype(int)

        diffs = np.diff(np.pad(is_event_merged, (1, 1), "constant"))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        file_events = 0
        for s, e in zip(starts, ends):
            duration = (e - s) * dt
            if duration < 0.4 or duration > 15.0:  # Valid crossing window
                continue

            event_win = detrended.iloc[s:e]

            row = {
                "filename": fpath.name,
                "start_sec": s * dt,
                "end_sec": e * dt,
                "duration_sec": duration,
            }

            for ch in event_win.columns:
                row[f"{ch}_min"] = event_win[ch].min()
                row[f"{ch}_max"] = event_win[ch].max()
                row[f"{ch}_mean"] = event_win[ch].mean()
                row[f"{ch}_std"] = event_win[ch].std()
                row[f"{ch}_range"] = event_win[ch].max() - event_win[ch].min()

            all_event_stats.append(row)
            file_events += 1

        print(
            f"  -> Extracted {file_events} vehicle crossings from {fpath.name}"
        )

# Save Master Summary CSV
summary_df = pd.DataFrame(all_event_stats)
csv_out = OUTPUT_DIR / "vehicle_crossings_statistics.csv"
summary_df.to_csv(csv_out, index=False)

print(
    f"\nSUCCESS: Extracted {len(summary_df)} total vehicle events across all files."
)
print(f"Saved master stats to: {csv_out}")

# Plot Summary Figures
if not summary_df.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Plot 1: Load Distribution Bar 5 vs Bar 4
    ax1.scatter(
        summary_df["ch 1_range"],
        summary_df["ch 8_range"],
        alpha=0.7,
        color="#1f77b4",
        edgecolors="k",
        linewidth=0.5,
    )
    max_val = max(
        summary_df["ch 1_range"].max(), summary_df["ch 8_range"].max()
    )
    ax1.plot(
        [0, max_val],
        [0, max_val],
        "r--",
        label="1:1 Equal Load Line",
        linewidth=1.5,
    )
    ax1.set_xlabel(r"Bar 4 Web Range ($\mu m/m$) [ch 1]")
    ax1.set_ylabel(r"Bar 5 Web Range ($\mu m/m$) [ch 8]")
    ax1.set_title("Cross-Bar Response Ratio (Bar 5 vs Bar 4)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Plot 2: Boxplot Analysis (Fixed Matplotlib parameter syntax)
    box_cols = [
        c for c in ["ch 1_range", "ch 8_range", "ch 7_range"] if c in summary_df
    ]
    box_labels = [c.replace("_range", "") for c in box_cols]

    try:
        ax2.boxplot(
            [summary_df[c] for c in box_cols],
            tick_labels=box_labels,
            patch_artist=True,
            boxprops=dict(facecolor="#e0e0e0", color="black"),
        )
    except TypeError:
        ax2.boxplot(
            [summary_df[c] for c in box_cols],
            labels=box_labels,
            patch_artist=True,
            boxprops=dict(facecolor="#e0e0e0", color="black"),
        )

    ax2.set_ylabel(r"Strain Range $\Delta \epsilon$ ($\mu m/m$)")
    ax2.set_title("Strain Range Distribution Across Vehicle Crossings")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vehicle_crossing_summary.png")
    plt.show()