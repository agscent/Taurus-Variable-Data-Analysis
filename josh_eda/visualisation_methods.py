import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 12

def create_visuals(cleaned_path: Path):
    df = pd.read_excel(cleaned_path)
    sample_name = cleaned_path.stem.replace("_CLEANED", "")
    save_dir = cleaned_path.parent / "plots"
    save_dir.mkdir(exist_ok=True)

    # Filter sensor columns: T1 to T28
    sensor_cols = [f"T{i}" for i in range(1, 29) if f"T{i}" in df.columns]
    if len(sensor_cols) == 0:
        print("No T1–T28 columns found!")
        return

    time = df["Time_ms"] / 1000  # convert to seconds

    # 1. All 28 channels – raw response over time
    """
    Full trace overlay: 
    instantly spot dead channels, outliers, 
    or abnormal breath/purge patterns.
    """
    plt.figure()
    for col in sensor_cols:
        plt.plot(time, df[col], label=col, linewidth=1.2)
    plt.title(f"{sample_name}\nAll 28 Sensor Channels – Raw Response")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Resistance (Ω)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_dir / f"01_{sample_name}_all_channels.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Heatmap of sensor response
    """
    Heatmap: 
    see at a glance which channels respond to 
    breath and which stay flat or drift.
    """
    plt.figure()
    sns.heatmap(df[sensor_cols].T, cmap="viridis", cbar_kws={'label': 'Resistance (Ω)'})
    plt.title(f"{sample_name}\nSensor Response Heatmap (T1–T28)")
    plt.xlabel("Time Point (index)")
    plt.ylabel("Sensor Channel")
    plt.tight_layout()
    plt.savefig(save_dir / f"02_{sample_name}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Baseline vs Peak response per channel
    """
    Baseline vs Peak bar plot: 
    absolute resistance values – reveals 
    scale differences and sensor saturation.
    """
    baseline = df[df["Name"] == "OPEN"][sensor_cols].mean()
    peak = df[df["Name"] == "BREATH"].max() if "BREATH" in df["Name"].values else df[sensor_cols].max()

    x = np.arange(len(sensor_cols))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, baseline, width, label="Baseline (OPEN)", color="skyblue")
    plt.bar(x + width/2, peak, width, label="Peak Response", color="salmon")
    plt.xticks(x, sensor_cols, rotation=90)
    plt.title(f"{sample_name}\nBaseline vs Peak Response per Channel")
    plt.ylabel("Resistance (Ω)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"03_{sample_name}_baseline_vs_peak.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Relative response ΔR/R₀ per channel
    """
    ΔR/R₀ bar plot: 
    the gold standard for sensor performance – directly 
    comparable across chips, boxes, and days.
    """
    delta_r = peak - baseline
    rel_response = delta_r / baseline

    plt.figure()
    plt.bar(sensor_cols, rel_response, color="purple", alpha=0.8)
    plt.xticks(rotation=90)
    plt.title(f"{sample_name}\nRelative Response ΔR/R₀ per Channel")
    plt.ylabel("ΔR / R₀")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_dir / f"04_{sample_name}_relative_response.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Drift in stable phases (OPEN + PURGE)
    """
    Drift (%): 
    measures long-term stability – critical for knowing 
    if a chip/box is degrading or trustworthy over multiple runs.
    """
    stable_mask = df["Name"].isin(["OPEN", "PURGE"])
    stable_df = df[stable_mask]
    if len(stable_df) > 100:
        early = stable_df.iloc[:50][sensor_cols].mean()
        late = stable_df.iloc[-50:][sensor_cols].mean()
        drift = (late - early) / early * 100

        plt.figure()
        plt.bar(sensor_cols, drift, color="orange")
        plt.xticks(rotation=90)
        plt.title(f"{sample_name}\nDrift in Stable Phases (% change)")
        plt.ylabel("Drift (%)")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(save_dir / f"05_{sample_name}_drift.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Top 10 most responsive channels
    """
    Top-10 ranking (|ΔR/R₀|): 
    immediately tells you which sensors are actually detecting pregnancy VOCs 
    vs noise – drives material selection and ML feature shortlisting.
    """
    plt.figure()
    top10 = rel_response.abs().sort_values(ascending=False).head(10)
    plt.bar(top10.index, top10.values, color="green", alpha=0.7)
    plt.title(f"{sample_name}\nTop 10 Most Responsive Channels (|ΔR/R₀|)")
    plt.ylabel("|ΔR / R₀|")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / f"06_{sample_name}_top10_channels.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"6 plots saved to:\n{save_dir}\n")

def create_visuals_with_faults(cleaned_path: Path):
    df = pd.read_excel(cleaned_path)
    sample_name = cleaned_path.stem.replace("_CLEANED", "")
    faults_file = cleaned_path.parent / f"{cleaned_path.stem.replace('_CLEANED', '')}_FAULTS.csv"
    
    faulty_channels = set()
    if faults_file.exists():
        fault_df = pd.read_csv(faults_file)
        faulty_channels = set(fault_df["Channel"].dropna())

    save_dir = cleaned_path.parent / "plots"
    save_dir.mkdir(exist_ok=True)
    sensor_cols = [f"T{i}" for i in range(1, 29) if f"T{i}" in df.columns]
    time = df["Time_ms"] / 1000

    # Reuse your existing plots, but color faulty channels RED
    plt.figure()
    for col in sensor_cols:
        color = "red" if col in faulty_channels else "tab:blue"
        alpha = 0.9 if col in faulty_channels else 0.6
        linewidth = 2.5 if col in faulty_channels else 1.0
        plt.plot(time, df[col], label=col, color=color, alpha=alpha, linewidth=linewidth)
    
    plt.title(f"{sample_name}\nAll Channels — Faulty = RED", fontsize=16, fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel("Resistance (Ω)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / f"00_{sample_name}_with_faults_highlighted.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Fault-highlighted plot saved for {sample_name}")