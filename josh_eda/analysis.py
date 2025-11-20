from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

root = Path("historical_reference_data")
results = []

print("TAURUS PREGNANCY CHIP — FULL DIAGNOSTIC REPORT (Nov 2025)\n")

for cleaned_file in sorted(root.rglob("*_CLEANED.xlsx")):
    faults_file = cleaned_file.with_name(cleaned_file.stem.replace("_CLEANED", "") + "_FAULTS.csv")
    
    if not faults_file.exists():
        print(f"Warning: No faults file for {cleaned_file.name}")
        continue

    df = pd.read_excel(cleaned_file)
    df_faults = pd.read_csv(faults_file)
    
    sensor_cols = [c for c in df.columns if c.startswith("T") and len(c) > 1 and c[1:].isdigit()]
    data = df[sensor_cols].copy()
    
    # Baseline subtraction
    baseline = data.iloc[:50].median()
    normalised = data - baseline
    response_strength = normalised.max() - normalised.min()
    
    total_faults = len(df_faults)
    is_calibration = any(x in cleaned_file.parts for x in ["Box_A_B_Test", "BOX_A", "BOX_B"])

    results.append({
        "sample": cleaned_file.stem.replace("_CLEANED", ""),
        "type": "Calibration" if is_calibration else "Cow Breath",
        "dead_channels": total_faults,
        "mean_response_kΩ": response_strength.mean() / 1000,
        "usable_channels": 28 - total_faults,
        "data_quality_%": round((28 - total_faults) / 28 * 100, 1)
    })

df_results = pd.DataFrame(results)

# Stats
breath = df_results[df_results["type"] == "Cow Breath"]["dead_channels"]
calib = df_results[df_results["type"] == "Calibration"]["dead_channels"]
t_stat, p_val = stats.ttest_ind(breath, calib, equal_var=False)

print("="*90)
print("FINAL SUMMARY")
print("="*90)
print(df_results)
print(f"\nCow Breath chips: average {breath.mean():.1f} dead channels")
print(f"Calibration runs: average {calib.mean():.1f} dead channels")
print(f"T-test: t = {t_stat:.2f}, p = {p_val:.2e} → {'EXTREMELY SIGNIFICANT' if p_val < 1e-6 else 'Significant'}")

# Plot
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

plt.figure(figsize=(14, 8))
colors = df_results["type"].map({"Calibration": "tab:red", "Cow Breath": "tab:blue"})
plt.bar(range(len(df_results)), df_results["dead_channels"], color=colors, alpha=0.8)
plt.xticks(range(len(df_results)), df_results["sample"], rotation=45, ha="right")
plt.ylabel("Dead / Faulty Channels")
plt.title("Taurus Pregnancy Chip Health vs Calibration Standards (Nov 2025)\nBlue = Cow Breath | Red = Box A/B Calibration", fontsize=14)
plt.axhline(10, color="orange", linestyle="--", linewidth=2, label="10/28 = unacceptable")
plt.legend()
plt.tight_layout()
output_plot = results_dir / "TAURUS_CHIP_HEALTH_NOV2025.png"
plt.savefig(output_plot, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFinal plot saved → {output_plot}")