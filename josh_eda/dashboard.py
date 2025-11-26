"""
dashboard.py - Interactive Sensor Data Dashboard
Uses Dash for a web-based UI that's scalable and AWS-ready.
Iteratively loads cleaned datasets from historical_reference_data repository.

Run:
python dashboard.py
Then open http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import json

# ====================================================================
# DATA LOADER - Iterative Discovery
# ====================================================================

class SensorDataLoader:
    """
    Iteratively loads and caches cleaned sensor files from the 
    historical_reference_data directory structure.
    Discovers all *_CLEANED.xlsx files and their associated *_FAULTS.csv files.
    """
    
    def __init__(self, root_dir: Path = Path("historical_reference_data")):
        self.root_dir = root_dir
        self.samples = {}
        self.faults = {}
        self.test_groups = {}  # Group samples by test type (Box_A, Box_B, etc.)
        self._discover_and_load_samples()
    
    def _discover_and_load_samples(self):
        """
        Iteratively scan for all cleaned files in the repository.
        Organizes them by test/box type for easy navigation.
        """
        if not self.root_dir.exists():
            print(f"Warning: Root directory not found: {self.root_dir}")
            return
        
        # Find all cleaned files recursively
        cleaned_files = list(self.root_dir.rglob("*_CLEANED.xlsx"))
        
        if not cleaned_files:
            print(f"Warning: No cleaned files found in {self.root_dir}")
            return
        
        print(f"Discovering cleaned files in {self.root_dir}...")
        
        for cleaned_file in sorted(cleaned_files):
            sample_name = cleaned_file.stem.replace("_CLEANED", "")
            faults_file = cleaned_file.with_name(sample_name + "_FAULTS.csv")
            
            # Determine test group from path (e.g., Box_A, Box_B, Preg_Chip_A, etc.)
            relative_path = cleaned_file.relative_to(self.root_dir)
            test_group = relative_path.parts[0] if len(relative_path.parts) > 1 else "Uncategorized"
            
            try:
                df = pd.read_excel(cleaned_file)
                self.samples[sample_name] = {
                    "df": df,
                    "path": cleaned_file,
                    "box_type": self._detect_box_type(df),
                    "test_group": test_group
                }
                
                if faults_file.exists():
                    try:
                        fault_df = pd.read_csv(faults_file)
                        self.faults[sample_name] = set(fault_df["Channel"].dropna().astype(str))
                    except Exception as e:
                        print(f"Warning: Could not read faults for {sample_name}: {e}")
                        self.faults[sample_name] = set()
                else:
                    self.faults[sample_name] = set()
                
                # Group by test type
                if test_group not in self.test_groups:
                    self.test_groups[test_group] = []
                self.test_groups[test_group].append(sample_name)
                
                print(f"  ✓ Loaded: {sample_name} (from {test_group})")
                
            except Exception as e:
                print(f"  ✗ Failed to load {sample_name}: {e}")
    
    def _detect_box_type(self, df):
        """Detect Taurus vs Variable from sensor columns."""
        if any(c.startswith("T") and c[1:].isdigit() for c in df.columns):
            return "Taurus"
        elif any(c.startswith("d") and c[1:].isdigit() for c in df.columns):
            return "Variable"
        return "Unknown"
    
    def get_test_groups(self):
        """Return list of test group names."""
        return sorted(self.test_groups.keys())
    
    def get_samples_by_group(self, test_group):
        """Get all samples in a specific test group."""
        return sorted(self.test_groups.get(test_group, []))
    
    def get_sample_list(self):
        """Return all sample names."""
        return sorted(self.samples.keys())
    
    def get_sample(self, sample_name):
        """Get data for a specific sample."""
        return self.samples.get(sample_name, {})
    
    def get_faults(self, sample_name):
        """Get faulty channels for a sample."""
        return self.faults.get(sample_name, set())
    
    def get_test_group_stats(self, test_group):
        """Get summary stats for all samples in a test group."""
        samples_in_group = self.test_groups.get(test_group, [])
        stats = {
            "total_samples": len(samples_in_group),
            "box_types": set(),
            "avg_channels": 0,
            "avg_data_quality": 0,
            "total_faults": 0
        }
        
        if not samples_in_group:
            return stats
        
        for sample_name in samples_in_group:
            sample = self.get_sample(sample_name)
            if sample:
                sensor_cols = [c for c in sample["df"].columns 
                              if (c.startswith("T") or c.startswith("d")) and c[1:].isdigit()]
                faulty = self.get_faults(sample_name)
                
                stats["box_types"].add(sample["box_type"])
                stats["total_faults"] += len(faulty)
                stats["avg_channels"] += len(sensor_cols)
                stats["avg_data_quality"] += ((len(sensor_cols) - len(faulty)) / len(sensor_cols) * 100) if sensor_cols else 0
        
        stats["avg_channels"] /= len(samples_in_group)
        stats["avg_data_quality"] /= len(samples_in_group)
        stats["box_types"] = ", ".join(sorted(stats["box_types"]))
        
        return stats

# ====================================================================
# PLOTLY VISUALIZATION FUNCTIONS
# ====================================================================

def create_multi_panel_plot(df, sample_name, faulty_channels):
    """Create 3-panel plot: Resistance, Temperature, Humidity."""
    sensor_cols = [c for c in df.columns if (c.startswith("T") or c.startswith("d")) and c[1:].isdigit()]
    
    if not sensor_cols:
        fig = go.Figure()
        fig.add_annotation(text="No sensor columns found", showarrow=False)
        return fig
    
    time_sec = df["Time_ms"] / 1000 if "Time_ms" in df.columns else range(len(df))
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Sensor Resistance", "Temperature", "Humidity"),
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Panel 1: Resistance (all channels)
    for col in sensor_cols:
        is_faulty = col in faulty_channels
        fig.add_trace(
            go.Scatter(
                x=time_sec, y=df[col],
                name=f"{col} (Fault)" if is_faulty else col,
                line=dict(
                    color="red" if is_faulty else "lightblue",
                    width=2.5 if is_faulty else 1
                ),
                opacity=0.9 if is_faulty else 0.5,
                legendgroup="channels",
                showlegend=is_faulty  # Only show faulty in legend
            ),
            row=1, col=1
        )
    
    # Panel 2: Temperature
    if "Temperature_C" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=time_sec, y=df["Temperature_C"],
                name="Temperature",
                line=dict(color="darkorange", width=2),
                legendgroup="env"
            ),
            row=2, col=1
        )
        fig.add_hline(y=60, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Target: 60°C")
    
    # Panel 3: Humidity
    if "Humidity_percent" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=time_sec, y=df["Humidity_percent"],
                name="Humidity",
                line=dict(color="green", width=2),
                legendgroup="env"
            ),
            row=3, col=1
        )
    
    fig.update_yaxes(title_text="Resistance (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="Temp (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=3, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    
    fig.update_layout(
        title=f"Sample: {sample_name}",
        height=800,
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig

def create_channel_heatmap(df):
    """Create a heatmap of all channels over time."""
    sensor_cols = [c for c in df.columns if (c.startswith("T") or c.startswith("d")) and c[1:].isdigit()]
    
    if not sensor_cols:
        fig = go.Figure()
        fig.add_annotation(text="No sensor columns found", showarrow=False)
        return fig
    
    # Normalize each channel to 0-1 for better heatmap visibility
    normalized = df[sensor_cols].copy()
    for col in normalized.columns:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val > min_val:
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
    
    fig = go.Figure(
        data=go.Heatmap(
            z=normalized.T,
            x=df["Time_ms"] / 1000 if "Time_ms" in df.columns else range(len(df)),
            y=normalized.columns,
            colorscale="RdYlBu_r",
            colorbar=dict(title="Normalized Response")
        )
    )
    
    fig.update_layout(
        title="Channel Response Heatmap",
        xaxis_title="Time (seconds)",
        yaxis_title="Channels",
        height=600,
        template="plotly_white"
    )
    
    return fig

def create_summary_stats(df, sample_name, faulty_channels, box_type):
    """Create a summary statistics display."""
    sensor_cols = [c for c in df.columns if (c.startswith("T") or c.startswith("d")) and c[1:].isdigit()]
    
    total_channels = len(sensor_cols)
    dead_channels = len(faulty_channels)
    usable_channels = total_channels - dead_channels
    data_quality = round((usable_channels / total_channels * 100), 1) if total_channels > 0 else 0
    
    temp_mean = df["Temperature_C"].mean() if "Temperature_C" in df.columns else None
    temp_std = df["Temperature_C"].std() if "Temperature_C" in df.columns else None
    duration_min = (df["Time_ms"].max() / 1000 / 60) if "Time_ms" in df.columns else None
    
    summary_text = f"""
Sample: {sample_name}
Box Type: {box_type}

Channel Health:
  Total: {total_channels} | Usable: {usable_channels} | Dead: {dead_channels}
  Data Quality: {data_quality}%

Environment:
  Mean Temp: {temp_mean:.1f}°C (±{temp_std:.1f}°C) if temp_mean else "N/A"
  Duration: {duration_min:.1f} min if duration_min else "N/A"
    """
    
    return summary_text

# ====================================================================
# DASH APP
# ====================================================================

loader = SensorDataLoader()

if not loader.get_sample_list():
    print("ERROR: No cleaned files found. Please run batch_clean.py first.")
    exit(1)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Sensor Data Analysis Dashboard", style={"textAlign": "center", "marginBottom": 30}),
        
        # Test Group Selection
        html.Div([
            html.Label("Select Test Group:", style={"fontWeight": "bold", "marginRight": 10}),
            dcc.Dropdown(
                id="test-group-dropdown",
                options=[{"label": g, "value": g} for g in loader.get_test_groups()],
                value=loader.get_test_groups()[0] if loader.get_test_groups() else None,
                style={"width": "100%"}
            )
        ], style={"marginBottom": 20, "width": "100%"}),
        
        # Sample Selection
        html.Div([
            html.Label("Select Sample:", style={"fontWeight": "bold", "marginRight": 10}),
            dcc.Dropdown(
                id="sample-dropdown",
                options=[],
                style={"width": "100%"}
            )
        ], style={"marginBottom": 20, "width": "100%"}),
        
        # Test Group Stats
        html.Div(id="test-group-stats", style={
            "backgroundColor": "#e8f4f8",
            "padding": "15px",
            "borderRadius": "5px",
            "marginBottom": 20,
            "fontFamily": "monospace",
            "fontSize": 12
        }),
        
        # Sample Stats
        html.Div(id="sample-stats", style={
            "backgroundColor": "#f0f0f0",
            "padding": "15px",
            "borderRadius": "5px",
            "marginBottom": 20,
            "fontFamily": "monospace",
            "whiteSpace": "pre-wrap",
            "fontSize": 11
        }),
        
    ], style={"maxWidth": 1200, "margin": "0 auto", "padding": 20}),
    
    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label="Full Time-Series", value="tab-1", children=[
            dcc.Graph(id="main-plot")
        ]),
        dcc.Tab(label="Channel Heatmap", value="tab-2", children=[
            dcc.Graph(id="heatmap-plot")
        ]),
    ], style={"marginTop": 20}),
    
], style={"fontFamily": "Arial, sans-serif", "padding": "20px", "backgroundColor": "#fafafa", "minHeight": "100vh"})

# Callback to update sample dropdown based on test group
@callback(
    Output("sample-dropdown", "options"),
    Output("sample-dropdown", "value"),
    Output("test-group-stats", "children"),
    Input("test-group-dropdown", "value")
)
def update_samples_by_group(selected_group):
    if not selected_group:
        return [], None, "Select a test group"
    
    samples = loader.get_samples_by_group(selected_group)
    stats = loader.get_test_group_stats(selected_group)
    
    stats_text = f"""Test Group: {selected_group}
Total Samples: {stats['total_samples']}
Box Types: {stats['box_types']}
Avg Channels: {stats['avg_channels']:.1f}
Avg Data Quality: {stats['avg_data_quality']:.1f}%
Total Faults: {stats['total_faults']}"""
    
    return (
        [{"label": s, "value": s} for s in samples],
        samples[0] if samples else None,
        stats_text
    )

# Callback to update plots and stats based on sample selection
@callback(
    [Output("main-plot", "figure"),
     Output("heatmap-plot", "figure"),
     Output("sample-stats", "children")],
    Input("sample-dropdown", "value")
)
def update_plots(selected_sample):
    if not selected_sample:
        return go.Figure(), go.Figure(), "No sample selected"
    
    sample_data = loader.get_sample(selected_sample)
    if not sample_data:
        return go.Figure(), go.Figure(), "Sample not found"
    
    df = sample_data.get("df")
    box_type = sample_data.get("box_type")
    faulty_channels = loader.get_faults(selected_sample)
    
    main_fig = create_multi_panel_plot(df, selected_sample, faulty_channels)
    heatmap_fig = create_channel_heatmap(df)
    stats = create_summary_stats(df, selected_sample, faulty_channels, box_type)
    
    return main_fig, heatmap_fig, stats

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Dashboard initialized with {len(loader.get_sample_list())} samples")
    print(f"Test Groups: {', '.join(loader.get_test_groups())}")
    print(f"{'='*60}")
    print("Starting server on http://localhost:8050\n")
    app.run(debug=True, port=8050)