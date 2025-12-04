if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Find the file - handles spaces/underscores automatically
    root = Path("historical_reference_data")
    
    # Search for the file
    files = list(root.rglob("*514*Disease*variable*CLEANED.xlsx"))
    
    if not files:
        print("File not found! Searching for any variable box files...")
        files = list(root.rglob("*variable*CLEANED.xlsx"))
        print(f"Found {len(files)} variable box files:")
        for f in files:
            print(f"  {f}")
    
    if files:
        file_path = files[0]
        print(f"\nUsing: {file_path}\n")
        
        df = pd.read_excel(file_path)
        
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}\n")
        
        # Check if 'd1' exists
        if 'd1' in df.columns:
            # Plot raw data for channel d1
            plt.figure(figsize=(14, 6))
            
            plt.subplot(2, 1, 1)
            plt.plot(df.index, df['d1'], linewidth=0.5)
            plt.title("Raw d1 data - Full timeseries")
            plt.xlabel("Index")
            plt.ylabel("Resistance (Ω)")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            # Plot first 100 points zoomed in
            plt.plot(df.index[:100], df['d1'].iloc[:100], 'o-', markersize=3)
            plt.title("First 100 points (zoomed)")
            plt.xlabel("Index")
            plt.ylabel("Resistance (Ω)")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('variable_box_diagnostic.png', dpi=150)
            plt.show()
            
            print("d1 statistics:")
            print(df['d1'].describe())
            print(f"\nSignal range: {df['d1'].max() - df['d1'].min():.2f} Ω")
        else:
            print("No 'd1' column found!")
        
        # Show what the Name column contains
        if 'Name' in df.columns:
            print("\n" + "="*60)
            print("Sequence names in order:")
            print("="*60)
            print(df[['Name']].head(30))
            
            print("\n" + "="*60)
            print("Unique sequence names:")
            print("="*60)
            for name in df['Name'].unique():
                count = len(df[df['Name'] == name])
                print(f"  {name:40s} : {count:4d} rows")
        else:
            print("\n⚠️ WARNING: No 'Name' column found!")
            print("Available columns:", df.columns.tolist())
    else:
        print("❌ No variable box files found!")