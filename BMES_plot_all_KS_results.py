#%%

import os
import pandas as pd
import numpy as np
import scienceplots

# Read the saved KS signal and event indices CSV file and plot it
import matplotlib.pyplot as plt

# Path to the saved CSV file
csv_dir = os.path.dirname(__file__)

# List all relevant CSV files
csv_files = [f for f in os.listdir(csv_dir) if f.startswith('KS_and_events_') and f.endswith('_after1000s.csv')]

plt.style.use('science')
with plt.style.context('science'):


    plt.figure()

    num_signals = len(csv_files)
    offset = 5  # Increase vertical offset for better separation

    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(file_path)
        # Normalize the signal to zero mean and unit variance for better visibility
        ks_signal = df['ks_signal']
        ks_norm = (ks_signal - ks_signal.mean()) / ks_signal.std()

        # Find the Experiment Start index
        start_idx_col = 'Experiment Start_idx'
        if start_idx_col in df.columns and pd.notna(df[start_idx_col][0]):
            start_idx = int(df[start_idx_col][0])
        else:
            continue  # Skip if no start index

        # Slice the signal and time from Experiment Start
        ks_norm = ks_norm[start_idx:]
        time_zeroed = df['time'][start_idx:] - df['time'][start_idx]
        y_offset = i * offset

        plt.plot(time_zeroed, ks_norm + y_offset, linewidth=3, alpha=0.6, label=csv_file.replace('_after1000s.csv', ''), color='b')

        # Show the max point with a red dot (after start)
        max_pos = ks_norm.values.argmax()
        plt.plot(time_zeroed.iloc[max_pos], ks_norm.iloc[max_pos] + y_offset, 'ro', markersize=6, label=None)

        # Overlay event markers as short dash lines on the signal only
        event_names = ['Experiment Start', 'Experiment End', 'Amazon Truck', 'Construction 1 Start',]
        for event in event_names:
            idx_col = f'{event}_idx'
            if idx_col in df.columns and pd.notna(df[idx_col][0]):
                idx = int(df[idx_col][0])
                if idx >= start_idx and idx < len(df['time']):
                    x = df['time'][idx] - df['time'][start_idx]
                    y = ks_norm.iloc[idx - start_idx] + y_offset
                    y_min = y - 0.3
                    y_max = y + 0.3
                    color = 'orange'
                    if event in ['Amazon Truck', 'Construction 1 Start']:
                        color = 'green'
                    plt.plot([x, x], [y_min, y_max], color=color, linestyle='-', alpha=0.9, linewidth=3)


plt.xlabel('Time (s)')
plt.ylabel('KS Score of Subjects')
plt.gca().axes.get_yaxis().set_ticks([])
# plt.title('Standardized KS Signal and Events for All Subjects (after 1000s)')
plt.tight_layout()
plt.show()

# Save the figure in high resolution
plt.savefig(os.path.join(csv_dir, 'KS_signals_all_subjects.png'), dpi=600, bbox_inches='tight')

# %%
# Calculate and print the time difference between max point and event markers for each subject
time_diffs = []

for i, csv_file in enumerate(csv_files):
    file_path = os.path.join(csv_dir, csv_file)
    df = pd.read_csv(file_path)
    ks_signal = df['ks_signal']
    ks_norm = (ks_signal - ks_signal.mean()) / ks_signal.std()

    start_idx_col = 'Experiment Start_idx'
    if start_idx_col in df.columns and pd.notna(df[start_idx_col][0]):
        start_idx = int(df[start_idx_col][0])
    else:
        continue

    ks_norm = ks_norm[start_idx:]
    time_zeroed = df['time'][start_idx:] - df['time'][start_idx]

    # Max point (red dot)
    max_pos = ks_norm.values.argmax()
    max_time = time_zeroed.iloc[max_pos]

    # Amazon Truck event
    amazon_idx_col = 'Amazon Truck_idx'
    if amazon_idx_col in df.columns and pd.notna(df[amazon_idx_col][0]):
        amazon_idx = int(df[amazon_idx_col][0])
        if amazon_idx >= start_idx and amazon_idx < len(df['time']):
            amazon_time = df['time'][amazon_idx] - df['time'][start_idx]
        else:
            continue
    else:
        continue

    # Construction 1 Start event
    construction_idx_col = 'Construction 1 Start_idx'
    if construction_idx_col in df.columns and pd.notna(df[construction_idx_col][0]):
        construction_idx = int(df[construction_idx_col][0])
        if construction_idx >= start_idx and construction_idx < len(df['time']):
            construction_time = df['time'][construction_idx] - df['time'][start_idx]
        else:
            continue
    else:
        continue

    # Calculate time differences
    diff_amazon = max_time - amazon_time
    diff_construction = max_time - construction_time
    time_diffs.append((diff_amazon, diff_construction))
    print(f"{csv_file.replace('_after1000s.csv', '')}: Max - Amazon Truck = {diff_amazon:.2f}s, Max - Construction 1 Start = {diff_construction:.2f}s")

if time_diffs:
    avg_amazon = np.mean([d[0] for d in time_diffs])
    avg_construction = np.mean([d[1] for d in time_diffs])
    print(f"\nAverage Max - Amazon Truck: {avg_amazon:.2f}s")
    print(f"Average Max - Construction 1 Start: {avg_construction:.2f}s")
else:
    print("No valid subjects found for time difference calculation.")


# %%
