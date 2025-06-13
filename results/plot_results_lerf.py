import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Kernel configurations from generate_sweep_yamls_lerf.py
KERNEL_CONFIGS = {
    4:   [1, 2, 4, 8, 16, 32, 64, 128],
    8:   [1, 2, 4, 8, 16, 32, 64],
    16:  [1, 2, 4, 8, 16, 32],
    32:  [1, 2, 4, 8, 16],
    64:  [1, 2, 4, 8],
    128: [1, 2, 4],
    256: [1, 2],
    512: [1],
}

SEEDS = [0, 1, 2]

def verify_missing_runs(df):
    """
    Verify which runs are missing based on the kernel configurations.
    """
    print("\n=== MISSING RUNS ANALYSIS ===")
    
    # Check for each model type (Conv-RNN and TCN)
    for model_type, enable_rec in [("Conv-RNN", True), ("TCN", False)]:
        print(f"\n{model_type} missing runs:")
        missing_count = 0
        
        for kernel_size, dilations in KERNEL_CONFIGS.items():
            for dilation in dilations:
                for seed in SEEDS:
                    # Check if this combination exists
                    mask = (
                        (df['kernel_size'] == kernel_size) & 
                        (df['constant_dilation'] == dilation) & 
                        (df['seed'] == seed) & 
                        (df['enable_rec'] == enable_rec) & 
                        (df['enable_conv'] == True)
                    )
                    
                    if not df[mask].empty:
                        # Check if the run is finished (not crashed or running)
                        finished_mask = mask & (df['State'] == 'finished')
                        if df[finished_mask].empty:
                            print(f"  - k={kernel_size}, d={dilation}, seed={seed}: {df[mask]['State'].iloc[0]}")
                            missing_count += 1
                    else:
                        print(f"  - k={kernel_size}, d={dilation}, seed={seed}: missing")
                        missing_count += 1
        
        print(f"Total missing {model_type} runs: {missing_count}")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Expected total runs per model type: {sum(len(dilations) * len(SEEDS) for dilations in KERNEL_CONFIGS.values())}")
    print(f"Actual finished runs Conv-RNN: {len(df[(df['enable_rec'] == True) & (df['enable_conv'] == True) & (df['State'] == 'finished')])}")
    print(f"Actual finished runs TCN: {len(df[(df['enable_rec'] == False) & (df['enable_conv'] == True) & (df['State'] == 'finished')])}")

# Read the CSV file
df = pd.read_csv('lerf.csv')
print(f"Initial number of rows: {len(df)}")
print(f"Number of crashed entries: {len(df[df['State'] == 'crashed'])}")
df = df[df['State'] != 'crashed']

# Verify missing runs
verify_missing_runs(df)

# Get the unique values actually present in the data
available_kernel_sizes = sorted(df['kernel_size'].unique())
available_dilations = sorted(df['constant_dilation'].unique())

print(f"Available kernel sizes: {available_kernel_sizes}")
print(f"Available dilations: {available_dilations}")

# Create empty matrices to store the results
# One for RNNs with convolution (enable_rec=True) and one for TCNs without RNN (enable_rec=False)
conv_rnn_matrix = np.zeros((len(available_kernel_sizes), len(available_dilations)))
conv_rnn_matrix.fill(np.nan)  # Fill with NaN initially

tcn_matrix = np.zeros((len(available_kernel_sizes), len(available_dilations)))
tcn_matrix.fill(np.nan)  # Fill with NaN initially

# Create matrices for standard deviation and number of parameters
conv_rnn_std = np.zeros((len(available_kernel_sizes), len(available_dilations)))
conv_rnn_std.fill(np.nan)
tcn_std = np.zeros((len(available_kernel_sizes), len(available_dilations)))
tcn_std.fill(np.nan)

conv_rnn_params = np.zeros((len(available_kernel_sizes), len(available_dilations)))
conv_rnn_params.fill(np.nan)
tcn_params = np.zeros((len(available_kernel_sizes), len(available_dilations)))
tcn_params.fill(np.nan)

# Group by kernel_size and constant_dilation, and calculate mean best_val_acc
for i, k in enumerate(available_kernel_sizes):
    for j, d in enumerate(available_dilations):
        # Filter for RNNs with convolution (Conv-RNN)
        conv_rnn_df = df[(df['kernel_size'] == k) & (df['constant_dilation'] == d) & 
                         df['enable_rec'] & df['enable_conv']]
        if not conv_rnn_df.empty:
            conv_rnn_matrix[i, j] = conv_rnn_df['best_val_acc'].mean() * 100  # Convert to percent
            conv_rnn_std[i, j] = conv_rnn_df['best_val_acc'].std() * 100      # Convert to percent
            conv_rnn_params[i, j] = conv_rnn_df['n_params'].mean()
            print(f"Conv-RNN - k={k},\td={d}:\t {len(conv_rnn_df)} entries, mean={conv_rnn_matrix[i, j]:.2f}%, std={conv_rnn_std[i, j]:.2f}%, params={int(conv_rnn_params[i, j])}")

for i, k in enumerate(available_kernel_sizes):
    for j, d in enumerate(available_dilations):        
        # Filter for TCN (RNNs without recurrent connections, just convolutions)
        tcn_df = df[(df['kernel_size'] == k) & (df['constant_dilation'] == d) & 
                    ~df['enable_rec'] & df['enable_conv']]
        if not tcn_df.empty:
            tcn_matrix[i, j] = tcn_df['best_val_acc'].mean() * 100  # Convert to percent
            tcn_std[i, j] = tcn_df['best_val_acc'].std() * 100      # Convert to percent
            tcn_params[i, j] = tcn_df['n_params'].mean()
            print(f"TCN\t - k={k},\td={d}:\t {len(tcn_df)} entries, mean={tcn_matrix[i, j]:.2f}%, std={tcn_std[i, j]:.2f}%, params={int(tcn_params[i, j])}")

# Function to create annotation text with mean, std, and params
def create_annotation(mean, std, params):
    if np.isnan(mean):
        return ""
    return f"{mean:.1f}±{std:.1f}%\n({int(params)})"

# Create annotation matrices
conv_rnn_annot = np.empty_like(conv_rnn_matrix, dtype=object)
tcn_annot = np.empty_like(tcn_matrix, dtype=object)

for i in range(conv_rnn_matrix.shape[0]):
    for j in range(conv_rnn_matrix.shape[1]):
        conv_rnn_annot[i, j] = create_annotation(conv_rnn_matrix[i, j], conv_rnn_std[i, j], conv_rnn_params[i, j])
        tcn_annot[i, j] = create_annotation(tcn_matrix[i, j], tcn_std[i, j], tcn_params[i, j])

# Invert the matrices and annotations (so larger kernel sizes appear at the top)
conv_rnn_matrix = conv_rnn_matrix[::-1]
tcn_matrix = tcn_matrix[::-1]
conv_rnn_annot = conv_rnn_annot[::-1]
tcn_annot = tcn_annot[::-1]

# Set px for scalable sizing
px = 1 / plt.rcParams['figure.dpi']

# Define all sizes using px
fig_width = 2000 * px
fig_height = 1000 * px
base_font_size = 800*px
annotation_font_size = 1100*px
title_font_size = 1200*px
label_font_size = 1000*px

plt.rcParams.update({'font.size': base_font_size})

# Create a figure and set its size
plt.figure(figsize=(fig_width, fig_height))

# Create subplots
plt.subplot(1, 2, 1)
ax1 = sns.heatmap(conv_rnn_matrix, annot=conv_rnn_annot, fmt="", cmap="YlGnBu", 
                 xticklabels=available_dilations, yticklabels=available_kernel_sizes[::-1],
                 vmin=40, vmax=85, annot_kws={"size": annotation_font_size})
plt.title('CD_ConvRNN: Mean Best Validation Accuracy', fontsize=title_font_size)
plt.xlabel('Constant Dilation', fontsize=label_font_size)
plt.ylabel('Kernel Size', fontsize=label_font_size)

plt.subplot(1, 2, 2)
ax2 = sns.heatmap(tcn_matrix, annot=tcn_annot, fmt="", cmap="YlGnBu", 
                 xticklabels=available_dilations, yticklabels=available_kernel_sizes[::-1],
                 vmin=40, vmax=85, annot_kws={"size": annotation_font_size})
plt.title('CD_TCN: Mean Best Validation Accuracy', fontsize=title_font_size)
plt.xlabel('Constant Dilation', fontsize=label_font_size)
plt.ylabel('Kernel Size', fontsize=label_font_size)

# Display the plot
plt.tight_layout()
plt.savefig('accuracy_comparison_heatmap.png', bbox_inches='tight')
# plt.show()  # Removed to prevent idling

exit()