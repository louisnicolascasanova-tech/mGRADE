import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os

def load_api_key():
    """Load wandb API key from file"""
    try:
        with open("wandb_api_key.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("wandb_api_key.txt not found")
        return None

def plot_grad_norms(entity, project, run_id, from_10k=False, ema=False, ema_coef=0.5):
    """Plot gradient norms for DCLSLayer positions and gates"""
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("No API key found. Please create wandb_api_key.txt with your key.")
        return
    
    # Login with API key
    wandb.login(key=api_key)
    
    try:
        # Initialize API
        api = wandb.Api()
        
        # Get the run
        run_path = f"{entity}/{project}/{run_id}"
        print(f"Downloading data for run: {run_path}")
        
        run = api.run(run_path)
        
        # Get run history with _step
        history = run.history(samples=None)
        print(f"Total history entries: {len(history)}")
        
        # Use _step as x-axis if available, otherwise fall back to index
        if '_step' in history.columns:
            x_axis = history['_step']
            # recover the first x_axis value that > 10k
            first_valid_index = x_axis[x_axis > 10000].index.min() if (x_axis > 10000).any() else None
            
            # If from_10k flag is set, filter data from 10k onwards
            if from_10k and first_valid_index is not None:
                history = history.iloc[first_valid_index:].copy()
                x_axis = history['_step']
                first_valid_index = 0  # Reset since we've already filtered
            
            x_label = 'Step'
        else:
            x_axis = history.index
            first_valid_index = None
            x_label = 'Index'
        
        # Helper function to apply EMA smoothing
        def apply_ema(data):
            if ema and len(data) > 0:
                return data.ewm(alpha=ema_coef).mean()
            return data
        
        # Helper function to compute stats from first_valid_index
        def compute_stats(data):
            if first_valid_index is not None and first_valid_index > 0:
                valid_data = data.iloc[first_valid_index:]
                return valid_data.max(), valid_data.mean()
            else:
                return data.max(), data.mean()
        
        # Create output directory
        output_dir = f"analysis/run_{run_id}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        train_acc = run.scan_history(keys=['train/acc'], page_size=50000)
        print([ta for ta in train_acc])

        # Check how many columns have None values
        none_counts = history.isnull().sum()
        columns_with_none = none_counts[none_counts > 0]
        if len(columns_with_none) > 0:
            print(f"Columns with None values: {len(columns_with_none)}")
            for col, count in columns_with_none.items():
                print(f"  {col}: {count} None values")
        else:
            print("No columns have None values")
        
        # Print columns with actual data
        columns_with_data = none_counts[none_counts == 0]
        if len(columns_with_data) > 0:
            print(f"\nColumns with data: {len(columns_with_data)}")
            for col in columns_with_data.index:
                print(f"  {col}")
        else:
            print("No columns have complete data")
        
        if len(history) == 0:
            print("No history data found")
            return
        
        # Create figure with 4 subplots (2x2)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define colors for each layer
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        # Plot positions grad norms (left subplot)
        positions_found = False
        for layer_idx in range(6):  # X from 0 to 5
            col_name = f'grad_norm/DCLSLayer_{layer_idx}/positions'
            if col_name in history.columns:
                data = history[col_name]
                max_val, mean_val = compute_stats(data)
                smoothed_data = apply_ema(data)
                ax1.plot(x_axis, smoothed_data, 
                        color=colors[layer_idx], 
                        label=f'Layer {layer_idx} (max:{max_val:.3e}, mean:{mean_val:.3e})', 
                        linewidth=1.5)
                positions_found = True
        
        if positions_found:
            ax1.set_title('Gradient Norm - DCLSLayer Positions', fontsize=14)
            ax1.set_xlabel(x_label)
            ax1.set_ylabel('Grad Norm')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No positions grad norm data found', 
                    transform=ax1.transAxes, ha='center', va='center')
            ax1.set_title('Gradient Norm - DCLSLayer Positions (No Data)')
        
        # Plot gates grad norms (right subplot)
        gates_found = False
        for layer_idx in range(6):  # X from 0 to 5
            col_name = f'grad_variance/DCLSLayer_{layer_idx}/positions'
            if col_name in history.columns:
                data = history[col_name]
                max_val, mean_val = compute_stats(data)
                smoothed_data = apply_ema(data)
                ax2.plot(x_axis, smoothed_data, 
                        color=colors[layer_idx], 
                        label=f'Layer {layer_idx} (max:{max_val:.3e}, mean:{mean_val:.3e})', 
                        linewidth=1.5)
                gates_found = True
        
        if gates_found:
            ax2.set_title('Gradient Variance - DCLSLayer Positions', fontsize=14)
            ax2.set_xlabel(x_label)
            ax2.set_ylabel('Grad Variance')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No positions grad variance data found', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Gradient Variance - DCLSLayer Positions (No Data)')
        
        # Plot weights grad norms (bottom left subplot)
        weights_norm_found = False
        for layer_idx in range(6):  # X from 0 to 5
            col_name = f'grad_norm/DCLSLayer_{layer_idx}/weights'
            if col_name in history.columns:
                data = history[col_name]
                max_val, mean_val = compute_stats(data)
                smoothed_data = apply_ema(data)
                ax3.plot(x_axis, smoothed_data, 
                        color=colors[layer_idx], 
                        label=f'Layer {layer_idx} (max:{max_val:.3e}, mean:{mean_val:.3e})', 
                        linewidth=1.5)
                weights_norm_found = True
        
        if weights_norm_found:
            ax3.set_title('Gradient Norm - DCLSLayer Weights', fontsize=14)
            ax3.set_xlabel(x_label)
            ax3.set_ylabel('Grad Norm')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No weights grad norm data found', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Gradient Norm - DCLSLayer Weights (No Data)')
        
        # Plot weights grad variance (bottom right subplot)
        weights_var_found = False
        for layer_idx in range(6):  # X from 0 to 5
            col_name = f'grad_variance/DCLSLayer_{layer_idx}/weights'
            if col_name in history.columns:
                data = history[col_name]
                max_val, mean_val = compute_stats(data)
                smoothed_data = apply_ema(data)
                ax4.plot(x_axis, smoothed_data, 
                        color=colors[layer_idx], 
                        label=f'Layer {layer_idx} (max:{max_val:.3e}, mean:{mean_val:.3e})', 
                        linewidth=1.5)
                weights_var_found = True
        
        if weights_var_found:
            ax4.set_title('Gradient Variance - DCLSLayer Weights', fontsize=14)
            ax4.set_xlabel(x_label)
            ax4.set_ylabel('Grad Variance')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No weights grad variance data found', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Gradient Variance - DCLSLayer Weights (No Data)')
        
        plt.tight_layout()
        
        # Save the plot
        ema_suffix = f"_ema_{ema_coef}" if ema else ""
        plot_file = f"{output_dir}/run_{run_id}_dcls{ema_suffix}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"DCLS layers gradient plot saved to: {plot_file}")
        
        # Create MLP figure with 6 rows x 2 columns (12 subplots)
        fig2, axes = plt.subplots(6, 2, figsize=(16, 20))
        
        # MLP components to plot
        mlp_components = [
            'LayerNorm_0/scale',
            'LayerNorm_0/bias', 
            'Dense_0/kernel',
            'Dense_0/bias',
            'Dense_1/kernel',
            'Dense_1/bias'
        ]
        
        component_titles = [
            'LayerNorm Scale',
            'LayerNorm Bias',
            'Dense_0 Kernel', 
            'Dense_0 Bias',
            'Dense_1 Kernel',
            'Dense_1 Bias'
        ]
        
        for row, (component, title) in enumerate(zip(mlp_components, component_titles)):
            # Plot grad norms (left column)
            norm_found = False
            for layer_idx in range(6):  # X from 0 to 5
                col_name = f'grad_norm/MLP_{layer_idx}/{component}'
                if col_name in history.columns:
                    data = history[col_name]
                    max_val, mean_val = compute_stats(data)
                    smoothed_data = apply_ema(data)
                    axes[row, 0].plot(x_axis, smoothed_data, 
                                    color=colors[layer_idx], 
                                    label=f'MLP {layer_idx} (max:{max_val:.3e}, mean:{mean_val:.3e})', 
                                    linewidth=1.5)
                    norm_found = True
            
            if norm_found:
                axes[row, 0].set_title(f'Gradient Norm - {title}', fontsize=12)
                axes[row, 0].set_xlabel(x_label)
                axes[row, 0].set_ylabel('Grad Norm')
                axes[row, 0].grid(True, alpha=0.3)
                axes[row, 0].legend()
            else:
                axes[row, 0].text(0.5, 0.5, f'No {component} grad norm data found', 
                                transform=axes[row, 0].transAxes, ha='center', va='center')
                axes[row, 0].set_title(f'Gradient Norm - {title} (No Data)')
            
            # Plot grad variance (right column)
            var_found = False
            for layer_idx in range(6):  # X from 0 to 5
                col_name = f'grad_variance/MLP_{layer_idx}/{component}'
                if col_name in history.columns:
                    data = history[col_name]
                    max_val, mean_val = compute_stats(data)
                    smoothed_data = apply_ema(data)
                    axes[row, 1].plot(x_axis, smoothed_data, 
                                    color=colors[layer_idx], 
                                    label=f'MLP {layer_idx} (max:{max_val:.3e}, mean:{mean_val:.3e})', 
                                    linewidth=1.5)
                    var_found = True
            
            if var_found:
                axes[row, 1].set_title(f'Gradient Variance - {title}', fontsize=12)
                axes[row, 1].set_xlabel(x_label)
                axes[row, 1].set_ylabel('Grad Variance')
                axes[row, 1].grid(True, alpha=0.3)
                axes[row, 1].legend()
            else:
                axes[row, 1].text(0.5, 0.5, f'No {component} grad variance data found', 
                                transform=axes[row, 1].transAxes, ha='center', va='center')
                axes[row, 1].set_title(f'Gradient Variance - {title} (No Data)')
        
        plt.tight_layout()
        
        # Save the MLP plot
        mlp_plot_file = f"{output_dir}/run_{run_id}_mlp{ema_suffix}.png"
        plt.savefig(mlp_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"MLP layers gradient plot saved to: {mlp_plot_file}")
        
        # Collect all statistics and save to JSON
        statistics = {
            "run_id": run_id,
            "dcls_layers": {},
            "mlp_layers": {}
        }
        
        # Collect DCLS layer statistics
        for layer_idx in range(6):
            layer_stats = {}
            
            # Positions grad norm
            col_name = f'grad_norm/DCLSLayer_{layer_idx}/positions'
            if col_name in history.columns:
                data = history[col_name]
                layer_stats['positions_grad_norm'] = {
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'min': float(data.min()),
                    'std': float(data.std())
                }
            
            # Positions grad variance
            col_name = f'grad_variance/DCLSLayer_{layer_idx}/positions'
            if col_name in history.columns:
                data = history[col_name]
                layer_stats['positions_grad_variance'] = {
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'min': float(data.min()),
                    'std': float(data.std())
                }
            
            # Weights grad norm
            col_name = f'grad_norm/DCLSLayer_{layer_idx}/weights'
            if col_name in history.columns:
                data = history[col_name]
                layer_stats['weights_grad_norm'] = {
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'min': float(data.min()),
                    'std': float(data.std())
                }
            
            # Weights grad variance
            col_name = f'grad_variance/DCLSLayer_{layer_idx}/weights'
            if col_name in history.columns:
                data = history[col_name]
                layer_stats['weights_grad_variance'] = {
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'min': float(data.min()),
                    'std': float(data.std())
                }
            
            if layer_stats:  # Only add if we found some data
                statistics['dcls_layers'][f'layer_{layer_idx}'] = layer_stats
        
        # Collect MLP layer statistics
        for layer_idx in range(6):
            layer_stats = {}
            
            for component in mlp_components:
                component_key = component.replace('/', '_')
                
                # Grad norm
                col_name = f'grad_norm/MLP_{layer_idx}/{component}'
                if col_name in history.columns:
                    data = history[col_name]
                    layer_stats[f'{component_key}_grad_norm'] = {
                        'max': float(data.max()),
                        'mean': float(data.mean()),
                        'min': float(data.min()),
                        'std': float(data.std())
                    }
                
                # Grad variance
                col_name = f'grad_variance/MLP_{layer_idx}/{component}'
                if col_name in history.columns:
                    data = history[col_name]
                    layer_stats[f'{component_key}_grad_variance'] = {
                        'max': float(data.max()),
                        'mean': float(data.mean()),
                        'min': float(data.min()),
                        'std': float(data.std())
                    }
            
            if layer_stats:  # Only add if we found some data
                statistics['mlp_layers'][f'layer_{layer_idx}'] = layer_stats
        
        # Save statistics to JSON
        stats_file = f"{output_dir}/run_{run_id}_gradient_statistics{ema_suffix}.json"
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"Gradient statistics saved to: {stats_file}")
        
        # Create HeinsenMinGeneralGRULayer figure
        fig3, (ax_gru1, ax_gru2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # HeinsenMinGeneralGRULayer components to plot
        gru_components = ['Dense_x/kernel', 'Dense_x/bias']
        gru_titles = ['Dense_x Kernel', 'Dense_x Bias']
        gru_axes = [ax_gru1, ax_gru2]
        
        for idx, (component, title, ax) in enumerate(zip(gru_components, gru_titles, gru_axes)):
            
            # Plot grad norms for this component
            norm_found = False
            for layer_idx in range(6):  # X from 0 to 5
                col_name = f'grad_norm/HeinsenMinGeneralGRULayer_{layer_idx}/{component}'
                if col_name in history.columns:
                    data = history[col_name]
                    max_val, mean_val = compute_stats(data)
                    smoothed_data = apply_ema(data)
                    ax.plot(x_axis, smoothed_data, 
                           color=colors[layer_idx], 
                           label=f'Layer {layer_idx} (max:{max_val:.3e}, mean:{mean_val:.3e})', 
                           linewidth=1.5)
                    norm_found = True
            
            if norm_found:
                ax.set_title(f'Gradient Norm - HeinsenMinGeneralGRU {title}', fontsize=14)
                ax.set_xlabel(x_label)
                ax.set_ylabel('Grad Norm')
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'No {component} grad norm data found', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'Gradient Norm - HeinsenMinGeneralGRU {title} (No Data)')
        
        plt.tight_layout()
        
        # Save the HeinsenMinGeneralGRU plot
        gru_plot_file = f"{output_dir}/run_{run_id}_heinsen_gru{ema_suffix}.png"
        plt.savefig(gru_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"HeinsenMinGeneralGRU layers gradient plot saved to: {gru_plot_file}")
        
        # Add HeinsenMinGeneralGRU statistics to the statistics dict
        statistics['heinsen_gru_layers'] = {}
        
        for layer_idx in range(6):
            layer_stats = {}
            
            for component in gru_components:
                component_key = component.replace('/', '_').replace('_x', '_x')
                
                # Grad norm
                col_name = f'grad_norm/HeinsenMinGeneralGRULayer_{layer_idx}/{component}'
                if col_name in history.columns:
                    data = history[col_name]
                    layer_stats[f'{component_key}_grad_norm'] = {
                        'max': float(data.max()),
                        'mean': float(data.mean()),
                        'min': float(data.min()),
                        'std': float(data.std())
                    }
                
                # Grad variance
                col_name = f'grad_variance/HeinsenMinGeneralGRULayer_{layer_idx}/{component}'
                if col_name in history.columns:
                    data = history[col_name]
                    layer_stats[f'{component_key}_grad_variance'] = {
                        'max': float(data.max()),
                        'mean': float(data.mean()),
                        'min': float(data.min()),
                        'std': float(data.std())
                    }
            
            if layer_stats:  # Only add if we found some data
                statistics['heinsen_gru_layers'][f'layer_{layer_idx}'] = layer_stats
        
        # Re-save statistics to JSON with HeinsenMinGeneralGRU data
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"Updated gradient statistics with HeinsenMinGeneralGRU data: {stats_file}")
        
        # Create Encoder/Dense_Out figure with 4 rows x 2 columns (8 subplots)
        fig4, axes = plt.subplots(4, 2, figsize=(16, 16))
        
        # Encoder/Dense_Out components to plot
        encoder_out_components = ['Encoder/kernel', 'Encoder/bias', 'Dense_Out/kernel', 'Dense_Out/bias']
        encoder_out_titles = ['Encoder Kernel', 'Encoder Bias', 'Dense_Out Kernel', 'Dense_Out Bias']
        
        for row, (component, title) in enumerate(zip(encoder_out_components, encoder_out_titles)):
            # Plot grad norms (left column)
            norm_found = False
            
            # For Encoder and Dense_Out, we don't have layer indices, just the component itself
            col_name = f'grad_norm/{component}'
            if col_name in history.columns:
                data = history[col_name]
                max_val, mean_val = compute_stats(data)
                axes[row, 0].plot(x_axis, data, 
                                color='blue', 
                                label=f'(max:{max_val:.3e}, mean:{mean_val:.3e})', 
                                linewidth=1.5)
                norm_found = True
            
            if norm_found:
                axes[row, 0].set_title(f'Gradient Norm - {title}', fontsize=12)
                axes[row, 0].set_xlabel(x_label)
                axes[row, 0].set_ylabel('Grad Norm')
                axes[row, 0].grid(True, alpha=0.3)
                axes[row, 0].legend()
            else:
                axes[row, 0].text(0.5, 0.5, f'No {component} grad norm data found', 
                                transform=axes[row, 0].transAxes, ha='center', va='center')
                axes[row, 0].set_title(f'Gradient Norm - {title} (No Data)')
            
            # Plot grad variance (right column)
            var_found = False
            col_name = f'grad_variance/{component}'
            if col_name in history.columns:
                data = history[col_name]
                max_val, mean_val = compute_stats(data)
                axes[row, 1].plot(x_axis, data, 
                                color='blue', 
                                label=f'(max:{max_val:.3e}, mean:{mean_val:.3e})', 
                                linewidth=1.5)
                var_found = True
            
            if var_found:
                axes[row, 1].set_title(f'Gradient Variance - {title}', fontsize=12)
                axes[row, 1].set_xlabel(x_label)
                axes[row, 1].set_ylabel('Grad Variance')
                axes[row, 1].grid(True, alpha=0.3)
                axes[row, 1].legend()
            else:
                axes[row, 1].text(0.5, 0.5, f'No {component} grad variance data found', 
                                transform=axes[row, 1].transAxes, ha='center', va='center')
                axes[row, 1].set_title(f'Gradient Variance - {title} (No Data)')
        
        plt.tight_layout()
        
        # Save the Encoder/Dense_Out plot
        encoder_out_plot_file = f"{output_dir}/run_{run_id}_encoder_dense_out{ema_suffix}.png"
        plt.savefig(encoder_out_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Encoder/Dense_Out gradient plot saved to: {encoder_out_plot_file}")
        
        # Add Encoder/Dense_Out statistics to the statistics dict
        statistics['encoder_dense_out'] = {}
        
        for component in encoder_out_components:
            component_key = component.replace('/', '_')
            component_stats = {}
            
            # Grad norm
            col_name = f'grad_norm/{component}'
            if col_name in history.columns:
                data = history[col_name]
                component_stats['grad_norm'] = {
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'min': float(data.min()),
                    'std': float(data.std())
                }
            
            # Grad variance
            col_name = f'grad_variance/{component}'
            if col_name in history.columns:
                data = history[col_name]
                component_stats['grad_variance'] = {
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'min': float(data.min()),
                    'std': float(data.std())
                }
            
            if component_stats:  # Only add if we found some data
                statistics['encoder_dense_out'][component_key] = component_stats
        
        # Re-save statistics to JSON with Encoder/Dense_Out data
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"Updated gradient statistics with Encoder/Dense_Out data: {stats_file}")
                
        # Print available grad norm columns for reference
        grad_norm_cols = [col for col in history.columns if col.startswith('grad_norm/')]
        if grad_norm_cols:
            print(f"\nAvailable grad norm columns:")
            for col in sorted(grad_norm_cols):
                print(f"  - {col}")
        
        # Print available grad variance columns for reference
        grad_var_cols = [col for col in history.columns if col.startswith('grad_variance/')]
        if grad_var_cols:
            print(f"\nAvailable grad variance columns:")
            for col in sorted(grad_var_cols):
                print(f"  - {col}")
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot gradient norms for DCLSLayer positions and gates')
    parser.add_argument('entity', nargs='?', default='torchet-tristan', help='WandB entity')
    parser.add_argument('project', nargs='?', default='DenGRU_general', help='WandB project')
    parser.add_argument('run_id', nargs='?', default='76gu8tai', help='WandB run ID')
    parser.add_argument('--from_10k', action='store_true', help='Start plots from step 10k instead of 0')
    parser.add_argument('--ema', action='store_true', help='Apply exponential moving average smoothing to all time series')
    parser.add_argument('--ema_coef', type=float, default=0.5, help='EMA smoothing coefficient (default: 0.5)')
    
    args = parser.parse_args()
    
    plot_grad_norms(args.entity, args.project, args.run_id, args.from_10k, args.ema, args.ema_coef)