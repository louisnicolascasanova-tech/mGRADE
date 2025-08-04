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

def plot_training_dyn(entity, project, run_id):
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
        
        # Get run history
        history = run.scan_history(keys=[
            'train/acc',             'train/loss', 
            'val/acc',               'val/loss', 
            'val/f1_class_0',        'val/f1_class_1', 
            'val/precision_class_0', 'val/precision_class_1', 
            'val/recall_class_0',    'val/recall_class_1', 
            'train/learning_rate'])
        
        # Collect all metrics in a single pass
        metrics = {
            'train_acc': [],
            'train_loss': [],
            'val_acc': [],
            'val_loss': [],
            'val_f1_class_0': [],
            'val_f1_class_1': [],
            'val_precision_class_0': [],
            'val_precision_class_1': [],
            'val_recall_class_0': [],
            'val_recall_class_1': [],
            'train_learning_rate': []
        }
        
        for row in history:
            if "train/acc" in row:
                metrics['train_acc'].append(row["train/acc"])
            if "train/loss" in row:
                metrics['train_loss'].append(row["train/loss"])
            if "val/acc" in row:
                metrics['val_acc'].append(row["val/acc"])
            if "val/loss" in row:
                metrics['val_loss'].append(row["val/loss"])
            if "val/f1_class_0" in row:
                metrics['val_f1_class_0'].append(row["val/f1_class_0"])
            if "val/f1_class_1" in row:
                metrics['val_f1_class_1'].append(row["val/f1_class_1"])
            if "val/precision_class_0" in row:
                metrics['val_precision_class_0'].append(row["val/precision_class_0"])
            if "val/precision_class_1" in row:
                metrics['val_precision_class_1'].append(row["val/precision_class_1"])
            if "val/recall_class_0" in row:
                metrics['val_recall_class_0'].append(row["val/recall_class_0"])
            if "val/recall_class_1" in row:
                metrics['val_recall_class_1'].append(row["val/recall_class_1"])
            if "train/learning_rate" in row:
                metrics['train_learning_rate'].append(row["train/learning_rate"])
        
        # Extract lists for plotting
        train_acc_list = metrics['train_acc']
        train_loss_list = metrics['train_loss']
        val_acc_list = metrics['val_acc']
        val_loss_list = metrics['val_loss']
        val_f1_class_0_list = metrics['val_f1_class_0']
        val_f1_class_1_list = metrics['val_f1_class_1']
        val_precision_class_0_list = metrics['val_precision_class_0']
        val_precision_class_1_list = metrics['val_precision_class_1']
        val_recall_class_0_list = metrics['val_recall_class_0']
        val_recall_class_1_list = metrics['val_recall_class_1']
        train_learning_rate_list = metrics['train_learning_rate']

        # Create output directory
        output_dir = f"analysis/run_{run_id}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Find highest validation accuracy and its epoch
        if val_acc_list:
            max_val_acc = max(val_acc_list)
            max_val_acc_epoch = val_acc_list.index(max_val_acc)
            
            # Get values at max val acc epoch for all metrics
            train_acc_at_max = train_acc_list[max_val_acc_epoch] if max_val_acc_epoch < len(train_acc_list) else None
            train_loss_at_max = train_loss_list[max_val_acc_epoch] if max_val_acc_epoch < len(train_loss_list) else None
            val_loss_at_max = val_loss_list[max_val_acc_epoch] if max_val_acc_epoch < len(val_loss_list) else None
            val_f1_class_0_at_max = val_f1_class_0_list[max_val_acc_epoch] if max_val_acc_epoch < len(val_f1_class_0_list) else None
            val_f1_class_1_at_max = val_f1_class_1_list[max_val_acc_epoch] if max_val_acc_epoch < len(val_f1_class_1_list) else None
            val_precision_class_0_at_max = val_precision_class_0_list[max_val_acc_epoch] if max_val_acc_epoch < len(val_precision_class_0_list) else None
            val_precision_class_1_at_max = val_precision_class_1_list[max_val_acc_epoch] if max_val_acc_epoch < len(val_precision_class_1_list) else None
            val_recall_class_0_at_max = val_recall_class_0_list[max_val_acc_epoch] if max_val_acc_epoch < len(val_recall_class_0_list) else None
            val_recall_class_1_at_max = val_recall_class_1_list[max_val_acc_epoch] if max_val_acc_epoch < len(val_recall_class_1_list) else None
            train_lr_at_max = train_learning_rate_list[max_val_acc_epoch] if max_val_acc_epoch < len(train_learning_rate_list) else None
        else:
            max_val_acc = None
            max_val_acc_epoch = None

        PX = 1/plt.rcParams['figure.dpi']  # pixel per inch
        # plot a 2 by 5 subplot 
        plt.figure(figsize=(4000*PX, 800*PX))  # Adjust size for better visibility
        # plot 1: train vs val acc
        plt.subplot(2, 5, 1)
        plt.plot(train_acc_list, label='Train Loss', color='blue')
        plt.plot(val_acc_list, label='Validation Accuracy', color='orange')
        
        # Add star and text for max validation accuracy
        if max_val_acc is not None:
            plt.plot(max_val_acc_epoch, max_val_acc, 'r*', markersize=10)
            plt.text(max_val_acc_epoch, max_val_acc + 0.01, f'{max_val_acc:.3f}', 
                    ha='center', va='top', fontsize=8, color='red')
            # Add star and text for train accuracy at max val epoch
            if train_acc_at_max is not None:
                plt.plot(max_val_acc_epoch, train_acc_at_max, 'b*', markersize=10)
                plt.text(max_val_acc_epoch, train_acc_at_max - 0.02, f'{train_acc_at_max:.3f}', 
                        ha='center', va='bottom', fontsize=8, color='blue')
        
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        # plt.title('Train Acc Over Epochs')
        plt.legend()
        plt.grid()

        # Add to plo1 1 a second y-axis for train learning rate
        ax2 = plt.gca().twinx()
        ax2.plot(train_learning_rate_list, label='Train Learning Rate', color='green', linestyle='--')
        ax2.set_ylabel('Learning Rate')
        ax2.legend(loc='upper right')
        plt.title('Train Acc and Learning Rate Over Epochs')


        # plot 2: train vs val loss
        plt.subplot(2, 5, 2)
        plt.plot(train_loss_list, label='Train Loss', color='blue')
        plt.plot(val_loss_list, label='Validation Loss', color='orange')
        
        # Add star and text for losses at max val acc epoch
        if max_val_acc_epoch is not None:
            if train_loss_at_max is not None:
                plt.plot(max_val_acc_epoch, train_loss_at_max, 'b*', markersize=10)
                plt.text(max_val_acc_epoch, train_loss_at_max + 0.01, f'{train_loss_at_max:.3f}', 
                        ha='center', va='bottom', fontsize=8, color='blue')
            if val_loss_at_max is not None:
                plt.plot(max_val_acc_epoch, val_loss_at_max, 'r*', markersize=10)
                plt.text(max_val_acc_epoch, val_loss_at_max - 0.01, f'{val_loss_at_max:.3f}', 
                        ha='center', va='top', fontsize=8, color='orange')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train Loss Over Epochs')
        plt.legend()
        plt.grid()

        # Add to plo1 1 a second y-axis for train learning rate
        ax2 = plt.gca().twinx()
        ax2.plot(train_learning_rate_list, label='Train Learning Rate', color='green', linestyle='--')
        ax2.set_ylabel('Learning Rate')
        ax2.legend(loc='upper right')
        plt.title('Train Acc and Learning Rate Over Epochs')


        # plot 3: val f1 class 0 and 1
        plt.subplot(2, 5, 3)
        plt.plot(val_f1_class_0_list, label='Validation F1 Class 0', color='green')
        plt.plot(val_f1_class_1_list, label='Validation F1 Class 1', color='red')
        
        # Add star and text for F1 scores at max val acc epoch
        if max_val_acc_epoch is not None:
            if val_f1_class_0_at_max is not None:
                plt.plot(max_val_acc_epoch, val_f1_class_0_at_max, 'g*', markersize=10)
                plt.text(max_val_acc_epoch, val_f1_class_0_at_max + 0.005, f'{val_f1_class_0_at_max:.3f}', 
                        ha='center', va='bottom', fontsize=8, color='green')
            if val_f1_class_1_at_max is not None:
                plt.plot(max_val_acc_epoch, val_f1_class_1_at_max, 'r*', markersize=10)
                plt.text(max_val_acc_epoch, val_f1_class_1_at_max - 0.005, f'{val_f1_class_1_at_max:.3f}', 
                        ha='center', va='top', fontsize=8, color='red')
        
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Scores Over Epochs')
        plt.legend()
        plt.grid()

        # plot 4: val precision class 0 and 1
        plt.subplot(2, 5, 4)
        plt.plot(val_precision_class_0_list, label='Validation Precision Class 0', color='green')
        plt.plot(val_precision_class_1_list, label='Validation Precision Class 1', color='red')
        
        # Add star and text for precision at max val acc epoch
        if max_val_acc_epoch is not None:
            if val_precision_class_0_at_max is not None:
                plt.plot(max_val_acc_epoch, val_precision_class_0_at_max, 'g*', markersize=10)
                plt.text(max_val_acc_epoch, val_precision_class_0_at_max + 0.01, f'{val_precision_class_0_at_max:.3f}', 
                        ha='center', va='bottom', fontsize=8, color='green')
            if val_precision_class_1_at_max is not None:
                plt.plot(max_val_acc_epoch, val_precision_class_1_at_max, 'r*', markersize=10)
                plt.text(max_val_acc_epoch, val_precision_class_1_at_max - 0.01, f'{val_precision_class_1_at_max:.3f}', 
                        ha='center', va='top', fontsize=8, color='red')
        
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.title('Validation Precision Over Epochs')
        plt.legend()
        plt.grid()

        # plot 5: val recall class 0 and 1
        plt.subplot(2, 5, 5)
        plt.plot(val_recall_class_0_list, label='Validation Recall Class 0', color='green')
        plt.plot(val_recall_class_1_list, label='Validation Recall Class 1', color='red')
        
        # Add star and text for recall at max val acc epoch
        if max_val_acc_epoch is not None:
            if val_recall_class_0_at_max is not None:
                plt.plot(max_val_acc_epoch, val_recall_class_0_at_max, 'g*', markersize=10)
                plt.text(max_val_acc_epoch, val_recall_class_0_at_max + 0.01, f'{val_recall_class_0_at_max:.3f}', 
                        ha='center', va='bottom', fontsize=8, color='green')
            if val_recall_class_1_at_max is not None:
                plt.plot(max_val_acc_epoch, val_recall_class_1_at_max, 'r*', markersize=10)
                plt.text(max_val_acc_epoch, val_recall_class_1_at_max - 0.01, f'{val_recall_class_1_at_max:.3f}', 
                        ha='center', va='top', fontsize=8, color='red')
        
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.title('Validation Recall Over Epochs')
        plt.legend()
        plt.grid()

        # Second row - zoomed versions
        # plot 6: train vs val acc (zoomed)
        plt.subplot(2, 5, 6)
        plt.plot(train_acc_list, label='Train Loss', color='blue')
        plt.plot(val_acc_list, label='Validation Accuracy', color='orange')
        
        # Add star and text for max validation accuracy
        if max_val_acc is not None:
            plt.plot(max_val_acc_epoch, max_val_acc, 'r*', markersize=10)
            plt.text(max_val_acc_epoch, max_val_acc + 0.005, f'{max_val_acc:.3f}', 
                    ha='center', va='bottom', fontsize=8, color='red')
            # Add star and text for train accuracy at max val epoch
            if train_acc_at_max is not None:
                plt.plot(max_val_acc_epoch, train_acc_at_max, 'b*', markersize=10)
                plt.text(max_val_acc_epoch, train_acc_at_max - 0.01, f'{train_acc_at_max:.3f}', 
                        ha='center', va='top', fontsize=8, color='blue')
        
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.ylim([0.8, 1])
        plt.legend()
        plt.grid()

        # Add second y-axis for train learning rate
        ax2 = plt.gca().twinx()
        ax2.plot(train_learning_rate_list, label='Train Learning Rate', color='green', linestyle='--')
        ax2.set_ylabel('Learning Rate')
        ax2.legend(loc='upper right')
        plt.title('Train Acc and Learning Rate (Zoomed)')

        # plot 7: train vs val loss (unchanged)
        plt.subplot(2, 5, 7)
        plt.plot(train_loss_list, label='Train Loss', color='blue')
        plt.plot(val_loss_list, label='Validation Loss', color='orange')
        
        # Add star and text for losses at max val acc epoch
        if max_val_acc_epoch is not None:
            if train_loss_at_max is not None:
                plt.plot(max_val_acc_epoch, train_loss_at_max, 'b*', markersize=10)
                plt.text(max_val_acc_epoch, train_loss_at_max + 0.01, f'{train_loss_at_max:.3f}', 
                        ha='center', va='bottom', fontsize=8, color='blue')
            if val_loss_at_max is not None:
                plt.plot(max_val_acc_epoch, val_loss_at_max, 'r*', markersize=10)
                plt.text(max_val_acc_epoch, val_loss_at_max - 0.01, f'{val_loss_at_max:.3f}', 
                        ha='center', va='top', fontsize=8, color='orange')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train Loss Over Epochs')
        plt.legend()
        plt.grid()

        # Add second y-axis for train learning rate
        ax2 = plt.gca().twinx()
        ax2.plot(train_learning_rate_list, label='Train Learning Rate', color='green', linestyle='--')
        ax2.set_ylabel('Learning Rate')
        ax2.legend(loc='upper right')

        # plot 8: val f1 class 0 and 1 (zoomed)
        plt.subplot(2, 5, 8)
        plt.plot(val_f1_class_0_list, label='Validation F1 Class 0', color='green')
        plt.plot(val_f1_class_1_list, label='Validation F1 Class 1', color='red')
        
        # Add star and text for F1 scores at max val acc epoch
        if max_val_acc_epoch is not None:
            if val_f1_class_0_at_max is not None:
                plt.plot(max_val_acc_epoch, val_f1_class_0_at_max, 'g*', markersize=10)
                plt.text(max_val_acc_epoch, val_f1_class_0_at_max + 0.002, f'{val_f1_class_0_at_max:.3f}', 
                        ha='center', va='bottom', fontsize=8, color='green')
            if val_f1_class_1_at_max is not None:
                plt.plot(max_val_acc_epoch, val_f1_class_1_at_max, 'r*', markersize=10)
                plt.text(max_val_acc_epoch, val_f1_class_1_at_max - 0.002, f'{val_f1_class_1_at_max:.3f}', 
                        ha='center', va='top', fontsize=8, color='red')
        
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.ylim([0.8, 0.9])
        plt.title('Validation F1 Scores (Zoomed)')
        plt.legend()
        plt.grid()

        # plot 9: val precision class 0 and 1 (zoomed)
        plt.subplot(2, 5, 9)
        plt.plot(val_precision_class_0_list, label='Validation Precision Class 0', color='green')
        plt.plot(val_precision_class_1_list, label='Validation Precision Class 1', color='red')
        
        # Add star and text for precision at max val acc epoch
        if max_val_acc_epoch is not None:
            if val_precision_class_0_at_max is not None:
                plt.plot(max_val_acc_epoch, val_precision_class_0_at_max, 'g*', markersize=10)
                plt.text(max_val_acc_epoch, val_precision_class_0_at_max + 0.005, f'{val_precision_class_0_at_max:.3f}', 
                        ha='center', va='bottom', fontsize=8, color='green')
            if val_precision_class_1_at_max is not None:
                plt.plot(max_val_acc_epoch, val_precision_class_1_at_max, 'r*', markersize=10)
                plt.text(max_val_acc_epoch, val_precision_class_1_at_max - 0.005, f'{val_precision_class_1_at_max:.3f}', 
                        ha='center', va='top', fontsize=8, color='red')
        
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.ylim([0.75, 0.95])
        plt.title('Validation Precision (Zoomed)')
        plt.legend()
        plt.grid()

        # plot 10: val recall class 0 and 1 (zoomed)
        plt.subplot(2, 5, 10)
        plt.plot(val_recall_class_0_list, label='Validation Recall Class 0', color='green')
        plt.plot(val_recall_class_1_list, label='Validation Recall Class 1', color='red')
        
        # Add star and text for recall at max val acc epoch
        if max_val_acc_epoch is not None:
            if val_recall_class_0_at_max is not None:
                plt.plot(max_val_acc_epoch, val_recall_class_0_at_max, 'g*', markersize=10)
                plt.text(max_val_acc_epoch, val_recall_class_0_at_max + 0.005, f'{val_recall_class_0_at_max:.3f}', 
                        ha='center', va='bottom', fontsize=8, color='green')
            if val_recall_class_1_at_max is not None:
                plt.plot(max_val_acc_epoch, val_recall_class_1_at_max, 'r*', markersize=10)
                plt.text(max_val_acc_epoch, val_recall_class_1_at_max - 0.005, f'{val_recall_class_1_at_max:.3f}', 
                        ha='center', va='top', fontsize=8, color='red')
        
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.ylim([0.75, 0.95])
        plt.title('Validation Recall (Zoomed)')
        plt.legend()
        plt.grid()

        plt.tight_layout()

        training_plot_file = f"{output_dir}/run_{run_id}_training_metrics.png"
        plt.savefig(training_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to CSV
        import pandas as pd
        
        # Find the maximum length to pad shorter lists
        max_len = max(len(lst) for lst in metrics.values() if lst)
        
        # Pad all lists to the same length with None and round to 3 decimal places
        padded_metrics = {}
        for key, values in metrics.items():
            # Round values to 3 decimal places, keeping None for missing values
            rounded_values = [round(val, 3) if val is not None else None for val in values]
            padded_values = rounded_values + [None] * (max_len - len(rounded_values))
            padded_metrics[key] = padded_values
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(padded_metrics)
        csv_file = f"{output_dir}/run_{run_id}_metrics.csv"
        df.to_csv(csv_file, index=True, index_label='epoch')
        print(f"Metrics saved to {csv_file}")
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 3:
        entity = sys.argv[1]
        project = sys.argv[2]
        run_id = sys.argv[3]
    else:
        # Default values
        entity = "torchet-tristan"
        project = "DenGRU_general"
        run_id = "76gu8tai" # v1epmjda
    
    plot_training_dyn(entity, project, run_id)