#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from executables.sc import SpeechCommands
from pathlib import Path

def test_speechcommands_simple():
    """Simple test of SpeechCommands dataset loading without JAX dependencies"""
    
    print("Testing SpeechCommands dataset loading (simple version)...")
    
    try:
        # Test the core dataset class directly
        print("Creating SpeechCommands dataset...")
        
        dataset_obj = SpeechCommands('sc', data_dir='./data/speech_commands_v2/')
        dataset_obj.all_classes = True
        dataset_obj.sr = 1  # subsampling rate
        
        print("Setting up dataset...")
        dataset_obj.setup()
        
        print(f"Dataset setup complete!")
        print(f"d_output (num classes): {dataset_obj.d_output}")
        
        # Check if datasets were created
        print(f"Train dataset size: {len(dataset_obj.dataset_train)}")
        print(f"Val dataset size: {len(dataset_obj.dataset_val)}")
        print(f"Test dataset size: {len(dataset_obj.dataset_test)}")
        
        # Check tensor shapes
        train_sample = dataset_obj.dataset_train.tensors[0]
        train_labels = dataset_obj.dataset_train.tensors[1]
        
        print(f"Train data shape: {train_sample.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Sample rate used: {dataset_obj.sr}")
        
        # Test a simple dataloader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(dataset_obj.dataset_train, batch_size=32, shuffle=True)
        
        print(f"\nTesting dataloader...")
        print(f"Number of batches: {len(train_loader)}")
        
        # Get one batch
        for batch_idx, (data, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Data shape: {data.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Data dtype: {data.dtype}")
            print(f"  Targets dtype: {targets.dtype}")
            print(f"  Sample labels: {targets[:5]}")
            print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
            break
            
        print("\n✓ Dataset loading successful!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_speechcommands_simple()