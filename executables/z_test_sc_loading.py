#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import create_speechcommands35_classification_dataset, prep_batch
import torch
import numpy as np

def test_speechcommands_loading():
    """Test loading SpeechCommands dataset and print batch shapes"""
    
    print("Testing SpeechCommands dataset loading...")
    
    # Create dataloaders
    try:
        trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE = create_speechcommands35_classification_dataset(
            bsz=32,
            seed=42
        )
        
        print(f"\nDataset Info:")
        print(f"Number of classes: {N_CLASSES}")
        print(f"Sequence length: {SEQ_LENGTH}")
        print(f"Input dimension: {IN_DIM}")
        print(f"Training size: {TRAIN_SIZE}")
        
        print(f"\nDataLoader sizes:")
        print(f"Training batches: {len(trn_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(tst_loader)}")
        
        # Test a batch from training loader
        print(f"\n=== Training Batch ===")
        for batch_idx, batch in enumerate(trn_loader):
            print(f"Raw batch type: {type(batch)}")
            print(f"Raw batch length: {len(batch)}")
            
            if len(batch) == 2:
                inputs, targets = batch
                aux_data = {}
            elif len(batch) == 3:
                inputs, targets, aux_data = batch
            else:
                print(f"Unexpected batch format with {len(batch)} elements")
                break
                
            print(f"Raw inputs shape: {inputs.shape}")
            print(f"Raw targets shape: {targets.shape}")
            print(f"Raw inputs dtype: {inputs.dtype}")
            print(f"Raw targets dtype: {targets.dtype}")
            if aux_data:
                print(f"Aux data keys: {list(aux_data.keys())}")
                for key, val in aux_data.items():
                    if hasattr(val, 'shape'):
                        print(f"  {key} shape: {val.shape}")
            
            # Test prep_batch function (used in main.py for LRA datasets)
            print(f"\n--- After prep_batch ---")
            prepared_inputs, prepared_targets = prep_batch(batch, SEQ_LENGTH, IN_DIM)
            
            if isinstance(prepared_inputs, tuple):
                print(f"Prepared inputs type: tuple (inputs, lengths)")
                print(f"  inputs shape: {prepared_inputs[0].shape}")
                print(f"  lengths shape: {prepared_inputs[1].shape}")
            else:
                print(f"Prepared inputs shape: {prepared_inputs.shape}")
            print(f"Prepared targets shape: {prepared_targets.shape}")
            print(f"Prepared inputs dtype: {prepared_inputs[0].dtype if isinstance(prepared_inputs, tuple) else prepared_inputs.dtype}")
            print(f"Prepared targets dtype: {prepared_targets.dtype}")
            
            break
            
        # Test auxiliary loaders if they exist
        if aux_loaders:
            print(f"\n=== Auxiliary Loaders ===")
            for aux_name, aux_loader in aux_loaders.items():
                print(f"{aux_name}: {len(aux_loader)} batches")
                for batch in aux_loader:
                    if len(batch) >= 2:
                        inputs, targets = batch[:2]
                        print(f"  {aux_name} inputs shape: {inputs.shape}")
                        print(f"  {aux_name} targets shape: {targets.shape}")
                    break
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_speechcommands_loading()