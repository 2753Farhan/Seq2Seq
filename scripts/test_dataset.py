#!/usr/bin/env python
"""Test script to verify dataset loading"""

import sys
sys.path.append('.')

from src.data.dataset import create_dataloaders
import yaml

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test dataset loading
    print("Testing dataset loading...")
    train_loader, valid_loader, test_loader, src_tokenizer, tgt_tokenizer = create_dataloaders(config)
    
    print("\n✅ Dataset loaded successfully!")
    print(f"Source vocab size: {len(src_tokenizer)}")
    print(f"Target vocab size: {len(tgt_tokenizer)}")
    
    # Test a batch
    for src, tgt in train_loader:
        print(f"\nBatch shapes - Source: {src.shape}, Target: {tgt.shape}")
        print(f"Source sample: {src_tokenizer.decode(src[0])}")
        print(f"Target sample: {tgt_tokenizer.decode(tgt[0])}")
        break

if __name__ == "__main__":
    main()