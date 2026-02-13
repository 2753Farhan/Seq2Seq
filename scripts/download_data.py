#!/usr/bin/env python

from datasets import load_dataset
import os
import pickle

def main():
    print("Downloading CodeSearchNet Python dataset...")
    
    os.makedirs('data', exist_ok=True)
    
    # Updated dataset name
    try:
        # Try the correct dataset name
        dataset = load_dataset("code_search_net", "python", trust_remote_code=False)
    except:
        try:
            # Alternative dataset name
            dataset = load_dataset("code-search-net/code-search-net", "python")
        except:
            # Fallback to a smaller dataset for testing
            print("CodeSearchNet not available, using a smaller dataset for testing...")
            from datasets import load_dataset
            dataset = load_dataset("neulab/conala", split="train")
            # Split into train/valid/test
            dataset = dataset.train_test_split(test_size=0.1, seed=42)
            dataset['validation'] = dataset['test']
    
    print(f"Dataset splits: {dataset.keys()}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")
    
    # Save a sample
    sample = {}
    for split in dataset.keys():
        if len(dataset[split]) > 0:
            sample[split] = dataset[split][0]
    
    with open('data/sample.pkl', 'wb') as f:
        pickle.dump(sample, f)
    
    print(f"Sample saved to data/sample.pkl")
    print("Done!")

if __name__ == "__main__":
    main()