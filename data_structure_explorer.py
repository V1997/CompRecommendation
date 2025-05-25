"""
Data structure explorer for the real estate appraisal dataset.

This module first examines the actual data structure to understand
the format before implementing proper extraction logic.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def examine_actual_structure(dataset, num_samples=3):
    """
    Examine the actual structure of the dataset records.
    
    Parameters
    ----------
    dataset : list
        List of appraisal records
    num_samples : int, default=3
        Number of sample records to examine
        
    Returns
    -------
    dict
        Detailed structure analysis
    """
    print(f"=== Examining Structure of {len(dataset)} Records ===")
    
    structure_info = {}
    
    for i in range(min(num_samples, len(dataset))):
        record = dataset[i]
        print(f"\n--- Record {i} Structure ---")
        
        if isinstance(record, dict):
            for key, value in record.items():
                print(f"\nField '{key}':")
                print(f"  Type: {type(value).__name__}")
                
                if isinstance(value, dict):
                    print(f"  Keys: {list(value.keys())}")
                    # Show sample of first few key-value pairs
                    sample_items = dict(list(value.items())[:3])
                    print(f"  Sample data: {sample_items}")
                    
                elif isinstance(value, list):
                    print(f"  Length: {len(value)}")
                    if value:
                        print(f"  First element type: {type(value[0]).__name__}")
                        if isinstance(value[0], dict):
                            print(f"  First element keys: {list(value[0].keys())}")
                            print(f"  First element sample: {dict(list(value[0].items())[:3])}")
                        else:
                            print(f"  First element: {value[0]}")
                            
                else:
                    print(f"  Value: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                    
        if i == 0:  # Store detailed info for first record
            structure_info = record
            
    return structure_info


def load_and_examine_dataset(file_path="appraisals_dataset.json"):
    """
    Load dataset and examine its structure.
    
    Parameters
    ----------
    file_path : str
        Path to the dataset file
        
    Returns
    -------
    tuple
        (dataset, structure_info)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Handle the nested structure
        if isinstance(raw_data, dict) and 'appraisals' in raw_data:
            dataset = raw_data['appraisals']
        elif isinstance(raw_data, list):
            dataset = raw_data
        else:
            raise ValueError("Unexpected dataset structure")
            
        print(f"Loaded {len(dataset)} appraisal records")
        
        # Examine structure
        structure_info = examine_actual_structure(dataset)
        
        return dataset, structure_info
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None


if __name__ == "__main__":
    dataset, structure = load_and_examine_dataset()
    
    if dataset:
        # Save structure analysis
        with open('dataset_structure_analysis.json', 'w') as f:
            json.dump(structure, f, indent=2, default=str)
        print(f"\nStructure analysis saved to 'dataset_structure_analysis.json'")