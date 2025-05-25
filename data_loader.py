"""
Data loader for the real estate appraisal dataset.

This module handles loading and initial exploration of the
Git LFS stored appraisal dataset.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_appraisal_dataset(file_path="appraisals_dataset.json"):
    """
    Load the appraisal dataset from JSON file.
    
    Parameters
    ----------
    file_path : str, default="appraisals_dataset.json"
        Path to the dataset file
        
    Returns
    -------
    dict
        Loaded dataset dictionary
        
    Raises
    ------
    FileNotFoundError
        If the dataset file is not found
    ValueError
        If the JSON file is corrupted or invalid
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}\n"
                "Make sure you have Git LFS installed and run:\n"
                "git lfs install\n"
                "git lfs pull"
            )
            
        # Check file size to ensure it's properly downloaded
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb < 20:  # Expected size is ~22MB
            print(f"Warning: File size is {file_size_mb:.1f}MB, expected ~22MB")
            print("This might indicate the LFS file wasn't properly downloaded")
            
        print(f"Loading dataset from {file_path} ({file_size_mb:.1f}MB)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        print(f"Successfully loaded dataset with {len(dataset)} records")
        return dataset
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")


def explore_dataset_structure(dataset):
    """
    Explore and analyze the structure of the appraisal dataset.
    
    Parameters
    ----------
    dataset : dict or list
        The loaded dataset
        
    Returns
    -------
    dict
        Dictionary containing dataset analysis
    """
    analysis = {}
    
    # Basic structure analysis
    analysis['total_records'] = len(dataset)
    analysis['data_type'] = type(dataset).__name__
    
    if isinstance(dataset, list) and len(dataset) > 0:
        # Analyze first record structure
        first_record = dataset[0]
        analysis['record_structure'] = {
            'keys': list(first_record.keys()) if isinstance(first_record, dict) else 'Not a dictionary',
            'sample_record': first_record if len(str(first_record)) < 500 else str(first_record)[:500] + "..."
        }
        
        # Check for common fields across records
        if isinstance(first_record, dict):
            all_keys = set()
            for record in dataset[:min(100, len(dataset))]:  # Sample first 100 records
                if isinstance(record, dict):
                    all_keys.update(record.keys())
            analysis['all_fields'] = sorted(list(all_keys))
            
            # Analyze field consistency
            field_presence = {}
            for key in all_keys:
                count = sum(1 for record in dataset[:100] if isinstance(record, dict) and key in record)
                field_presence[key] = count / min(100, len(dataset))
            analysis['field_presence'] = field_presence
            
    elif isinstance(dataset, dict):
        analysis['top_level_keys'] = list(dataset.keys())
        
    return analysis


def extract_property_features(dataset):
    """
    Extract and standardize property features from the dataset.
    
    Parameters
    ----------
    dataset : list
        List of appraisal records
        
    Returns
    -------
    pandas.DataFrame
        Standardized property features DataFrame
    """
    properties_data = []
    
    for i, record in enumerate(dataset):
        try:
            if not isinstance(record, dict):
                continue
                
            # Extract subject property
            if 'subject_property' in record:
                subject = record['subject_property'].copy()
                subject['record_id'] = i
                subject['property_type'] = 'subject'
                properties_data.append(subject)
            
            # Extract comparable properties
            if 'comparable_properties' in record:
                for j, comp in enumerate(record['comparable_properties']):
                    if isinstance(comp, dict):
                        comp_data = comp.copy()
                        comp_data['record_id'] = i
                        comp_data['comp_index'] = j
                        comp_data['property_type'] = 'comparable'
                        properties_data.append(comp_data)
                        
            # Extract selected comparables
            if 'selected_comparables' in record:
                for j, selected in enumerate(record['selected_comparables']):
                    if isinstance(selected, dict):
                        selected_data = selected.copy()
                        selected_data['record_id'] = i
                        selected_data['selected_index'] = j
                        selected_data['property_type'] = 'selected_comparable'
                        properties_data.append(selected_data)
                        
        except Exception as e:
            print(f"Error processing record {i}: {e}")
            continue
    
    # Convert to DataFrame
    if properties_data:
        df = pd.DataFrame(properties_data)
        print(f"Extracted {len(df)} property records")
        return df
    else:
        print("No property data found in dataset")
        return pd.DataFrame()


def analyze_data_quality(df):
    """
    Analyze data quality and identify cleaning requirements.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Property features DataFrame
        
    Returns
    -------
    dict
        Data quality analysis results
    """
    if df.empty:
        return {"error": "Empty DataFrame provided"}
        
    analysis = {}
    
    # Basic statistics
    analysis['shape'] = df.shape
    analysis['columns'] = list(df.columns)
    analysis['dtypes'] = df.dtypes.to_dict()
    
    # Missing values analysis
    missing_stats = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_stats[col] = {
            'count': int(missing_count),
            'percentage': float(missing_count / len(df) * 100)
        }
    analysis['missing_values'] = missing_stats
    
    # Duplicate analysis
    analysis['duplicates'] = {
        'total_duplicates': int(df.duplicated().sum()),
        'unique_records': int(len(df) - df.duplicated().sum())
    }
    
    # Numerical features analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        analysis['numerical_summary'] = df[numerical_cols].describe().to_dict()
    
    # Categorical features analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_summary = {}
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        categorical_summary[col] = {
            'unique_count': int(unique_vals),
            'top_values': df[col].value_counts().head().to_dict() if unique_vals < 50 else "Too many unique values"
        }
    analysis['categorical_summary'] = categorical_summary
    
    return analysis


if __name__ == "__main__":
    # Example usage
    try:
        # Load dataset
        dataset = load_appraisal_dataset()
        
        # Explore structure
        structure_analysis = explore_dataset_structure(dataset)
        print("\n=== Dataset Structure Analysis ===")
        for key, value in structure_analysis.items():
            print(f"{key}: {value}")
        
        # Extract property features
        properties_df = extract_property_features(dataset)
        
        if not properties_df.empty:
            # Analyze data quality
            quality_analysis = analyze_data_quality(properties_df)
            print("\n=== Data Quality Analysis ===")
            print(f"Shape: {quality_analysis['shape']}")
            print(f"Columns: {len(quality_analysis['columns'])}")
            print(f"Duplicates: {quality_analysis['duplicates']['total_duplicates']}")
            
            # Save processed data
            properties_df.to_csv('processed_properties.csv', index=False)
            print("\nProcessed data saved to 'processed_properties.csv'")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. Git LFS is installed: git lfs install")
        print("2. LFS files are downloaded: git lfs pull")
        print("3. appraisals_dataset.json is in the current directory")