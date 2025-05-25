"""
Fixed data loader for the real estate appraisal dataset.

This module handles the correct structure where appraisals are nested
under a top-level 'appraisals' key.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_appraisal_dataset(file_path="appraisals_dataset.json"):
    """
    Load the appraisal dataset from JSON file with correct structure handling.
    
    Parameters
    ----------
    file_path : str, default="appraisals_dataset.json"
        Path to the dataset file
        
    Returns
    -------
    list
        List of appraisal records
        
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
            raw_data = json.load(f)
            
        # Handle the nested structure
        if isinstance(raw_data, dict) and 'appraisals' in raw_data:
            dataset = raw_data['appraisals']
            print(f"Successfully loaded dataset with {len(dataset)} appraisal records")
        elif isinstance(raw_data, list):
            dataset = raw_data
            print(f"Successfully loaded dataset with {len(dataset)} records")
        else:
            raise ValueError("Unexpected dataset structure. Expected 'appraisals' key or list of records.")
            
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
    dataset : list
        List of appraisal records
        
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
            'sample_keys_preview': list(first_record.keys())[:10] if isinstance(first_record, dict) else None
        }
        
        # Check for common fields across records
        if isinstance(first_record, dict):
            all_keys = set()
            sample_size = min(10, len(dataset))  # Sample first 10 records for structure analysis
            
            for record in dataset[:sample_size]:
                if isinstance(record, dict):
                    all_keys.update(record.keys())
            
            analysis['all_fields'] = sorted(list(all_keys))
            
            # Analyze field consistency
            field_presence = {}
            for key in all_keys:
                count = sum(1 for record in dataset[:sample_size] if isinstance(record, dict) and key in record)
                field_presence[key] = count / sample_size
            analysis['field_presence'] = field_presence
            
            # Analyze nested structure of important fields
            important_fields = ['subject_property', 'comparable_properties', 'selected_comparables']
            nested_analysis = {}
            
            for field in important_fields:
                if field in first_record:
                    field_data = first_record[field]
                    if isinstance(field_data, dict):
                        nested_analysis[field] = {
                            'type': 'dict',
                            'keys': list(field_data.keys())[:10],  # First 10 keys
                            'sample_record': {k: str(v)[:50] + "..." if len(str(v)) > 50 else v 
                                            for k, v in list(field_data.items())[:3]}
                        }
                    elif isinstance(field_data, list):
                        nested_analysis[field] = {
                            'type': 'list',
                            'length': len(field_data),
                            'sample_element_type': type(field_data[0]).__name__ if field_data else 'empty',
                            'sample_element_keys': list(field_data[0].keys())[:10] if field_data and isinstance(field_data[0], dict) else None
                        }
                    else:
                        nested_analysis[field] = {
                            'type': type(field_data).__name__,
                            'value': str(field_data)[:100] + "..." if len(str(field_data)) > 100 else field_data
                        }
            
            analysis['nested_structure'] = nested_analysis
            
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
    
    print(f"Processing {len(dataset)} appraisal records...")
    
    for i, record in enumerate(dataset):
        try:
            if not isinstance(record, dict):
                continue
                
            # Extract subject property
            if 'subject_property' in record and isinstance(record['subject_property'], dict):
                subject = record['subject_property'].copy()
                subject['appraisal_id'] = i
                subject['property_role'] = 'subject'
                properties_data.append(subject)
            
            # Extract comparable properties
            if 'comparable_properties' in record and isinstance(record['comparable_properties'], list):
                for j, comp in enumerate(record['comparable_properties']):
                    if isinstance(comp, dict):
                        comp_data = comp.copy()
                        comp_data['appraisal_id'] = i
                        comp_data['comp_index'] = j
                        comp_data['property_role'] = 'comparable'
                        properties_data.append(comp_data)
                        
            # Extract selected comparables
            if 'selected_comparables' in record and isinstance(record['selected_comparables'], list):
                for j, selected in enumerate(record['selected_comparables']):
                    if isinstance(selected, dict):
                        selected_data = selected.copy()
                        selected_data['appraisal_id'] = i
                        selected_data['selected_index'] = j
                        selected_data['property_role'] = 'selected'
                        selected_data['appraiser_selected'] = True
                        properties_data.append(selected_data)
                        
        except Exception as e:
            print(f"Error processing record {i}: {e}")
            continue
    
    # Convert to DataFrame
    if properties_data:
        df = pd.DataFrame(properties_data)
        print(f"Extracted {len(df)} property records")
        
        # Add a unique property ID
        df['property_id'] = range(len(df))
        
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
    
    # Property role distribution
    if 'property_role' in df.columns:
        role_dist = df['property_role'].value_counts().to_dict()
        analysis['property_role_distribution'] = role_dist
    
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
        numerical_summary = {}
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                numerical_summary[col] = {
                    'count': len(col_data),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median())
                }
        analysis['numerical_summary'] = numerical_summary
    
    # Categorical features analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_summary = {}
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        categorical_summary[col] = {
            'unique_count': int(unique_vals),
            'top_values': df[col].value_counts().head(5).to_dict() if unique_vals < 100 else "Too many unique values"
        }
    analysis['categorical_summary'] = categorical_summary
    
    return analysis


def identify_real_estate_features(df):
    """
    Identify and categorize real estate-specific features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Property features DataFrame
        
    Returns
    -------
    dict
        Categorized features for real estate analysis
    """
    all_columns = set(df.columns.str.lower())
    
    # Define feature categories based on common real estate attributes
    feature_categories = {
        'size_features': [
            'square_feet', 'sqft', 'living_area', 'total_area', 'floor_area',
            'lot_size', 'lot_area', 'land_area', 'acres', 'lot_sqft'
        ],
        'structural_features': [
            'bedrooms', 'bathrooms', 'rooms', 'beds', 'baths', 'full_baths', 'half_baths',
            'stories', 'floors', 'year_built', 'age', 'construction_year'
        ],
        'location_features': [
            'latitude', 'longitude', 'lat', 'lon', 'address', 'street', 'city', 'state', 'zip',
            'zipcode', 'postal_code', 'neighborhood', 'district', 'county'
        ],
        'financial_features': [
            'sale_price', 'price', 'value', 'assessment', 'tax_assessment', 'asking_price',
            'list_price', 'sold_price', 'market_value'
        ],
        'temporal_features': [
            'sale_date', 'list_date', 'sold_date', 'assessment_date', 'appraisal_date'
        ],
        'property_type_features': [
            'property_type', 'building_type', 'style', 'construction_type',
            'use_code', 'zoning', 'classification'
        ],
        'condition_features': [
            'condition', 'quality', 'grade', 'rating', 'status', 'renovation', 'updated'
        ]
    }
    
    # Find matching features
    identified_features = {}
    for category, keywords in feature_categories.items():
        matches = []
        for keyword in keywords:
            # Exact matches
            exact_matches = [col for col in df.columns if col.lower() == keyword]
            matches.extend(exact_matches)
            
            # Partial matches (contains keyword)
            partial_matches = [col for col in df.columns if keyword in col.lower() and col not in matches]
            matches.extend(partial_matches)
        
        identified_features[category] = list(set(matches))  # Remove duplicates
    
    # Identify uncategorized features
    all_identified = set()
    for features in identified_features.values():
        all_identified.update(features)
    
    uncategorized = [col for col in df.columns if col not in all_identified]
    identified_features['uncategorized'] = unategorized
    
    return identified_features


if __name__ == "__main__":
    # Example usage with corrected structure handling
    try:
        # Load dataset
        print("=== Loading Dataset ===")
        dataset = load_appraisal_dataset()
        
        # Explore structure
        print("\n=== Dataset Structure Analysis ===")
        structure_analysis = explore_dataset_structure(dataset)
        for key, value in structure_analysis.items():
            if key == 'nested_structure':
                print(f"\n{key}:")
                for nested_key, nested_value in value.items():
                    print(f"  {nested_key}: {nested_value}")
            else:
                print(f"{key}: {value}")
        
        # Extract property features
        print("\n=== Extracting Property Features ===")
        properties_df = extract_property_features(dataset)
        
        if not properties_df.empty:
            print(f"\nDataFrame shape: {properties_df.shape}")
            print(f"Columns ({len(properties_df.columns)}): {list(properties_df.columns)}")
            
            # Show sample of the data
            print(f"\nFirst few rows:")
            print(properties_df.head())
            
            # Analyze data quality
            print("\n=== Data Quality Analysis ===")
            quality_analysis = analyze_data_quality(properties_df)
            
            print(f"Total records: {quality_analysis['shape'][0]}")
            print(f"Total features: {quality_analysis['shape'][1]}")
            print(f"Duplicates: {quality_analysis['duplicates']['total_duplicates']}")
            
            if 'property_role_distribution' in quality_analysis:
                print("\nProperty Role Distribution:")
                for role, count in quality_analysis['property_role_distribution'].items():
                    print(f"  {role}: {count}")
            
            # Identify real estate features
            print("\n=== Real Estate Feature Analysis ===")
            feature_categories = identify_real_estate_features(properties_df)
            for category, features in feature_categories.items():
                if features:  # Only show categories with features
                    print(f"{category}: {features}")
            
            # Save processed data
            print("\n=== Saving Processed Data ===")
            properties_df.to_csv('processed_properties.csv', index=False)
            print("Processed data saved to 'processed_properties.csv'")
            
            # Save analysis results
            import json
            with open('data_analysis_results.json', 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                analysis_results = {
                    'structure_analysis': structure_analysis,
                    'quality_analysis': quality_analysis,
                    'feature_categories': feature_categories
                }
                
                json.dump(analysis_results, f, indent=2, default=convert_numpy)
            print("Analysis results saved to 'data_analysis_results.json'")
            
        else:
            print("No properties extracted. Please check the dataset structure.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure:")
        print("1. Git LFS is installed: git lfs install")
        print("2. LFS files are downloaded: git lfs pull")
        print("3. appraisals_dataset.json is in the current directory")