"""
Corrected data extractor for the real estate appraisal dataset.

This module properly extracts property data based on the actual
dataset structure with fields: orderID, subject, comps, properties.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_appraisal_dataset(file_path="appraisals_dataset.json"):
    """
    Load the appraisal dataset from JSON file.
    
    Parameters
    ----------
    file_path : str, default="appraisals_dataset.json"
        Path to the dataset file
        
    Returns
    -------
    list
        List of appraisal records
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"Loading dataset from {file_path} ({file_size_mb:.1f}MB)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Handle the nested structure
        if isinstance(raw_data, dict) and 'appraisals' in raw_data:
            dataset = raw_data['appraisals']
        elif isinstance(raw_data, list):
            dataset = raw_data
        else:
            raise ValueError("Unexpected dataset structure")
            
        print(f"Successfully loaded dataset with {len(dataset)} appraisal records")
        return dataset
        
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")


def extract_properties_corrected(dataset):
    """
    Extract property features using the correct field names.
    
    Based on the actual dataset structure:
    - 'subject': subject property
    - 'comps': comparable properties (appraiser selected)
    - 'properties': all potential comparable properties
    - 'orderID': unique identifier for each appraisal
    
    Parameters
    ----------
    dataset : list
        List of appraisal records
        
    Returns
    -------
    pandas.DataFrame
        Extracted and standardized property data
    """
    all_properties = []
    
    print(f"Processing {len(dataset)} appraisal records...")
    
    for appraisal_idx, record in enumerate(dataset):
        try:
            if not isinstance(record, dict):
                print(f"Skipping record {appraisal_idx}: not a dictionary")
                continue
                
            order_id = record.get('orderID', f'appraisal_{appraisal_idx}')
            
            # Extract subject property
            if 'subject' in record:
                subject_data = record['subject']
                
                if isinstance(subject_data, dict):
                    subject_property = subject_data.copy()
                    subject_property.update({
                        'orderID': order_id,
                        'appraisal_index': appraisal_idx,
                        'property_role': 'subject',
                        'is_appraiser_selected': False  # Subject is not a comp
                    })
                    all_properties.append(subject_property)
                    
            # Extract appraiser-selected comparables
            if 'comps' in record:
                comps_data = record['comps']
                
                if isinstance(comps_data, list):
                    for comp_idx, comp in enumerate(comps_data):
                        if isinstance(comp, dict):
                            comp_property = comp.copy()
                            comp_property.update({
                                'orderID': order_id,
                                'appraisal_index': appraisal_idx,
                                'comp_index': comp_idx,
                                'property_role': 'selected_comparable',
                                'is_appraiser_selected': True
                            })
                            all_properties.append(comp_property)
                            
            # Extract all potential comparables
            if 'properties' in record:
                properties_data = record['properties']
                
                if isinstance(properties_data, list):
                    for prop_idx, prop in enumerate(properties_data):
                        if isinstance(prop, dict):
                            property_record = prop.copy()
                            property_record.update({
                                'orderID': order_id,
                                'appraisal_index': appraisal_idx,
                                'property_index': prop_idx,
                                'property_role': 'potential_comparable',
                                'is_appraiser_selected': False  # These are not selected by appraiser
                            })
                            all_properties.append(property_record)
                            
        except Exception as e:
            print(f"Error processing appraisal {appraisal_idx}: {e}")
            continue
    
    if all_properties:
        df = pd.DataFrame(all_properties)
        
        # Add unique property ID
        df['property_id'] = range(len(df))
        
        print(f"Successfully extracted {len(df)} property records")
        return df
    else:
        print("No property data extracted")
        return pd.DataFrame()


def analyze_extracted_data(df):
    """
    Analyze the extracted property data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Extracted property data
        
    Returns
    -------
    dict
        Analysis results
    """
    if df.empty:
        return {"error": "No data to analyze"}
        
    analysis = {}
    
    # Basic statistics
    analysis['total_properties'] = len(df)
    analysis['total_appraisals'] = df['appraisal_index'].nunique()
    analysis['columns'] = list(df.columns)
    
    # Property role distribution
    if 'property_role' in df.columns:
        role_distribution = df['property_role'].value_counts().to_dict()
        analysis['property_role_distribution'] = role_distribution
        
    # Appraiser selection analysis
    if 'is_appraiser_selected' in df.columns:
        selection_stats = df['is_appraiser_selected'].value_counts().to_dict()
        analysis['appraiser_selection_stats'] = selection_stats
        
    # Properties per appraisal
    if 'appraisal_index' in df.columns:
        props_per_appraisal = df.groupby('appraisal_index').size()
        analysis['properties_per_appraisal'] = {
            'mean': float(props_per_appraisal.mean()),
            'min': int(props_per_appraisal.min()),
            'max': int(props_per_appraisal.max()),
            'median': float(props_per_appraisal.median())
        }
        
    # Selected comps per appraisal
    selected_comps = df[df['property_role'] == 'selected_comparable']
    if not selected_comps.empty:
        comps_per_appraisal = selected_comps.groupby('appraisal_index').size()
        analysis['selected_comps_per_appraisal'] = {
            'mean': float(comps_per_appraisal.mean()),
            'min': int(comps_per_appraisal.min()),
            'max': int(comps_per_appraisal.max()),
            'median': float(comps_per_appraisal.median())
        }
        
    # Missing values analysis
    missing_analysis = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_analysis[col] = {
            'count': int(missing_count),
            'percentage': float(missing_count / len(df) * 100)
        }
    analysis['missing_values'] = missing_analysis
    
    # Data types analysis
    analysis['data_types'] = df.dtypes.to_dict()
    
    return analysis


def identify_real_estate_features(df):
    """
    Identify real estate features in the extracted data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Extracted property data
        
    Returns
    -------
    dict
        Categorized real estate features
    """
    if df.empty:
        return {}
        
    columns = [col.lower() for col in df.columns]
    
    # Define real estate feature patterns
    feature_patterns = {
        'size_features': [
            'sqft', 'square_feet', 'living_area', 'total_area', 'floor_area',
            'lot_size', 'lot_area', 'land_area', 'acres', 'gla'
        ],
        'structural_features': [
            'bedroom', 'bathroom', 'room', 'bed', 'bath', 'full_bath', 'half_bath',
            'story', 'floor', 'year_built', 'age', 'construction'
        ],
        'location_features': [
            'latitude', 'longitude', 'lat', 'lon', 'address', 'street', 'city', 
            'state', 'zip', 'postal', 'neighborhood', 'district', 'county'
        ],
        'financial_features': [
            'price', 'value', 'assessment', 'tax', 'asking', 'list', 'sold', 'market'
        ],
        'temporal_features': [
            'date', 'time', 'year', 'month', 'day', 'sale_date', 'list_date'
        ],
        'property_characteristics': [
            'type', 'style', 'condition', 'quality', 'grade', 'rating', 'class'
        ]
    }
    
    identified_features = {}
    
    for category, patterns in feature_patterns.items():
        matches = []
        for pattern in patterns:
            # Find columns containing the pattern
            matching_cols = [col for col in df.columns if pattern in col.lower()]
            matches.extend(matching_cols)
        
        # Remove duplicates and store
        identified_features[category] = list(set(matches))
    
    # Find uncategorized features
    all_identified = set()
    for features in identified_features.values():
        all_identified.update(features)
    
    # Exclude system fields
    system_fields = [
        'orderID', 'appraisal_index', 'property_id', 'property_role', 
        'is_appraiser_selected', 'comp_index', 'property_index'
    ]
    
    uncategorized = [
        col for col in df.columns 
        if col not in all_identified and col not in system_fields
    ]
    identified_features['uncategorized'] = uncategorized
    
    return identified_features


if __name__ == "__main__":
    try:
        print("=== Corrected Real Estate Data Extraction ===")
        
        # Load dataset
        dataset = load_appraisal_dataset()
        
        # Extract properties with correct field mapping
        properties_df = extract_properties_corrected(dataset)
        
        if not properties_df.empty:
            print(f"\n=== Extraction Results ===")
            print(f"Total properties extracted: {len(properties_df)}")
            print(f"Columns: {list(properties_df.columns)}")
            
            # Show sample data
            print(f"\n=== Sample Data ===")
            print(properties_df.head())
            
            # Analyze the data
            print(f"\n=== Data Analysis ===")
            analysis = analyze_extracted_data(properties_df)
            
            for key, value in analysis.items():
                if key not in ['missing_values', 'data_types']:  # Skip detailed sections for main output
                    print(f"{key}: {value}")
            
            # Identify real estate features
            print(f"\n=== Real Estate Features ===")
            features = identify_real_estate_features(properties_df)
            for category, feature_list in features.items():
                if feature_list:  # Only show categories with features
                    print(f"{category}: {feature_list}")
            
            # Save results
            print(f"\n=== Saving Results ===")
            properties_df.to_csv('extracted_properties_corrected.csv', index=False)
            print("Property data saved to 'extracted_properties_corrected.csv'")
            
            # Save analysis
            with open('extraction_analysis.json', 'w') as f:
                json.dump({
                    'analysis': analysis,
                    'features': features
                }, f, indent=2, default=str)
            print("Analysis saved to 'extraction_analysis.json'")
            
        else:
            print("Failed to extract any property data. Please check the dataset structure.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()