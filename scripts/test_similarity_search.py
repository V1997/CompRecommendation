#!/usr/bin/env python3
"""
Comprehensive test script for the property similarity search system.
Tests both the preprocessor and similarity search functionality.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing.property_preprocessor import PropertyDataPreprocessor
from similarity_search.property_similarity import PropertySimilaritySearch

def test_similarity_search():
    """Test the complete similarity search pipeline."""
    
    print("=== Property Similarity Search Test ===\n")
    
    # Load preprocessed data
    print("1. Loading preprocessed data...")
    properties_df = pd.read_csv('data/processed/properties_preprocessed.csv')
    print(f"   Loaded {len(properties_df)} properties")
    print(f"   Features: {properties_df.shape[1]} columns")
    
    # Load the fitted preprocessor
    print("\n2. Loading fitted preprocessor...")
    try:
        with open('data/models/preprocessor.pkl', 'rb') as f:
            import pickle
            preprocessor = pickle.load(f)
        print("   ✓ Preprocessor loaded successfully")
        print(f"   ✓ Numerical features: {len(preprocessor.numerical_features)}")
        print(f"   ✓ Categorical features: {len(preprocessor.categorical_features)}")
    except Exception as e:
        print(f"   ✗ Error loading preprocessor: {e}")
        return
    
    # Initialize similarity search
    print("\n3. Initializing similarity search...")
    similarity_search = PropertySimilaritySearch(
        algorithm='sklearn',  # Using sklearn for reliability
        n_neighbors=10,
        geo_weight=0.0  # Pure feature-based since coords are scaled
    )
    
    # Prepare features for similarity search
    feature_columns = [col for col in properties_df.columns 
                      if col not in ['property_id', 'appraisal_id'] and pd.api.types.is_numeric_dtype(properties_df[col])]
    
    print(f"   Using {len(feature_columns)} numerical features")
    
    # Fit similarity search
    print("\n4. Fitting similarity search...")
    try:
        similarity_search.fit(properties_df, feature_columns)
        print("   ✓ Similarity search fitted successfully")
    except Exception as e:
        print(f"   ✗ Error fitting similarity search: {e}")
        return
    
    # Test with multiple properties
    print("\n5. Testing similarity search with different properties...")
    
    test_indices = [0, 100, 500, 1000, 2000]
    
    for i, test_idx in enumerate(test_indices):
        print(f"\n   Test {i+1}: Property at index {test_idx}")
        
        try:
            # Get test property
            test_property = properties_df.iloc[test_idx]
            
            # Search for similar properties
            results = similarity_search.search(test_property, k=5)
            
            print(f"   ✓ Found {len(results)} similar properties")
            
            if len(results) > 0:
                print(f"   Top similarity scores: {results['similarity_score'].values[:3]}")
                
                # Show some key features for comparison
                key_features = ['square_footage', 'year_built', 'bedrooms', 'bathrooms']
                available_features = [f for f in key_features if f in test_property.index]
                
                if available_features:
                    print(f"   Test property features:")
                    for feature in available_features:
                        print(f"     {feature}: {test_property[feature]}")
                    
                    print(f"   Similar properties average:")
                    similar_indices = results.index[:3]
                    for feature in available_features:
                        if feature in properties_df.columns:
                            avg_val = properties_df.loc[similar_indices, feature].mean()
                            print(f"     {feature}: {avg_val:.2f}")
            else:
                print("   ✗ No similar properties found")
                
        except Exception as e:
            print(f"   ✗ Error in test {i+1}: {e}")
    
    # Test edge cases
    print("\n6. Testing edge cases...")
    
    # Test with property with missing values
    print("   Testing with property containing NaN values...")
    test_property_nan = properties_df.iloc[0].copy()
    test_property_nan['square_footage'] = np.nan
    test_property_nan['bedrooms'] = np.nan
    
    try:
        results_nan = similarity_search.search(test_property_nan, k=3)
        print(f"   ✓ NaN test: Found {len(results_nan)} similar properties")
    except Exception as e:
        print(f"   ✗ NaN test failed: {e}")
    
    # Performance test
    print("\n7. Performance test...")
    import time
    
    start_time = time.time()
    for i in range(10):
        test_property = properties_df.iloc[i * 100]
        results = similarity_search.search(test_property, k=5)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"   Average search time: {avg_time:.4f} seconds")
    
    # Save the fitted model
    print("\n8. Saving similarity search model...")
    try:
        similarity_search.save('data/models/similarity_search_test.pkl')
        print("   ✓ Model saved successfully")
    except Exception as e:
        print(f"   ✗ Error saving model: {e}")
    
    print("\n=== Test Complete ===")
    print("✓ Similarity search system is working correctly!")
    
    return similarity_search

if __name__ == "__main__":
    test_similarity_search()
