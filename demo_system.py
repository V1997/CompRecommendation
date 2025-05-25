#!/usr/bin/env python3
"""
Simple demonstration of the property recommendation system.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, 'src')
from similarity_search.property_similarity import PropertySimilaritySearch

def main():
    print("=== Property Recommendation System Demo ===\n")
    
    # Load data
    print("Loading property data...")
    properties_df = pd.read_csv('data/processed/properties_preprocessed.csv')
    print(f"✓ Loaded {len(properties_df):,} properties")
    
    # Initialize similarity search
    print("\nInitializing similarity search...")
    similarity_search = PropertySimilaritySearch(algorithm='sklearn', n_neighbors=10)
    
    # Prepare features
    feature_columns = [col for col in properties_df.columns 
                      if col not in ['property_id', 'appraisal_id'] and 
                      pd.api.types.is_numeric_dtype(properties_df[col])]
    
    print(f"✓ Using {len(feature_columns)} numerical features")
    
    # Fit the model
    print("\nFitting similarity model...")
    similarity_search.fit(properties_df, feature_columns)
    print("✓ Model fitted successfully")
    
    # Demonstrate recommendations
    print("\n=== Property Recommendations Demo ===")
    
    # Test with a few different properties
    test_indices = [0, 100, 500, 1000]
    
    for i, idx in enumerate(test_indices, 1):
        print(f"\nExample {i}: Property at index {idx}")
        
        # Get subject property
        subject_property = properties_df.iloc[idx]
        
        # Find similar properties
        recommendations = similarity_search.search(subject_property, k=5)
        
        # Show key features of subject property
        key_features = ['gla', 'year_built', 'bedrooms', 'num_baths']
        available_features = [f for f in key_features if f in subject_property.index and pd.notna(subject_property[f])]
        
        print("  Subject Property Features:")
        for feature in available_features:
            value = subject_property[feature]
            if isinstance(value, (int, float)):
                print(f"    {feature}: {value:.0f}")
            else:
                print(f"    {feature}: {value}")
        
        print(f"  Top 3 Similar Properties:")
        for j, (rec_idx, row) in enumerate(recommendations.head(3).iterrows(), 1):
            score = row['similarity_score']
            print(f"    {j}. Index {rec_idx} (similarity: {score:.3f})")
    
    # Performance test
    print("\n=== Performance Test ===")
    import time
    
    test_property = properties_df.iloc[0]
    start_time = time.time()
    
    for _ in range(50):
        results = similarity_search.search(test_property, k=5)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 50
    
    print(f"Average search time: {avg_time*1000:.2f} ms")
    print(f"Searches per second: {1/avg_time:.0f}")
    
    # Save the model
    print("\nSaving model...")
    similarity_search.save('data/models/similarity_search_demo.pkl')
    print("✓ Model saved successfully")
    
    print("\n=== Demo Complete ===")
    print("✓ Property recommendation system is fully functional!")
    print(f"✓ {len(properties_df):,} properties indexed")
    print(f"✓ {len(feature_columns)} features used for similarity")
    print(f"✓ Fast search performance: {avg_time*1000:.2f} ms per query")

if __name__ == "__main__":
    main()
