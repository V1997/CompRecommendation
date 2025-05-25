"""
Property similarity search for real estate valuation.

Implements multiple similarity algorithms optimized for real estate
property comparison with geographic and feature-based similarity.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import faiss
from geopy.distance import geodesic
import pickle
import sys
import os

# Add the parent directory to the path to import the preprocessor
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

# Import the preprocessor class to make it available for pickle
try:
    from data_preprocessing.property_preprocessor import PropertyDataPreprocessor
except ImportError:
    # Alternative import paths
    try:
        sys.path.insert(0, os.path.join(grandparent_dir, 'src'))
        from data_preprocessing.property_preprocessor import PropertyDataPreprocessor
    except ImportError:
        # If still failing, create a dummy class to avoid pickle errors
        print("Warning: Could not import PropertyDataPreprocessor. Creating dummy class.")
        class PropertyDataPreprocessor:
            pass


class PropertySimilaritySearch:
    """
    Advanced similarity search for real estate properties.
    
    Combines geographic proximity with feature-based similarity
    to find the most relevant comparable properties.
    """
    
    def __init__(self, algorithm='hybrid', n_neighbors=20, geo_weight=0.3):
        """
        Initialize the property similarity search.
        
        Parameters
        ----------
        algorithm : str, default='hybrid'
            Algorithm to use ('faiss', 'sklearn', 'hybrid')
        n_neighbors : int, default=20
            Number of neighbors to retrieve
        geo_weight : float, default=0.3
            Weight for geographic similarity (0-1)
        """
        self.algorithm = algorithm
        self.n_neighbors = n_neighbors
        self.geo_weight = geo_weight
        self.feature_weight = 1 - geo_weight
        
        # Algorithm components
        self.faiss_index = None
        self.sklearn_index = None
        self.reference_data = None
        self.feature_columns = None
        
        self.is_fitted = False
        
    def fit(self, df, feature_columns):
        """
        Fit the similarity search on property data.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Property data for indexing
        feature_columns : list
            List of feature columns to use for similarity
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        print(f"Fitting similarity search on {len(df)} properties...")
        
        self.reference_data = df.copy()
        self.feature_columns = feature_columns
        
        # Prepare feature matrix
        X = df[feature_columns].values.astype('float32')
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        if self.algorithm in ['faiss', 'hybrid']:
            # Build FAISS index for efficient similarity search
            dimension = X.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(X)
            
        if self.algorithm in ['sklearn', 'hybrid']:
            # Build sklearn index
            self.sklearn_index = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm='auto',
                metric='euclidean'
            )
            self.sklearn_index.fit(X)
        
        self.is_fitted = True
        print(f"Similarity search fitted with {len(feature_columns)} features")
        
        return self
    
    def search(self, query_property, k=5, max_distance_km=10.0):
        """
        Search for similar properties.
        
        Parameters
        ----------
        query_property : pandas.Series or dict
            Query property to find similarities for
        k : int, default=5
            Number of similar properties to return
        max_distance_km : float, default=10.0
            Maximum geographic distance in kilometers
            
        Returns
        -------
        pandas.DataFrame
            Similar properties with similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Similarity search must be fitted before searching")
            
        # Convert query to Series if needed
        if isinstance(query_property, dict):
            query_property = pd.Series(query_property)
            
        if self.algorithm == 'hybrid':
            return self._hybrid_search(query_property, k, max_distance_km)
        elif self.algorithm == 'faiss':
            return self._faiss_search(query_property, k)
        else:
            return self._sklearn_search(query_property, k)
    
    def _hybrid_search(self, query_property, k, max_distance_km):
        """Hybrid search combining geographic and feature similarity."""
        # Step 1: Geographic filtering
        if self._has_location_data(query_property):
            candidates = self._geographic_filter(query_property, max_distance_km)
            
            # If no candidates found and coordinates look scaled, try feature-only search
            if len(candidates) == 0:
                query_lat, query_lon = self._extract_coordinates(query_property)
                # Check if coordinates look like scaled values (typically between -5 and 5)
                if abs(query_lat) < 10 and abs(query_lon) < 10:
                    print("Coordinates appear to be scaled. Using feature-only similarity search...")
                    return self._sklearn_search(query_property, k)
                else:
                    # Try with much larger radius for real coordinates
                    candidates = self._geographic_filter(query_property, max_distance_km * 10)
        else:
            candidates = self.reference_data.copy()
            
        if len(candidates) == 0:
            print("No candidates found within geographic range")
            return pd.DataFrame()
            
        # Step 2: Feature-based similarity on candidates
        query_features = query_property[self.feature_columns].values.reshape(1, -1)
        query_features = np.nan_to_num(query_features, nan=0.0).astype('float32')
        
        candidate_features = candidates[self.feature_columns].values.astype('float32')
        candidate_features = np.nan_to_num(candidate_features, nan=0.0)
        
        # Calculate feature distances
        feature_distances = pairwise_distances(
            query_features, candidate_features, metric='euclidean'
        )[0]
        
        # Calculate geographic distances if available
        if self._has_location_data(query_property) and len(candidates) < len(self.reference_data):
            geo_distances = self._calculate_geographic_distances(
                query_property, candidates
            )
            
            # Normalize distances to [0, 1]
            feature_distances_norm = feature_distances / (feature_distances.max() + 1e-10)
            geo_distances_norm = geo_distances / (geo_distances.max() + 1e-10)
            
            # Combined similarity score
            combined_distances = (
                self.feature_weight * feature_distances_norm +
                self.geo_weight * geo_distances_norm
            )
        else:
            combined_distances = feature_distances
            
        # Get top k similar properties
        top_indices = np.argsort(combined_distances)[:k]
        
        # Prepare results
        results = candidates.iloc[top_indices].copy()
        results['similarity_score'] = 1 - (combined_distances[top_indices] / 
                                         (combined_distances.max() + 1e-10))
        results['feature_distance'] = feature_distances[top_indices]
        
        if self._has_location_data(query_property) and len(candidates) < len(self.reference_data):
            results['geographic_distance_km'] = geo_distances[top_indices]
            
        return results.sort_values('similarity_score', ascending=False)
    
    def _faiss_search(self, query_property, k):
        """FAISS-based similarity search."""
        query_features = query_property[self.feature_columns].values.reshape(1, -1)
        query_features = np.nan_to_num(query_features, nan=0.0).astype('float32')
        
        # Search with FAISS
        distances, indices = self.faiss_index.search(query_features, k)
        
        # Prepare results
        results = self.reference_data.iloc[indices[0]].copy()
        results['similarity_score'] = 1 - (distances[0] / (distances[0].max() + 1e-10))
        results['distance'] = distances[0]
        
        return results.sort_values('similarity_score', ascending=False)
    
    def _sklearn_search(self, query_property, k):
        """Scikit-learn based similarity search."""
        # Extract features and handle mixed types carefully
        query_values = []
        problematic_features = []
        
        for col in self.feature_columns:
            if col in query_property.index:
                value = query_property[col]
                try:
                    # Try to convert to float
                    float_value = float(value)
                    query_values.append(float_value)
                except (ValueError, TypeError):
                    # If conversion fails, use 0.0 as default
                    query_values.append(0.0)
                    problematic_features.append(col)
            else:
                # Feature not in query property
                query_values.append(0.0)
                problematic_features.append(col)
        
        if problematic_features:
            print(f"Warning: {len(problematic_features)} features had conversion issues")
            
        query_features = np.array(query_values, dtype=np.float32).reshape(1, -1)
        
        # Handle any remaining NaN values
        query_features = np.nan_to_num(query_features, nan=0.0)
        
        # Search with sklearn
        distances, indices = self.sklearn_index.kneighbors(query_features)
        
        # Prepare results
        results = self.reference_data.iloc[indices[0]].copy()
        results['similarity_score'] = 1 - (distances[0] / (distances[0].max() + 1e-10))
        results['distance'] = distances[0]
        
        return results.sort_values('similarity_score', ascending=False)
    
    def _has_location_data(self, property_data):
        """Check if property has location data."""
        location_cols = ['latitude', 'longitude', 'lat', 'lon']
        available_cols = [col for col in location_cols if col in property_data.index]
        return len(available_cols) >= 2
    
    def _geographic_filter(self, query_property, max_distance_km):
        """Filter properties by geographic distance."""
        if not self._has_location_data(query_property):
            return self.reference_data.copy()
            
        # Get query coordinates
        query_lat, query_lon = self._extract_coordinates(query_property)
        print(f"Query coordinates: lat={query_lat}, lon={query_lon}")
        
        # Check how many reference properties have location data
        ref_with_location = 0
        valid_coords = 0
        
        # Calculate distances for all properties
        distances = []
        for idx, prop in self.reference_data.iterrows():
            if self._has_location_data(prop):
                ref_with_location += 1
                prop_lat, prop_lon = self._extract_coordinates(prop)
                
                # Check if coordinates are reasonable (rough bounds for Earth)
                if (-90 <= prop_lat <= 90) and (-180 <= prop_lon <= 180):
                    valid_coords += 1
                    try:
                        distance = geodesic((query_lat, query_lon), (prop_lat, prop_lon)).kilometers
                        distances.append(distance)
                    except Exception as e:
                        print(f"Error calculating distance for property {idx}: {e}")
                        distances.append(float('inf'))
                else:
                    distances.append(float('inf'))  # Invalid coordinates
            else:
                distances.append(float('inf'))  # Exclude properties without location
                
        print(f"Reference properties with location data: {ref_with_location}/{len(self.reference_data)}")
        print(f"Reference properties with valid coordinates: {valid_coords}/{len(self.reference_data)}")
        
        # Filter by distance
        distance_array = np.array(distances)
        distance_mask = distance_array <= max_distance_km
        within_range = distance_mask.sum()
        
        print(f"Properties within {max_distance_km}km: {within_range}")
        if within_range > 0:
            min_distance = distance_array[distance_mask].min()
            max_distance = distance_array[distance_mask].max()
            print(f"Distance range: {min_distance:.2f}km to {max_distance:.2f}km")
        else:
            # Show closest properties for debugging
            if len(distance_array[distance_array != float('inf')]) > 0:
                closest_distances = np.sort(distance_array[distance_array != float('inf')])[:5]
                print(f"Closest 5 properties are at: {closest_distances} km")
        
        return self.reference_data[distance_mask].copy()
    
    def _calculate_geographic_distances(self, query_property, candidates):
        """Calculate geographic distances to candidate properties."""
        query_lat, query_lon = self._extract_coordinates(query_property)
        
        distances = []
        for _, prop in candidates.iterrows():
            if self._has_location_data(prop):
                prop_lat, prop_lon = self._extract_coordinates(prop)
                distance = geodesic((query_lat, query_lon), (prop_lat, prop_lon)).kilometers
                distances.append(distance)
            else:
                distances.append(0.0)  # Default for missing location
                
        return np.array(distances)
    
    def _extract_coordinates(self, property_data):
        """Extract latitude and longitude from property data."""
        # Try different column name variations
        lat_cols = ['latitude', 'lat']
        lon_cols = ['longitude', 'lon', 'lng']
        
        lat = None
        lon = None
        
        for col in lat_cols:
            if col in property_data.index and pd.notna(property_data[col]):
                lat = property_data[col]
                break
                
        for col in lon_cols:
            if col in property_data.index and pd.notna(property_data[col]):
                lon = property_data[col]
                break
                
        return lat, lon
    
    def save(self, filepath):
        """Save the fitted similarity search."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted similarity search")
            
        save_data = {
            'algorithm': self.algorithm,
            'n_neighbors': self.n_neighbors,
            'geo_weight': self.geo_weight,
            'feature_columns': self.feature_columns,
            'reference_data': self.reference_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
            
    def load(self, filepath):
        """Load a fitted similarity search."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
            
        self.algorithm = save_data['algorithm']
        self.n_neighbors = save_data['n_neighbors']
        self.geo_weight = save_data['geo_weight']
        self.feature_columns = save_data['feature_columns']
        self.reference_data = save_data['reference_data']
        
        # Rebuild indices
        self.fit(self.reference_data, self.feature_columns)
        
        return self


if __name__ == "__main__":
    # Example usage
    try:
        # Load preprocessed data
        print("Loading preprocessed property data...")
        df = pd.read_csv('data/processed/properties_preprocessed.csv')
        
        # Load preprocessor to get feature names
        with open('data/models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
            
        # Get feature names from the preprocessor
        all_features = preprocessor.get_feature_names()
        
        # Filter features that exist in the current dataframe
        feature_columns = [col for col in all_features if col in df.columns]
        print(f"Using {len(feature_columns)} features for similarity search (out of {len(all_features)} total)")
        
        if len(feature_columns) != len(all_features):
            missing_features = [col for col in all_features if col not in df.columns]
            print(f"Missing features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
        
        # Initialize and fit similarity search
        similarity_search = PropertySimilaritySearch(
            algorithm='hybrid',
            n_neighbors=20,
            geo_weight=0.3
        )
        
        # Fit on potential comparables (the pool for similarity search)
        potential_comps = df[df['property_role'] == 'potential_comparable']
        similarity_search.fit(potential_comps, feature_columns)
        
        # Test with a subject property
        subjects = df[df['property_role'] == 'subject']
        if len(subjects) > 0:
            test_subject = subjects.iloc[0]
            print(f"\nTesting similarity search with subject property from appraisal {test_subject['appraisal_index']}")
            
            # Find similar properties
            similar_props = similarity_search.search(
                test_subject, 
                k=5, 
                max_distance_km=10.0
            )
            
            print(f"Found {len(similar_props)} similar properties")
            
            if len(similar_props) > 0:
                print("Top 3 similar properties:")
                display_cols = ['similarity_score']
                if 'feature_distance' in similar_props.columns:
                    display_cols.append('feature_distance')
                if 'geographic_distance_km' in similar_props.columns:
                    display_cols.append('geographic_distance_km')
                    
                print(similar_props[display_cols].head(3))
            else:
                print("No similar properties found. Trying with larger search radius...")
                
                # Try with a much larger radius
                similar_props_large = similarity_search.search(
                    test_subject, 
                    k=5, 
                    max_distance_km=50.0  # Much larger radius
                )
                
                if len(similar_props_large) > 0:
                    print(f"Found {len(similar_props_large)} properties with 50km radius")
                    display_cols = ['similarity_score']
                    if 'feature_distance' in similar_props_large.columns:
                        display_cols.append('feature_distance')
                    if 'geographic_distance_km' in similar_props_large.columns:
                        display_cols.append('geographic_distance_km')
                    print(similar_props_large[display_cols].head(3))
                else:
                    print("Still no properties found. Checking if location data exists...")
                    # Check location columns
                    location_cols = ['latitude', 'longitude', 'lat', 'lon']
                    available_loc_cols = [col for col in location_cols if col in test_subject.index]
                    print(f"Available location columns: {available_loc_cols}")
                    if available_loc_cols:
                        for col in available_loc_cols:
                            print(f"{col}: {test_subject[col]}")
                    else:
                        print("No location data found in subject property. Using feature-only search...")
                        # Try feature-only search
                        feature_only_search = PropertySimilaritySearch(
                            algorithm='sklearn',
                            n_neighbors=20
                        )
                        feature_only_search.fit(potential_comps, feature_columns)
                        similar_props_features = feature_only_search.search(test_subject, k=5)
                        
                        if len(similar_props_features) > 0:
                            print(f"Found {len(similar_props_features)} similar properties using feature-only search")
                            display_cols = ['similarity_score']
                            if 'distance' in similar_props_features.columns:
                                display_cols.append('distance')
                            print(similar_props_features[display_cols].head(3))
            
        # Save the fitted similarity search
        similarity_search.save('data/models/similarity_search.pkl')
        print("\nSimilarity search saved to 'data/models/similarity_search.pkl'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()