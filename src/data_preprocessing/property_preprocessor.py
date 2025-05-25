"""
Property data preprocessor for real estate valuation system.

This module handles data cleaning, normalization, and feature engineering
for property comparison and valuation tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')


class PropertyDataPreprocessor:
    """
    Comprehensive preprocessor for real estate property data.
    
    Handles missing values, feature engineering, and data normalization
    optimized for property comparison tasks.
    """
    
    def __init__(self, scaling_method='robust', imputation_strategy='knn'):
        """
        Initialize the property data preprocessor.
        
        Parameters
        ----------
        scaling_method : str, default='robust'
            Scaling method ('standard', 'robust', 'minmax')
        imputation_strategy : str, default='knn'
            Missing value imputation strategy ('mean', 'median', 'knn')
        """
        self.scaling_method = scaling_method
        self.imputation_strategy = imputation_strategy
        
        # Initialize scalers and imputers
        if scaling_method == 'robust':
            self.numerical_scaler = RobustScaler()
        else:
            self.numerical_scaler = StandardScaler()
            
        if imputation_strategy == 'knn':
            self.numerical_imputer = KNNImputer(n_neighbors=5)
        else:
            self.numerical_imputer = SimpleImputer(strategy=imputation_strategy)
            
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.label_encoders = {}
        
        # Feature categories
        self.numerical_features = []
        self.categorical_features = []
        self.location_features = []
        self.engineered_features = []
        
        self.is_fitted = False
        
    def fit(self, df):
        """
        Fit the preprocessor on the training data.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Training property data
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        print("Fitting property data preprocessor...")
        
        # Create a copy for feature engineering
        df_fitted = df.copy()
        
        # Engineer features first to get complete feature set
        df_fitted = self._engineer_features(df_fitted)
        
        # Identify feature types on the engineered data
        self._identify_feature_types(df_fitted)
        
        # Fit imputers and scalers
        if self.numerical_features:
            print(f"Original numerical features: {len(self.numerical_features)}")
            
            # First, filter out features that are all NaN or non-numeric
            valid_features = []
            for col in self.numerical_features:
                col_data = df_fitted[col]
                # Check if column is numeric and has at least some non-null values
                if pd.api.types.is_numeric_dtype(col_data) and not col_data.isna().all():
                    valid_features.append(col)
                else:
                    print(f"Removing feature {col}: not numeric or all NaN")
            
            # Update numerical features to only include valid ones
            self.numerical_features = valid_features
            print(f"Valid numerical features after filtering: {len(self.numerical_features)}")
            
            if self.numerical_features:
                # Fit numerical imputer on valid features
                self.numerical_imputer.fit(df_fitted[self.numerical_features])
                
                # Impute and check for zero variance
                imputed_numerical = self.numerical_imputer.transform(df_fitted[self.numerical_features])
                
                # Check for zero variance after imputation
                final_valid_features = []
                for i, col in enumerate(self.numerical_features):
                    col_data = imputed_numerical[:, i]
                    if len(np.unique(col_data[~np.isnan(col_data)])) > 1:
                        final_valid_features.append(col)
                    else:
                        print(f"Removing feature {col}: zero variance after imputation")
                
                # Update features list one more time
                if len(final_valid_features) != len(self.numerical_features):
                    self.numerical_features = final_valid_features
                    print(f"Final valid numerical features: {len(self.numerical_features)}")
                    
                    # Re-fit imputer and scaler with final feature set
                    if self.numerical_features:
                        self.numerical_imputer.fit(df_fitted[self.numerical_features])
                        imputed_numerical = self.numerical_imputer.transform(df_fitted[self.numerical_features])
                        self.numerical_scaler.fit(imputed_numerical)
                else:
                    # Fit scaler on current imputed data
                    self.numerical_scaler.fit(imputed_numerical)
            
        if self.categorical_features:
            # Fit categorical imputer
            self.categorical_imputer.fit(df_fitted[self.categorical_features])
            
            # Fit label encoders
            imputed_categorical = self.categorical_imputer.transform(df_fitted[self.categorical_features])
            categorical_df = pd.DataFrame(
                imputed_categorical, 
                columns=self.categorical_features
            )
            
            for col in self.categorical_features:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(categorical_df[col].astype(str))
        
        self.is_fitted = True
        print(f"Preprocessor fitted on {len(df)} records")
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        
        return self
    
    def transform(self, df):
        """
        Transform the data using fitted preprocessors.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Property data to transform
            
        Returns
        -------
        pandas.DataFrame
            Transformed property data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
            
        df_processed = df.copy()
        
        # Engineer new features
        df_processed = self._engineer_features(df_processed)
        
        # Handle numerical features
        if self.numerical_features:
            # Check which features exist in the current dataframe
            available_numerical = [col for col in self.numerical_features if col in df_processed.columns]
            print(f"Available numerical features: {len(available_numerical)} out of {len(self.numerical_features)}")
            
            if len(available_numerical) != len(self.numerical_features):
                missing_features = [col for col in self.numerical_features if col not in df_processed.columns]
                print(f"Missing features: {missing_features}")
            
            # Impute missing values
            imputed_numerical = self.numerical_imputer.transform(
                df_processed[available_numerical]
            )
            
            # Scale features
            scaled_numerical = self.numerical_scaler.transform(imputed_numerical)
            
            print(f"Scaled numerical shape: {scaled_numerical.shape}")
            print(f"Available numerical features length: {len(available_numerical)}")
            
            # Update dataframe
            df_processed[available_numerical] = scaled_numerical
            
        # Handle categorical features
        if self.categorical_features:
            # Impute missing values
            imputed_categorical = self.categorical_imputer.transform(
                df_processed[self.categorical_features]
            )
            categorical_df = pd.DataFrame(
                imputed_categorical, 
                columns=self.categorical_features,
                index=df_processed.index
            )
            
            # Encode categorical variables
            for col in self.categorical_features:
                try:
                    df_processed[col] = self.label_encoders[col].transform(
                        categorical_df[col].astype(str)
                    )
                except ValueError:
                    # Handle unseen categories
                    df_processed[col] = self.label_encoders[col].transform(
                        categorical_df[col].astype(str).fillna('unknown')
                    )
        
        return df_processed
    
    def fit_transform(self, df):
        """
        Fit the preprocessor and transform the data.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Property data to fit and transform
            
        Returns
        -------
        pandas.DataFrame
            Transformed property data
        """
        return self.fit(df).transform(df)
    
    def _identify_feature_types(self, df):
        """Identify numerical and categorical features."""
        # System columns to exclude
        system_columns = [
            'orderID', 'appraisal_index', 'property_id', 'property_role',
            'is_appraiser_selected', 'comp_index', 'property_index'
        ]
        
        # Identify numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        self.numerical_features = [
            col for col in numerical_cols 
            if col not in system_columns
        ]
        
        # Identify categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        self.categorical_features = [
            col for col in categorical_cols 
            if col not in system_columns
        ]
        
        # Identify location features
        location_patterns = ['lat', 'lon', 'latitude', 'longitude']
        self.location_features = [
            col for col in self.numerical_features
            if any(pattern in col.lower() for pattern in location_patterns)
        ]
        
    def _engineer_features(self, df):
        """Engineer domain-specific features for real estate."""
        df_eng = df.copy()
        
        # Property age (if year_built available)
        year_cols = [col for col in df.columns if 'year_built' in col.lower()]
        if year_cols:
            current_year = 2025  # Updated to current year
            year_col = year_cols[0]
            
            # Clean and convert year data to numeric
            year_data = pd.to_numeric(df_eng[year_col], errors='coerce')
            
            # Calculate age only for valid years (between 1800 and current year)
            valid_years = (year_data >= 1800) & (year_data <= current_year)
            df_eng[f'{year_col}_age'] = np.where(
                valid_years,
                current_year - year_data,
                np.nan
            )
            self.engineered_features.append(f'{year_col}_age')
        
        # Price per square foot (if both available)
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        sqft_cols = [col for col in df.columns if any(
            pattern in col.lower() for pattern in ['sqft', 'square_feet', 'area']
        )]
        
        if price_cols and sqft_cols:
            price_col = price_cols[0]
            sqft_col = sqft_cols[0]
            
            # Clean and convert to numeric
            price_data = pd.to_numeric(df_eng[price_col], errors='coerce')
            sqft_data = pd.to_numeric(df_eng[sqft_col], errors='coerce')
            
            # Calculate price per sqft, avoiding division by zero or invalid values
            df_eng[f'{price_col}_per_sqft'] = np.where(
                (price_data > 0) & (sqft_data > 0),
                price_data / sqft_data,
                np.nan
            )
            self.engineered_features.append(f'{price_col}_per_sqft')
        
        # Bedroom to bathroom ratio
        bed_cols = [col for col in df.columns if 'bedroom' in col.lower()]
        bath_cols = [col for col in df.columns if 'bathroom' in col.lower()]
        
        if bed_cols and bath_cols:
            bed_col = bed_cols[0]
            bath_col = bath_cols[0]
            df_eng[f'{bed_col}_to_{bath_col}_ratio'] = (
                df_eng[bed_col] / df_eng[bath_col].replace(0, np.nan)
            )
            self.engineered_features.append(f'{bed_col}_to_{bath_col}_ratio')
        
        # Log transformations for skewed numerical features
        for col in self.numerical_features:
            if col in df_eng.columns and df_eng[col].dtype.kind in 'if':
                # Apply log transformation if data is highly skewed
                if df_eng[col].min() > 0 and df_eng[col].skew() > 2:
                    df_eng[f'{col}_log'] = np.log1p(df_eng[col])
                    self.engineered_features.append(f'{col}_log')
        
        return df_eng
    
    def get_feature_names(self):
        """Get all feature names for ML models."""
        all_features = (
            self.numerical_features + 
            self.categorical_features + 
            self.engineered_features
        )
        return all_features


def create_training_dataset(df):
    """
    Create training dataset for ML models.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed property data
        
    Returns
    -------
    tuple
        (subject_properties, comparable_properties, labels)
    """
    # Separate by property role
    subjects = df[df['property_role'] == 'subject'].copy()
    selected_comps = df[df['property_role'] == 'selected_comparable'].copy()
    potential_comps = df[df['property_role'] == 'potential_comparable'].copy()
    
    print(f"Training data composition:")
    print(f"  Subject properties: {len(subjects)}")
    print(f"  Selected comparables: {len(selected_comps)}")
    print(f"  Potential comparables: {len(potential_comps)}")
    
    # Create training pairs
    training_pairs = []
    labels = []
    
    # For each appraisal, create subject-comparable pairs
    for appraisal_id in subjects['appraisal_index'].unique():
        subject = subjects[subjects['appraisal_index'] == appraisal_id]
        
        if len(subject) == 0:
            continue
            
        subject_row = subject.iloc[0]
        
        # Positive examples: selected comparables
        appraisal_selected = selected_comps[
            selected_comps['appraisal_index'] == appraisal_id
        ]
        
        for _, comp in appraisal_selected.iterrows():
            training_pairs.append({
                'subject': subject_row,
                'comparable': comp,
                'appraisal_id': appraisal_id
            })
            labels.append(1)  # Positive label
        
        # Negative examples: potential but not selected comparables
        appraisal_potential = potential_comps[
            potential_comps['appraisal_index'] == appraisal_id
        ]
        
        # Sample negative examples (limit to avoid imbalance)
        n_negatives = min(len(appraisal_potential), len(appraisal_selected) * 3)
        negative_samples = appraisal_potential.sample(n=n_negatives, random_state=42)
        
        for _, comp in negative_samples.iterrows():
            training_pairs.append({
                'subject': subject_row,
                'comparable': comp,
                'appraisal_id': appraisal_id
            })
            labels.append(0)  # Negative label
    
    print(f"Created {len(training_pairs)} training pairs")
    print(f"  Positive examples: {sum(labels)}")
    print(f"  Negative examples: {len(labels) - sum(labels)}")
    
    return training_pairs, labels


if __name__ == "__main__":
    # Example usage
    try:
        # Load the extracted data
        print("Loading extracted property data...")
        df = pd.read_csv('extracted_properties_corrected.csv')
        
        print(f"Loaded {len(df)} property records")
        print(f"Property roles: {df['property_role'].value_counts().to_dict()}")
        
        # Initialize and fit preprocessor
        preprocessor = PropertyDataPreprocessor(
            scaling_method='robust',
            imputation_strategy='knn'
        )
        
        # Fit and transform the data
        df_processed = preprocessor.fit_transform(df)
        
        # Create training dataset
        training_pairs, labels = create_training_dataset(df_processed)
        
        # Save processed data and training dataset
        df_processed.to_csv('data/processed/properties_preprocessed.csv', index=False)
        
        # Save training pairs
        import pickle
        with open('data/processed/training_pairs.pkl', 'wb') as f:
            pickle.dump((training_pairs, labels), f)
            
        # Save preprocessor
        with open('data/models/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
            
        print("\nProcessing complete!")
        print("Files saved:")
        print("  - data/processed/properties_preprocessed.csv")
        print("  - data/processed/training_pairs.pkl")
        print("  - data/models/preprocessor.pkl")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()