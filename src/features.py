import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = []
        self.numeric_features = []
        self.categorical_features = []
        
    def create_medical_features(self, df):
        """Create medical composite features"""
        print("Creating medical features...")
        
        # Charlson Comorbidity Index approximation
        df['comorbidity_index'] = (
            (df['number_diagnoses'] > 5).astype(int) +
            (df['num_medications'] > 10).astype(int) +
            (df['number_inpatient'] > 0).astype(int) +
            (df['age'] > 65).astype(int)
        )
        
        # Prior admissions composite
        df['prior_admissions'] = df['number_inpatient'] + df['number_emergency'] * 0.5
        
        # Healthcare utilization score
        df['utilization_score'] = (
            df['number_outpatient'] * 0.3 + 
            df['number_emergency'] * 0.5 + 
            df['number_inpatient'] * 0.7
        )
        
        # Procedure intensity
        df['procedure_intensity'] = df['num_procedures'] * df['num_lab_procedures'] / 10
        
        # Length of stay categories
        df['los_category'] = pd.cut(
            df['length_of_stay'], 
            bins=[0, 3, 7, 14, float('inf')],
            labels=['Short', 'Medium', 'Long', 'Very Long']
        )
        
        return df
    
    def create_sdoh_features(self, df):
        """Create social determinants of health features"""
        print("Creating SDOH features...")
        
        # Overall vulnerability score (weighted average of SVI themes)
        df['overall_vulnerability'] = (
            df['RPL_THEME1'] * 0.3 +  # Socioeconomic
            df['RPL_THEME2'] * 0.2 +  # Household composition
            df['RPL_THEME3'] * 0.25 + # Minority status
            df['RPL_THEME4'] * 0.25   # Housing/transportation
        )
        
        # Economic hardship composite
        df['economic_hardship'] = (
            df['EP_POV'] * 0.4 +
            df['EP_UNEMP'] * 0.3 +
            df['EP_NOHSDP'] * 0.3
        )
        
        # Healthcare access barrier
        df['healthcare_access_barrier'] = (
            (df['EP_NOVEH'] > 10).astype(int) +
            (df['EP_POV'] > 20).astype(int) +
            (df['EP_LIMENG'] > 5).astype(int)
        )
        
        # Social isolation risk
        df['social_isolation_risk'] = (
            (df['EP_AGE65'] > 15).astype(int) +
            (df['EP_SNGPNT'] > 10).astype(int) +
            (df['EP_DISABL'] > 10).astype(int)
        )
        
        # Categorize vulnerability levels
        df['vulnerability_level'] = pd.cut(
            df['overall_vulnerability'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        return df
    
    def create_demographic_features(self, df):
        """Create demographic features"""
        print("Creating demographic features...")
        
        # Age groups
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 18, 35, 50, 65, 75, float('inf')],
            labels=['0-18', '19-35', '36-50', '51-65', '66-75', '75+']
        )
        
        # Risk age flag
        df['elderly'] = (df['age'] >= 65).astype(int)
        df['young_adult'] = ((df['age'] >= 18) & (df['age'] <= 35)).astype(int)
        
        return df
    
    def build_preprocessor(self):
        """Build preprocessing pipeline"""
        print("Building preprocessing pipeline...")
        
        # Define feature groups
        self.numeric_features = [
            'length_of_stay', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses', 'prior_admissions',
            'comorbidity_index', 'utilization_score', 'procedure_intensity',
            'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4',
            'RPL_TOTAL', 'EP_POV', 'EP_UNEMP', 'EP_PCI', 'EP_NOHSDP',
            'overall_vulnerability', 'economic_hardship', 
            'healthcare_access_barrier', 'social_isolation_risk'
        ]
        
        self.categorical_features = [
            'gender', 'race', 'age_group', 'los_category', 'vulnerability_level',
            'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
        ]
        
        # Numeric pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline  
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        return self.preprocessor
    
    def prepare_features(self, df, target_col='readmitted_30'):
        """Prepare features for modeling"""
        print("Preparing features for modeling...")
        
        # Create all feature types
        df = self.create_medical_features(df)
        df = self.create_sdoh_features(df) 
        df = self.create_demographic_features(df)
        
        # Handle missing values in target
        df = df.dropna(subset=[target_col])
        
        # Separate features and target
        X = df.copy()
        y = X[target_col]
        X = X.drop(columns=[target_col, 'readmitted', 'patient_id', 'hospital_id', 
                           'ZIP', 'FIPS', 'COUNTY_patient', 'STATE_patient', 
                           'COUNTY', 'STATE', 'hospital_name', 'measure_name'])
        
        # Build preprocessor if not exists
        if self.preprocessor is None:
            self.build_preprocessor()
        
        # Fit and transform features
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        feature_names = self.numeric_features.copy()
        categorical_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_features = categorical_encoder.get_feature_names_out(self.categorical_features)
        feature_names.extend(categorical_features)
        
        self.feature_names = feature_names
        
        # Create final DataFrame
        X_final = pd.DataFrame(X_processed, columns=feature_names)
        
        print(f"Final feature matrix: {X_final.shape}")
        print(f"Target distribution: {y.value_counts(normalize=True)}")
        
        return X_final, y, feature_names
    
    def handle_imbalance(self, X, y, method='smote'):
        """Handle class imbalance"""
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN
        
        print(f"Handling class imbalance using {method}...")
        print(f"Original class distribution: {y.value_counts().to_dict()}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'smoteenn':
            sampler = SMOTEENN(random_state=42)
        else:
            print("Using original data (no sampling)")
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled

# Usage example
if __name__ == "__main__":
    from etl import DataETL
    
    # Run ETL and feature engineering
    etl = DataETL()
    data = etl.run_pipeline()
    
    feature_engineer = FeatureEngineer()
    X, y, feature_names = feature_engineer.prepare_features(data)
    
    print("Feature engineering completed!")