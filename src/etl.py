import pandas as pd
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataETL:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.clean_dir = self.data_dir / 'clean'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        for directory in [self.raw_dir, self.clean_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        self.svi_data = None
        self.cms_data = None
        self.kaggle_data = None
        self.crosswalk_data = None
        self.merged_data = None
        
    def load_real_svi_data(self):
        """Load real CDC SVI data from downloaded file"""
        svi_path = self.raw_dir / "CDC_SVI_2022.csv"
        if svi_path.exists():
            print("Loading CDC SVI 2022 data...")
            self.svi_data = pd.read_csv(svi_path, encoding='ISO-8859-1')
            
            # Select key columns (adjust based on actual SVI structure)
            svi_columns = ['FIPS', 'COUNTY', 'STATE', 'RPL_THEME1', 'RPL_THEME2', 
                          'RPL_THEME3', 'RPL_THEME4', 'RPL_TOTAL', 'EP_POV', 
                          'EP_UNEMP', 'EP_PCI', 'EP_NOHSDP', 'EP_AGE65', 'EP_AGE17',
                          'EP_DISABL', 'EP_SNGPNT', 'EP_MINRTY', 'EP_LIMENG',
                          'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ']
            
            # Only keep columns that exist in the data
            available_columns = [col for col in svi_columns if col in self.svi_data.columns]
            self.svi_data = self.svi_data[available_columns]
            
            print(f"✓ Loaded CDC SVI: {len(self.svi_data):,} counties")
            return True
        else:
            print("⚠ CDC SVI file not found. Please run data_downloader.py first.")
            return False
    
    def load_real_cms_data(self):
        """Load real CMS HRRP data from downloaded file"""
        cms_path = self.raw_dir / "CMS_HRRP_Readmissions.csv"
        if cms_path.exists():
            print("Loading CMS HRRP data...")
            self.cms_data = pd.read_csv(cms_path)
            
            # Standardize column names and select relevant columns
            column_mapping = {
                'Facility ID': 'hospital_id',
                'Facility Name': 'hospital_name',
                'ZIP Code': 'ZIP',
                'State': 'STATE',
                'Excess Readmission Ratio': 'excess_readmission_ratio',
                'Number of Discharges': 'number_of_discharges'
            }
            
            # Map existing columns
            for old_col, new_col in column_mapping.items():
                if old_col in self.cms_data.columns:
                    self.cms_data[new_col] = self.cms_data[old_col]
            
            # Create hospital_id if not present
            if 'hospital_id' not in self.cms_data.columns:
                self.cms_data['hospital_id'] = range(1, len(self.cms_data) + 1)
            
            print(f"✓ Loaded CMS HRRP: {len(self.cms_data):,} hospitals")
            return True
        else:
            print("⚠ CMS HRRP file not found. Please run data_downloader.py first.")
            return False
    
    def load_real_crosswalk_data(self):
        """Load real HUD ZIP to County crosswalk"""
        crosswalk_path = self.raw_dir / "ZIP_COUNTY_122022.csv"
        if crosswalk_path.exists():
            print("Loading HUD ZIP-County crosswalk...")
            self.crosswalk_data = pd.read_csv(crosswalk_path)
            
            # Standardize column names
            if 'zip' in self.crosswalk_data.columns:
                self.crosswalk_data['ZIP'] = self.crosswalk_data['zip'].astype(str).str.zfill(5)
            if 'county' in self.crosswalk_data.columns:
                self.crosswalk_data['FIPS'] = self.crosswalk_data['county'].astype(str).str.zfill(5)
            
            # Ensure we have required columns
            if 'ZIP' not in self.crosswalk_data.columns or 'FIPS' not in self.crosswalk_data.columns:
                print("⚠ Crosswalk missing required columns")
                return False
            
            print(f"✓ Loaded crosswalk: {len(self.crosswalk_data):,} mappings")
            return True
        else:
            print("⚠ Crosswalk file not found. Please run data_downloader.py first.")
            return False
    
    def load_real_patient_data(self):
        """Load real patient data (Kaggle diabetes or synthetic)"""
        kaggle_path = self.raw_dir / "diabetes.csv"
        synthetic_path = self.raw_dir / "synthetic_patient_data.csv"
        
        if kaggle_path.exists():
            print("Loading Kaggle diabetes data...")
            self.kaggle_data = pd.read_csv(kaggle_path)
            
            # Map Kaggle columns to our expected format
            column_mapping = {
                'encounter_id': 'patient_id',
                'patient_nbr': 'hospital_id',
                'race': 'race',
                'gender': 'gender',
                'age': 'age',
                'time_in_hospital': 'length_of_stay',
                'num_lab_procedures': 'num_lab_procedures',
                'num_procedures': 'num_procedures',
                'num_medications': 'num_medications',
                'number_outpatient': 'number_outpatient',
                'number_emergency': 'number_emergency',
                'number_inpatient': 'number_inpatient',
                'number_diagnoses': 'number_diagnoses',
                'max_glu_serum': 'max_glu_serum',
                'A1Cresult': 'A1Cresult',
                'change': 'change',
                'diabetesMed': 'diabetesMed',
                'readmitted': 'readmitted'
            }
            
            # Apply mapping for existing columns
            for old_col, new_col in column_mapping.items():
                if old_col in self.kaggle_data.columns:
                    self.kaggle_data[new_col] = self.kaggle_data[old_col]
            
            # Create 30-day readmission target
            self.kaggle_data['readmitted_30'] = (self.kaggle_data['readmitted'] == '<30').astype(int)
            
            # Add ZIP codes for geographic mapping (synthetic for now)
            np.random.seed(42)
            self.kaggle_data['ZIP'] = [f'{np.random.randint(10000, 99999)}' for _ in range(len(self.kaggle_data))]
            
            print(f"✓ Loaded Kaggle diabetes data: {len(self.kaggle_data):,} patients")
            return True
            
        elif synthetic_path.exists():
            print("Loading synthetic patient data...")
            self.kaggle_data = pd.read_csv(synthetic_path)
            print(f"✓ Loaded synthetic patient data: {len(self.kaggle_data):,} patients")
            return True
        else:
            print("⚠ No patient data found. Generating synthetic data...")
            self.kaggle_data = self.generate_synthetic_patient_data()
            return True
    
    def generate_synthetic_svi_data(self):
        """Generate synthetic CDC SVI data for demonstration"""
        np.random.seed(42)
        n_counties = 3200
        
        synthetic_svi = pd.DataFrame({
            'FIPS': [f'{str(i).zfill(5)}' for i in range(1000, 1000 + n_counties)],
            'COUNTY': [f'County {i}' for i in range(n_counties)],
            'STATE': np.random.choice(['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'], n_counties),
            'RPL_THEME1': np.random.uniform(0, 1, n_counties),  # Socioeconomic
            'RPL_THEME2': np.random.uniform(0, 1, n_counties),  # Household Composition
            'RPL_THEME3': np.random.uniform(0, 1, n_counties),  # Minority Status
            'RPL_THEME4': np.random.uniform(0, 1, n_counties),  # Housing/Transportation
            'RPL_TOTAL': np.random.uniform(0, 1, n_counties),   # Overall SVI
            'EP_POV': np.random.uniform(0, 50, n_counties),     # Poverty %
            'EP_UNEMP': np.random.uniform(0, 20, n_counties),   # Unemployment %
            'EP_PCI': np.random.uniform(20000, 80000, n_counties), # Per capita income
            'EP_NOHSDP': np.random.uniform(0, 40, n_counties),  # No high school diploma %
            'EP_AGE65': np.random.uniform(0, 30, n_counties),   # Age 65+ %
            'EP_AGE17': np.random.uniform(0, 30, n_counties),   # Age <17 %
            'EP_DISABL': np.random.uniform(0, 20, n_counties),  # Disability %
            'EP_SNGPNT': np.random.uniform(0, 20, n_counties),  # Single parent %
            'EP_MINRTY': np.random.uniform(0, 100, n_counties), # Minority %
            'EP_LIMENG': np.random.uniform(0, 30, n_counties),  # Limited English %
            'EP_MUNIT': np.random.uniform(0, 40, n_counties),   # Multi-unit housing %
            'EP_MOBILE': np.random.uniform(0, 20, n_counties),  # Mobile homes %
            'EP_CROWD': np.random.uniform(0, 15, n_counties),   # Crowding %
            'EP_NOVEH': np.random.uniform(0, 25, n_counties),   # No vehicle %
            'EP_GROUPQ': np.random.uniform(0, 10, n_counties)   # Group quarters %
        })
        
        return synthetic_svi
    
    def generate_synthetic_cms_data(self):
        """Generate synthetic CMS HRRP data"""
        np.random.seed(42)
        n_hospitals = 5000
        
        synthetic_cms = pd.DataFrame({
            'hospital_id': range(1, n_hospitals + 1),
            'hospital_name': [f'Hospital {i}' for i in range(1, n_hospitals + 1)],
            'ZIP': [f'{np.random.randint(10000, 99999)}' for _ in range(n_hospitals)],
            'STATE': np.random.choice(['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'], n_hospitals),
            'readmission_rate': np.random.uniform(10, 25, n_hospitals),
            'predicted_rate': np.random.uniform(12, 22, n_hospitals),
            'excess_readmission_ratio': np.random.uniform(0.8, 1.2, n_hospitals),
            'number_of_discharges': np.random.randint(100, 5000, n_hospitals),
            'measure_name': np.random.choice(['READM-30-HIP-KNEE', 'READM-30-COPD', 'READM-30-HF', 'READM-30-PN', 'READM-30-AMI'], n_hospitals)
        })
        
        return synthetic_cms
    
    def generate_synthetic_kaggle_data(self):
        """Generate synthetic patient-level EHR data based on Kaggle diabetes dataset"""
        np.random.seed(42)
        n_patients = 100000
        
        synthetic_patients = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'hospital_id': np.random.randint(1, 5001, n_patients),
            'ZIP': [f'{np.random.randint(10000, 99999)}' for _ in range(n_patients)],
            'age': np.random.randint(0, 100, n_patients),
            'gender': np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52]),
            'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'], 
                                   n_patients, p=[0.6, 0.18, 0.15, 0.05, 0.02]),
            'length_of_stay': np.random.randint(1, 30, n_patients),
            'num_lab_procedures': np.random.randint(1, 50, n_patients),
            'num_procedures': np.random.randint(0, 10, n_patients),
            'num_medications': np.random.randint(1, 30, n_patients),
            'number_outpatient': np.random.randint(0, 20, n_patients),
            'number_emergency': np.random.randint(0, 15, n_patients),
            'number_inpatient': np.random.randint(0, 10, n_patients),
            'number_diagnoses': np.random.randint(1, 15, n_patients),
            'max_glu_serum': np.random.choice(['None', 'Norm', '>200', '>300'], n_patients, p=[0.8, 0.1, 0.05, 0.05]),
            'A1Cresult': np.random.choice(['None', 'Norm', '>7', '>8'], n_patients, p=[0.7, 0.2, 0.05, 0.05]),
            'change': np.random.choice(['No', 'Yes'], n_patients),
            'diabetesMed': np.random.choice(['No', 'Yes'], n_patients),
            'readmitted': np.random.choice(['NO', '<30', '>30'], n_patients, p=[0.85, 0.1, 0.05])
        })
        
        # Create target variable for 30-day readmission
        synthetic_patients['readmitted_30'] = (synthetic_patients['readmitted'] == '<30').astype(int)
        
        return synthetic_patients
    
    def create_crosswalk(self):
        """Create ZIP to FIPS crosswalk"""
        np.random.seed(42)
        
        # Generate synthetic crosswalk data
        zip_codes = [f'{np.random.randint(10000, 99999)}' for _ in range(10000)]
        crosswalk_data = []
        
        for zip_code in zip_codes:
            crosswalk_data.append({
                'ZIP': zip_code,
                'FIPS': f'{np.random.randint(1, 57):02d}{np.random.randint(1, 999):03d}',
                'COUNTY': f'County {np.random.randint(1, 100)}',
                'STATE': np.random.choice(['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'])
            })
        
        return pd.DataFrame(crosswalk_data).drop_duplicates('ZIP')
    
    def load_all_real_data(self):
        """Load all real datasets"""
        print("Loading REAL datasets...")
        
        success_count = 0
        total_datasets = 4
        
        # Load SVI data
        if self.load_real_svi_data():
            success_count += 1
        
        # Load CMS data
        if self.load_real_cms_data():
            success_count += 1
        
        # Load crosswalk data
        if self.load_real_crosswalk_data():
            success_count += 1
        
        # Load patient data
        if self.load_real_patient_data():
            success_count += 1
        
        print(f"✓ Successfully loaded {success_count}/{total_datasets} real datasets")
        
        if success_count == 0:
            print("❌ No real datasets loaded. Falling back to synthetic data.")
            return self.load_all_synthetic_data()
        
        return True
    
    def load_all_synthetic_data(self):
        """Load/generate all synthetic datasets"""
        print("Loading/GENERATING synthetic datasets...")
        
        # Generate synthetic data
        self.svi_data = self.generate_synthetic_svi_data()
        self.cms_data = self.generate_synthetic_cms_data()
        self.kaggle_data = self.generate_synthetic_kaggle_data()
        self.crosswalk_data = self.create_crosswalk()
        
        print(f"SVI Data: {self.svi_data.shape}")
        print(f"CMS Data: {self.cms_data.shape}")
        print(f"Kaggle Data: {self.kaggle_data.shape}")
        print(f"Crosswalk Data: {self.crosswalk_data.shape}")
        
        return True
    
    def merge_real_datasets(self):
        """Merge real datasets using geographic keys"""
        print("Merging REAL datasets...")
        
        try:
            # Merge patient data with crosswalk
            patient_with_fips = self.kaggle_data.merge(
                self.crosswalk_data, on='ZIP', how='left'
            )
            
            # Merge with SVI data
            patient_with_svi = patient_with_fips.merge(
                self.svi_data, on='FIPS', how='left'
            )
            
            # Merge with hospital data
            self.merged_data = patient_with_svi.merge(
                self.cms_data, on='hospital_id', how='left', suffixes=('_patient', '_hospital')
            )
            
            print(f"✓ Merged dataset: {self.merged_data.shape}")
            
            # Fill missing values from real data merge
            self._handle_missing_values()
            
            # Save cleaned data
            self.merged_data.to_parquet(self.clean_dir / 'merged_data.parquet', index=False)
            
            return self.merged_data
            
        except Exception as e:
            print(f"❌ Real data merge failed: {e}")
            print("Falling back to synthetic data merge...")
            return self.merge_synthetic_datasets()
    
    def merge_synthetic_datasets(self):
        """Merge synthetic datasets"""
        print("Merging SYNTHETIC datasets...")
        
        # Merge patient data with crosswalk
        patient_with_fips = self.kaggle_data.merge(
            self.crosswalk_data, on='ZIP', how='left'
        )
        
        # Merge with SVI data
        patient_with_svi = patient_with_fips.merge(
            self.svi_data, on='FIPS', how='left'
        )
        
        # Merge with hospital data
        self.merged_data = patient_with_svi.merge(
            self.cms_data, on='hospital_id', how='left', suffixes=('_patient', '_hospital')
        )
        
        print(f"✓ Merged dataset: {self.merged_data.shape}")
        
        # Save cleaned data
        self.merged_data.to_parquet(self.clean_dir / 'merged_data.parquet', index=False)
        
        return self.merged_data
    
    def _handle_missing_values(self):
        """Handle missing values in merged dataset"""
        print("Handling missing values...")
        
        # Fill missing SVI data with synthetic values for prototyping
        svi_columns = ['RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4', 'RPL_TOTAL']
        
        for col in svi_columns:
            if col in self.merged_data.columns and self.merged_data[col].isnull().any():
                self.merged_data[col] = self.merged_data[col].fillna(
                    np.random.uniform(0, 1, self.merged_data[col].isnull().sum())
                )
        
        print(f"✓ Missing values handled")
    
    def run_real_data_pipeline(self):
        """Run complete ETL pipeline with REAL data"""
        print("Starting REAL DATA ETL pipeline...")
        
        # Load real data
        if not self.load_all_real_data():
            print("❌ Real data loading failed. Using synthetic data.")
            return self.run_pipeline()
        
        # Merge datasets
        merged_data = self.merge_real_datasets()
        
        print("✓ REAL DATA ETL pipeline completed successfully!")
        return merged_data
    
    def run_pipeline(self):
        """Run complete ETL pipeline with synthetic data"""
        print("Starting SYNTHETIC DATA ETL pipeline...")
        
        self.load_all_synthetic_data()
        merged_data = self.merge_synthetic_datasets()
        
        print("✓ SYNTHETIC DATA ETL pipeline completed successfully!")
        return merged_data

# Usage
if __name__ == "__main__":
    etl = DataETL()
    
    # Ask user for data preference
    print("Choose data source:")
    print("1. REAL DATA")
    print("2. SYNTHETIC DATA")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        data = etl.run_real_data_pipeline()
    else:
        data = etl.run_pipeline()
    
    print(f"Final dataset: {data.shape}")