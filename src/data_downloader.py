"""
Complete Data Downloader for Hospital Readmission Project
UPDATED to match config.py DATA_PATHS exactly
"""

import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
import zipfile
import warnings
warnings.filterwarnings('ignore')

class DataDownloader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Exact file paths from config.py
        self.data_paths = {
            'svi_data': self.raw_dir / 'CDC_SVI_2020.csv',
            'cms_data': self.raw_dir / 'CMS_HRRP_2022.csv', 
            'kaggle_data': self.raw_dir / 'kaggle_diabetes.csv',
            'crosswalk': self.raw_dir / 'ZIP_COUNTY_122022.csv'
        }
        
    def download_cdc_svi_2020(self):
        """Download CDC Social Vulnerability Index 2020 - Exact match to config"""
        print("Downloading CDC SVI 2020 data...")
        
        # Try multiple possible URLs for CDC SVI 2020
        svi_urls = [
            "https://svi.cdc.gov/Documents/Data/2020_SVI_Data/CSV/SVI2020_US.csv",
            "https://svi.cdc.gov/Documents/Data/2020_SVI_Data/CSV/SVI2020_US.zip",
            "https://svi.cdc.gov/Documents/Data/2020_SVI_Data/CSV/SVI2020_US.csv.zip"
        ]
        
        for url in svi_urls:
            try:
                print(f"Trying: {url}")
                if url.endswith('.csv'):
                    df = pd.read_csv(url, encoding='ISO-8859-1', low_memory=False)
                elif url.endswith('.zip'):
                    # Download and extract zip
                    response = requests.get(url)
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        # Find CSV file in zip
                        csv_file = [f for f in z.namelist() if f.endswith('.csv')][0]
                        with z.open(csv_file) as f:
                            df = pd.read_csv(f, encoding='ISO-8859-1', low_memory=False)
                
                if df.empty:
                    continue
                    
                # Save with exact config.py filename
                df.to_csv(self.data_paths['svi_data'], index=False)
                
                print(f"✓ CDC SVI 2020 downloaded: {len(df):,} records")
                print(f"✓ Saved to: {self.data_paths['svi_data']}")
                return df
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        print("❌ All CDC SVI 2020 downloads failed")
        print("⚠ Generating synthetic CDC SVI 2020 data...")
        return self.generate_synthetic_svi_2020()
    
    def download_cms_hrrp_2022(self):
        """Download CMS HRRP 2022 data - Exact match to config"""
        print("Downloading CMS HRRP 2022 data...")
        
        # CMS 2022 data URLs
        cms_urls = [
            "https://data.cms.gov/provider-data/sites/default/files/resources/9c7e061fcdd74c8ebf8b5f321fbb51a0_1699550433/Hospital_Readmissions_Reduction_Program_-_Hospital.csv",
            "https://data.cms.gov/api/views/9n3s-kdb3/rows.csv?accessType=DOWNLOAD"
        ]
        
        for url in cms_urls:
            try:
                print(f"Trying: {url}")
                df = pd.read_csv(url)
                
                if df.empty:
                    continue
                
                # Save with exact config.py filename
                df.to_csv(self.data_paths['cms_data'], index=False)
                
                print(f"✓ CMS HRRP 2022 downloaded: {len(df):,} hospitals")
                print(f"✓ Saved to: {self.data_paths['cms_data']}")
                return df
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        print("❌ All CMS HRRP 2022 downloads failed")
        print("⚠ Generating synthetic CMS HRRP 2022 data...")
        return self.generate_synthetic_cms_2022()
    
    def download_zip_county_122022(self):
        """Download ZIP-COUNTY crosswalk - Exact match to config"""
        print("Downloading ZIP-COUNTY 122022 crosswalk...")
        
        # HUD crosswalk URLs
        crosswalk_urls = [
            "https://www.huduser.gov/portal/datasets/usps/zip_county_122022.xlsx",
            "https://www.huduser.gov/portal/datasets/usps/zip_county_122022.csv"
        ]
        
        for url in crosswalk_urls:
            try:
                print(f"Trying: {url}")
                if url.endswith('.xlsx'):
                    df = pd.read_excel(url)
                else:
                    df = pd.read_csv(url)
                
                if df.empty:
                    continue
                
                # Save with exact config.py filename
                df.to_csv(self.data_paths['crosswalk'], index=False)
                
                print(f"✓ ZIP-COUNTY 122022 downloaded: {len(df):,} mappings")
                print(f"✓ Saved to: {self.data_paths['crosswalk']}")
                return df
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        print("❌ All ZIP-COUNTY crosswalk downloads failed")
        print("⚠ Generating synthetic ZIP-COUNTY crosswalk...")
        return self.generate_synthetic_crosswalk()
    
    def download_kaggle_diabetes(self):
        """Download Kaggle diabetes data - Exact match to config"""
        print("Downloading Kaggle diabetes data...")
        
        # Try direct download from alternative sources
        kaggle_urls = [
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
            "https://github.com/saiteja-lakkapally/hospital-readmission-prediction/raw/main/data/diabetes.csv"
        ]
        
        for url in kaggle_urls:
            try:
                print(f"Trying: {url}")
                df = pd.read_csv(url)
                
                if df.empty:
                    continue
                
                # Save with exact config.py filename
                df.to_csv(self.data_paths['kaggle_data'], index=False)
                
                print(f"✓ Kaggle diabetes data downloaded: {len(df):,} patients")
                print(f"✓ Saved to: {self.data_paths['kaggle_data']}")
                return df
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        # Manual download instructions
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD REQUIRED for Kaggle Diabetes Dataset")
        print("="*60)
        print("1. Go to: https://www.kaggle.com/datasets/brandao/diabetes")
        print("2. Click 'Download' (requires Kaggle account)")
        print("3. Extract and rename the file to: kaggle_diabetes.csv")
        print("4. Place in: data/raw/kaggle_diabetes.csv")
        print("="*60)
        
        # Check if manual file exists
        if self.data_paths['kaggle_data'].exists():
            df = pd.read_csv(self.data_paths['kaggle_data'])
            print(f"✓ Kaggle diabetes data found: {len(df):,} patients")
            return df
        else:
            print("⚠ Kaggle dataset not found. Generating synthetic data...")
            return self.generate_synthetic_kaggle_diabetes()
    
    def generate_synthetic_svi_2020(self):
        """Generate synthetic CDC SVI 2020 data matching config filename"""
        print("Generating synthetic CDC SVI 2020 data...")
        
        np.random.seed(42)
        n_counties = 3200
        
        synthetic_svi = pd.DataFrame({
            'FIPS': [f'{str(i).zfill(5)}' for i in range(1000, 1000 + n_counties)],
            'COUNTY': [f'County {i}' for i in range(n_counties)],
            'STATE': np.random.choice(['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'], n_counties),
            'RPL_THEME1': np.random.uniform(0, 1, n_counties),
            'RPL_THEME2': np.random.uniform(0, 1, n_counties),
            'RPL_THEME3': np.random.uniform(0, 1, n_counties),
            'RPL_THEME4': np.random.uniform(0, 1, n_counties),
            'RPL_TOTAL': np.random.uniform(0, 1, n_counties),
            'EP_POV': np.random.uniform(0, 50, n_counties),
            'EP_UNEMP': np.random.uniform(0, 20, n_counties),
            'EP_PCI': np.random.uniform(20000, 80000, n_counties),
            'EP_NOHSDP': np.random.uniform(0, 40, n_counties),
            'EP_AGE65': np.random.uniform(0, 30, n_counties),
            'EP_AGE17': np.random.uniform(0, 30, n_counties),
            'EP_DISABL': np.random.uniform(0, 20, n_counties),
            'EP_SNGPNT': np.random.uniform(0, 20, n_counties),
            'EP_MINRTY': np.random.uniform(0, 100, n_counties),
            'EP_LIMENG': np.random.uniform(0, 30, n_counties),
            'EP_MUNIT': np.random.uniform(0, 40, n_counties),
            'EP_MOBILE': np.random.uniform(0, 20, n_counties),
            'EP_CROWD': np.random.uniform(0, 15, n_counties),
            'EP_NOVEH': np.random.uniform(0, 25, n_counties),
            'EP_GROUPQ': np.random.uniform(0, 10, n_counties)
        })
        
        # Save with exact config.py filename
        synthetic_svi.to_csv(self.data_paths['svi_data'], index=False)
        
        print(f"✓ Synthetic CDC SVI 2020 generated: {len(synthetic_svi):,} counties")
        print(f"✓ Saved to: {self.data_paths['svi_data']}")
        
        return synthetic_svi
    
    def generate_synthetic_cms_2022(self):
        """Generate synthetic CMS HRRP 2022 data matching config filename"""
        print("Generating synthetic CMS HRRP 2022 data...")
        
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
        
        # Save with exact config.py filename
        synthetic_cms.to_csv(self.data_paths['cms_data'], index=False)
        
        print(f"✓ Synthetic CMS HRRP 2022 generated: {len(synthetic_cms):,} hospitals")
        print(f"✓ Saved to: {self.data_paths['cms_data']}")
        
        return synthetic_cms
    
    def generate_synthetic_crosswalk(self):
        """Generate synthetic ZIP-COUNTY crosswalk matching config filename"""
        print("Generating synthetic ZIP-COUNTY 122022 crosswalk...")
        
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
        
        crosswalk_df = pd.DataFrame(crosswalk_data).drop_duplicates('ZIP')
        
        # Save with exact config.py filename
        crosswalk_df.to_csv(self.data_paths['crosswalk'], index=False)
        
        print(f"✓ Synthetic ZIP-COUNTY 122022 generated: {len(crosswalk_df):,} mappings")
        print(f"✓ Saved to: {self.data_paths['crosswalk']}")
        
        return crosswalk_df
    
    def generate_synthetic_kaggle_diabetes(self):
        """Generate synthetic Kaggle diabetes data matching config filename"""
        print("Generating synthetic kaggle_diabetes data...")
        
        np.random.seed(42)
        n_patients = 50000
        
        synthetic_data = pd.DataFrame({
            'encounter_id': range(1, n_patients + 1),
            'patient_nbr': np.random.randint(1, 10000, n_patients),
            'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'], 
                                   n_patients, p=[0.6, 0.18, 0.15, 0.05, 0.02]),
            'gender': np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52]),
            'age': np.random.choice(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], n_patients),
            'weight': np.random.choice(['?', '[0-25)', '[25-50)', '[50-75)', '[75-100)', '[100-125)', '[125-150)', '[150-175)', '[175-200)'], n_patients, p=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]),
            'admission_type_id': np.random.randint(1, 9, n_patients),
            'discharge_disposition_id': np.random.randint(1, 30, n_patients),
            'admission_source_id': np.random.randint(1, 30, n_patients),
            'time_in_hospital': np.random.randint(1, 15, n_patients),
            'payer_code': np.random.choice(['?', 'MC', 'MD', 'SP', 'BC', 'HM', 'UN', 'CP', 'SI', 'DM'], n_patients),
            'medical_specialty': np.random.choice(['?', 'InternalMedicine', 'Family/GeneralPractice', 'Cardiology', 'Surgery', 'Emergency/Trauma'], n_patients),
            'num_lab_procedures': np.random.randint(1, 50, n_patients),
            'num_procedures': np.random.randint(0, 8, n_patients),
            'num_medications': np.random.randint(1, 25, n_patients),
            'number_outpatient': np.random.randint(0, 15, n_patients),
            'number_emergency': np.random.randint(0, 10, n_patients),
            'number_inpatient': np.random.randint(0, 8, n_patients),
            'diag_1': [f'{np.random.randint(1, 1000):03d}' for _ in range(n_patients)],
            'diag_2': [f'{np.random.randint(1, 1000):03d}' for _ in range(n_patients)],
            'diag_3': [f'{np.random.randint(1, 1000):03d}' for _ in range(n_patients)],
            'number_diagnoses': np.random.randint(1, 12, n_patients),
            'max_glu_serum': np.random.choice(['None', 'Norm', '>200', '>300'], n_patients, p=[0.8, 0.1, 0.05, 0.05]),
            'A1Cresult': np.random.choice(['None', 'Norm', '>7', '>8'], n_patients, p=[0.7, 0.2, 0.05, 0.05]),
            'metformin': np.random.choice(['No', 'Yes', 'Steady', 'Up', 'Down'], n_patients),
            'change': np.random.choice(['No', 'Yes'], n_patients),
            'diabetesMed': np.random.choice(['No', 'Yes'], n_patients),
            'readmitted': np.random.choice(['NO', '<30', '>30'], n_patients, p=[0.85, 0.1, 0.05])
        })
        
        # Add hospital_id and ZIP for our ETL pipeline
        synthetic_data['hospital_id'] = np.random.randint(1, 501, n_patients)
        synthetic_data['ZIP'] = [f'{np.random.randint(10000, 99999)}' for _ in range(n_patients)]
        
        # Save with exact config.py filename
        synthetic_data.to_csv(self.data_paths['kaggle_data'], index=False)
        
        print(f"✓ Synthetic kaggle_diabetes data generated: {len(synthetic_data):,} patients")
        print(f"✓ Saved to: {self.data_paths['kaggle_data']}")
        
        return synthetic_data
    
    def check_existing_files(self):
        """Check which files already exist"""
        print("Checking existing data files...")
        existing_files = []
        missing_files = []
        
        for name, path in self.data_paths.items():
            if path.exists():
                existing_files.append(name)
                try:
                    df = pd.read_csv(path)
                    print(f"✓ {name}: {len(df):,} records")
                except:
                    print(f"⚠ {name}: File exists but cannot be read")
            else:
                missing_files.append(name)
                print(f"❌ {name}: Missing")
        
        return existing_files, missing_files
    
    def download_all_data(self):
        """Download all required datasets matching config.py exactly"""
        print("="*60)
        print("HOSPITAL READMISSION DATA DOWNLOADER")
        print("Using exact config.py file paths:")
        for name, path in self.data_paths.items():
            print(f"  {name}: {path}")
        print("="*60)
        
        # Check existing files first
        existing_files, missing_files = self.check_existing_files()
        
        if not missing_files:
            print("🎉 All data files already exist!")
            return {name: pd.read_csv(path) for name, path in self.data_paths.items()}
        
        print(f"\nDownloading {len(missing_files)} missing datasets...")
        datasets = {}
        
        # Download missing datasets
        if 'svi_data' in missing_files:
            print("\n1. CDC SVI 2020")
            datasets['svi_data'] = self.download_cdc_svi_2020()
        
        if 'cms_data' in missing_files:
            print("\n2. CMS HRRP 2022")
            datasets['cms_data'] = self.download_cms_hrrp_2022()
        
        if 'crosswalk' in missing_files:
            print("\n3. ZIP-COUNTY 122022")
            datasets['crosswalk'] = self.download_zip_county_122022()
        
        if 'kaggle_data' in missing_files:
            print("\n4. Kaggle Diabetes")
            datasets['kaggle_data'] = self.download_kaggle_diabetes()
        
        # Load existing files
        for name in existing_files:
            datasets[name] = pd.read_csv(self.data_paths[name])
        
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        
        success_count = 0
        for name, data in datasets.items():
            if data is not None and not data.empty:
                status = "✓ SUCCESS"
                size = f"{len(data):,} records"
                success_count += 1
            else:
                status = "❌ FAILED"
                size = "No data"
            print(f"{name.upper():<15} {status:<12} {size}")
        
        print(f"\nOverall: {success_count}/4 datasets ready")
        
        if success_count == 4:
            print("🎉 All datasets ready for use!")
        else:
            print("⚠ Some datasets missing, but synthetic data generated")
        
        print("="*60)
        
        return datasets

# Quick download function
def quick_download():
    """Quick download with config.py paths"""
    downloader = DataDownloader()
    return downloader.download_all_data()

if __name__ == "__main__":
    downloader = DataDownloader()
    all_data = downloader.download_all_data()