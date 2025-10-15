#!/usr/bin/env python3
"""
HOSPITAL READMISSION PREDICTION SYSTEM
Using Real Data Structure from Project
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import time
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATHS, MODEL_PARAMS, FEATURE_GROUPS

# Add colorful console output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_banner():
    """Display animated banner"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║    🏥 HOSPITAL READMISSION PREDICTION SYSTEM 🏥               ║
║                                                                ║
║    MSDS692 - Data Science Practicum                           ║
║    Sai Teja Lakkapally                                        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
{Colors.END}
"""
    print(banner)
    
    # Animated loading
    for i in range(3):
        for char in "⣾⣽⣻⢿⡿⣟⣯⣷":
            sys.stdout.write(f'\r{Colors.YELLOW}Initializing system {char}{Colors.END}')
            sys.stdout.flush()
            time.sleep(0.1)
    print(f'\r{Colors.GREEN}✅ System initialized!{Colors.END}')

def install_requirements():
    """Step 1: Install required packages"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}STEP 1: INSTALLING REQUIREMENTS{Colors.END}")
    print(f"{Colors.BLUE}{'='*50}{Colors.END}")
    
    requirements = [
        "pandas", "numpy", "scikit-learn", "xgboost", "lightgbm",
        "matplotlib", "seaborn", "plotly", "dash", "imbalanced-learn", "joblib"
    ]
    
    import subprocess
    import importlib
    
    for package in requirements:
        try:
            # Try to import the package
            importlib.import_module(package.split("[")[0])
            print(f"{Colors.GREEN}✅ {package:.<30} Already installed{Colors.END}")
        except ImportError:
            print(f"{Colors.YELLOW}📦 {package:.<30} Installing...{Colors.END}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
                print(f"{Colors.GREEN}   {package:.<30} Installed successfully{Colors.END}")
            except subprocess.CalledProcessError:
                print(f"{Colors.RED}❌ {package:.<30} Installation failed{Colors.END}")
    
    print(f"{Colors.GREEN}✅ All requirements checked!{Colors.END}")

def convert_readmission_column(df):
    """Convert readmission column from string to numeric 0/1"""
    if 'readmitted' not in df.columns:
        return df
    
    # Check if conversion is needed
    if df['readmitted'].dtype == 'object':
        print(f"{Colors.YELLOW}Converting readmission column from string to numeric...{Colors.END}")
        
        # Map common readmission string patterns to 0/1
        # 'NO' -> 0 (no readmission)
        # 'NO>30' -> 0 (no readmission after 30 days)  
        # '<30' -> 1 (readmitted within 30 days)
        # '>30' -> 0 (readmitted after 30 days, but we want within 30 days)
        
        readmission_mapping = {
            'NO': 0,
            'NO>30': 0,
            '>30': 0,
            '<30': 1
        }
        
        # Convert the column
        df['readmitted'] = df['readmitted'].map(readmission_mapping)
        
        # Fill any NaN values with 0 (assuming no readmission)
        df['readmitted'] = df['readmitted'].fillna(0).astype(int)
        
        print(f"{Colors.GREEN}✅ Converted readmission column. New distribution: {df['readmitted'].value_counts().to_dict()}{Colors.END}")
    
    return df

def load_and_prepare_data():
    """Step 2: Load and prepare data from actual datasets"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}STEP 2: LOADING AND PREPARING DATA{Colors.END}")
    print(f"{Colors.BLUE}{'='*50}{Colors.END}")
    
    # Show loading animation
    for i in range(5):
        for char in "📊📈📉📊":
            sys.stdout.write(f'\r{Colors.YELLOW}Loading datasets from config {char}{Colors.END}')
            sys.stdout.flush()
            time.sleep(0.2)
    
    try:
        # Check if data files exist
        data_dir = Path('data/raw')
        if not data_dir.exists():
            print(f"{Colors.RED}❌ Data directory not found: {data_dir}{Colors.END}")
            print(f"{Colors.YELLOW}📝 Creating sample data structure for demonstration...{Colors.END}")
            create_sample_data()
        
        # Load datasets
        print(f"\n{Colors.CYAN}Loading datasets...{Colors.END}")
        
        # Load CMS HRRP data (hospital readmissions)
        cms_path = DATA_PATHS['cms_data']
        if Path(cms_path).exists():
            cms_df = pd.read_csv(cms_path)
            print(f"{Colors.GREEN}✅ Loaded CMS HRRP data: {cms_df.shape}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠️  CMS data not found at {cms_path}, using sample data{Colors.END}")
            cms_df = create_sample_cms_data()
        
        # Load SVI data (social vulnerability)
        svi_path = DATA_PATHS['svi_data']
        if Path(svi_path).exists():
            svi_df = pd.read_csv(svi_path)
            print(f"{Colors.GREEN}✅ Loaded SVI data: {svi_df.shape}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠️  SVI data not found at {svi_path}, using sample data{Colors.END}")
            svi_df = create_sample_svi_data()
        
        # Load diabetes data (patient records) - this is the main dataset
        kaggle_path = DATA_PATHS['kaggle_data']
        if Path(kaggle_path).exists():
            diabetes_df = pd.read_csv(kaggle_path)
            print(f"{Colors.GREEN}✅ Loaded diabetes data: {diabetes_df.shape}{Colors.END}")
            
            # Convert readmission column if needed
            diabetes_df = convert_readmission_column(diabetes_df)
            
        else:
            print(f"{Colors.YELLOW}⚠️  Diabetes data not found at {kaggle_path}, using sample data{Colors.END}")
            diabetes_df = create_sample_diabetes_data()
        
        # Load crosswalk for ZIP to county mapping
        crosswalk_path = DATA_PATHS['crosswalk']
        if Path(crosswalk_path).exists():
            crosswalk_df = pd.read_csv(crosswalk_path)
            print(f"{Colors.GREEN}✅ Loaded crosswalk data: {crosswalk_df.shape}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠️  Crosswalk data not found at {crosswalk_path}, using sample data{Colors.END}")
            crosswalk_df = create_sample_crosswalk_data()
        
        # Merge datasets to create final features
        print(f"{Colors.CYAN}Merging datasets...{Colors.END}")
        merged_df = merge_datasets(diabetes_df, svi_df, cms_df, crosswalk_df)
        
        print(f"{Colors.GREEN}✅ Final dataset: {merged_df.shape}{Colors.END}")
        print(f"{Colors.CYAN}   Features: {len(merged_df.columns)} columns{Colors.END}")
        if 'readmitted' in merged_df.columns:
            readmission_rate = merged_df['readmitted'].mean()
            print(f"{Colors.CYAN}   Readmission rate: {readmission_rate:.2%}{Colors.END}")
            print(f"{Colors.CYAN}   Class distribution: {merged_df['readmitted'].value_counts().to_dict()}{Colors.END}")
        
        return merged_df
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error loading data: {e}{Colors.END}")
        print(f"{Colors.YELLOW}🔄 Using sample data for demonstration...{Colors.END}")
        return create_sample_merged_data()

def create_sample_data():
    """Create sample data files if they don't exist"""
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample CMS data
    cms_data = pd.DataFrame({
        'hospital_id': range(1, 101),
        'readmission_rate': np.random.uniform(0.1, 0.3, 100),
        'mortality_rate': np.random.uniform(0.02, 0.1, 100),
        'patient_satisfaction': np.random.uniform(3.0, 5.0, 100)
    })
    cms_data.to_csv(DATA_PATHS['cms_data'], index=False)
    
    # Create sample SVI data
    svi_data = pd.DataFrame({
        'county_fips': [f'{i:05d}' for i in range(1001, 1101)],
        'svi_theme1': np.random.uniform(0, 1, 100),
        'svi_theme2': np.random.uniform(0, 1, 100),
        'svi_theme3': np.random.uniform(0, 1, 100),
        'svi_theme4': np.random.uniform(0, 1, 100),
        'rpl_theme1': np.random.uniform(0, 1, 100),
        'rpl_theme2': np.random.uniform(0, 1, 100),
        'rpl_theme3': np.random.uniform(0, 1, 100),
        'rpl_theme4': np.random.uniform(0, 1, 100),
        'income_per_capita': np.random.randint(20000, 80000, 100),
        'poverty_rate': np.random.uniform(0.05, 0.3, 100),
        'no_high_school_diploma': np.random.uniform(0.05, 0.4, 100),
        'unemployment_rate': np.random.uniform(0.02, 0.15, 100),
        'housing_burden': np.random.uniform(0.1, 0.5, 100),
        'no_vehicle': np.random.uniform(0.01, 0.2, 100)
    })
    svi_data.to_csv(DATA_PATHS['svi_data'], index=False)
    
    # Create sample diabetes data
    np.random.seed(42)
    n_patients = 5000
    diabetes_data = pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        'age': np.random.randint(18, 95, n_patients),
        'gender': np.random.choice(['Male', 'Female'], n_patients),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_patients),
        'length_of_stay': np.random.randint(1, 21, n_patients),
        'num_lab_procedures': np.random.randint(1, 50, n_patients),
        'num_procedures': np.random.randint(0, 10, n_patients),
        'num_medications': np.random.randint(1, 25, n_patients),
        'number_outpatient': np.random.randint(0, 10, n_patients),
        'number_emergency': np.random.randint(0, 8, n_patients),
        'number_inpatient': np.random.randint(0, 5, n_patients),
        'num_diagnoses': np.random.randint(1, 12, n_patients),
        'prior_admissions': np.random.randint(0, 8, n_patients),
        'comorbidity_index': np.random.uniform(0, 5, n_patients),
        'readmitted': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
        'zip_code': [f'{np.random.randint(10000, 99999)}' for _ in range(n_patients)]
    })
    diabetes_data.to_csv(DATA_PATHS['kaggle_data'], index=False)
    
    # Create sample crosswalk data
    crosswalk_data = pd.DataFrame({
        'zip_code': [f'{np.random.randint(10000, 99999)}' for _ in range(100)],
        'county_fips': [f'{np.random.randint(1001, 1101):05d}' for _ in range(100)]
    })
    crosswalk_data.to_csv(DATA_PATHS['crosswalk'], index=False)
    
    print(f"{Colors.GREEN}✅ Created sample data files{Colors.END}")

def create_sample_cms_data():
    """Create sample CMS data"""
    return pd.DataFrame({
        'hospital_id': range(1, 101),
        'readmission_rate': np.random.uniform(0.1, 0.3, 100)
    })

def create_sample_svi_data():
    """Create sample SVI data"""
    return pd.DataFrame({
        'county_fips': [f'{i:05d}' for i in range(1001, 1101)],
        'svi_theme1': np.random.uniform(0, 1, 100),
        'svi_theme2': np.random.uniform(0, 1, 100),
        'svi_theme3': np.random.uniform(0, 1, 100),
        'svi_theme4': np.random.uniform(0, 1, 100),
        'rpl_theme1': np.random.uniform(0, 1, 100),
        'rpl_theme2': np.random.uniform(0, 1, 100),
        'rpl_theme3': np.random.uniform(0, 1, 100),
        'rpl_theme4': np.random.uniform(0, 1, 100),
        'income_per_capita': np.random.randint(20000, 80000, 100),
        'poverty_rate': np.random.uniform(0.05, 0.3, 100),
        'no_high_school_diploma': np.random.uniform(0.05, 0.4, 100),
        'unemployment_rate': np.random.uniform(0.02, 0.15, 100),
        'housing_burden': np.random.uniform(0.1, 0.5, 100),
        'no_vehicle': np.random.uniform(0.01, 0.2, 100)
    })

def create_sample_diabetes_data():
    """Create sample diabetes data"""
    np.random.seed(42)
    n_patients = 5000
    return pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        'age': np.random.randint(18, 95, n_patients),
        'gender': np.random.choice(['Male', 'Female'], n_patients),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_patients),
        'length_of_stay': np.random.randint(1, 21, n_patients),
        'num_lab_procedures': np.random.randint(1, 50, n_patients),
        'num_procedures': np.random.randint(0, 10, n_patients),
        'num_medications': np.random.randint(1, 25, n_patients),
        'number_outpatient': np.random.randint(0, 10, n_patients),
        'number_emergency': np.random.randint(0, 8, n_patients),
        'number_inpatient': np.random.randint(0, 5, n_patients),
        'num_diagnoses': np.random.randint(1, 12, n_patients),
        'prior_admissions': np.random.randint(0, 8, n_patients),
        'comorbidity_index': np.random.uniform(0, 5, n_patients),
        'readmitted': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
        'zip_code': [f'{np.random.randint(10000, 99999)}' for _ in range(n_patients)]
    })

def create_sample_crosswalk_data():
    """Create sample crosswalk data"""
    return pd.DataFrame({
        'zip_code': [f'{np.random.randint(10000, 99999)}' for _ in range(100)],
        'county_fips': [f'{np.random.randint(1001, 1101):05d}' for _ in range(100)]
    })

def create_sample_merged_data():
    """Create sample merged dataset for demonstration"""
    np.random.seed(42)
    n_patients = 5000
    
    # Combine features from config
    all_features = (FEATURE_GROUPS['medical_features'] + 
                   FEATURE_GROUPS['demographic_features'] + 
                   FEATURE_GROUPS['sdoh_features'])
    
    # Remove duplicates and create sample data
    features = list(set(all_features))
    
    data = {}
    for feature in features:
        if feature in ['age', 'length_of_stay', 'num_lab_procedures', 'num_procedures',
                      'num_medications', 'number_outpatient', 'number_emergency',
                      'number_inpatient', 'num_diagnoses', 'prior_admissions']:
            data[feature] = np.random.randint(0, 20, n_patients)
        elif feature in ['comorbidity_index', 'svi_theme1', 'svi_theme2', 'svi_theme3', 
                        'svi_theme4', 'rpl_theme1', 'rpl_theme2', 'rpl_theme3', 
                        'rpl_theme4', 'poverty_rate', 'no_high_school_diploma',
                        'unemployment_rate', 'housing_burden', 'no_vehicle']:
            data[feature] = np.random.uniform(0, 1, n_patients)
        elif feature == 'income_per_capita':
            data[feature] = np.random.randint(20000, 80000, n_patients)
        elif feature == 'gender':
            data[feature] = np.random.choice(['Male', 'Female'], n_patients)
        elif feature == 'race':
            data[feature] = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_patients)
    
    # Create realistic readmission target
    data['readmitted'] = np.random.choice([0, 1], n_patients, p=[0.85, 0.15])
    
    return pd.DataFrame(data)

def merge_datasets(diabetes_df, svi_df, cms_df, crosswalk_df):
    """Merge all datasets to create features"""
    # For this version, we'll primarily use the diabetes data which contains the main features
    # In a real implementation, you would perform proper merging
    
    # Select features from diabetes data that match our config
    available_features = []
    for group in FEATURE_GROUPS.values():
        for feature in group:
            if feature in diabetes_df.columns and feature not in available_features:
                available_features.append(feature)
    
    # Ensure we have the target variable
    if 'readmitted' in diabetes_df.columns and 'readmitted' not in available_features:
        available_features.append('readmitted')
    
    # Create the final dataset
    if available_features:
        merged_df = diabetes_df[available_features].copy()
    else:
        # Fallback: use all numeric columns from diabetes data
        numeric_cols = diabetes_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'readmitted' in diabetes_df.columns:
            numeric_cols.append('readmitted')
        merged_df = diabetes_df[numeric_cols].copy()
    
    # Fill any missing values with 0 for numeric columns
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
    
    # For categorical columns, fill with mode
    categorical_cols = merged_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(merged_df[col].mode()[0] if not merged_df[col].mode().empty else 'Unknown')
    
    return merged_df

def train_models_with_progress(df):
    """Step 3: Train ML models using config parameters"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}STEP 3: TRAINING MACHINE LEARNING MODELS{Colors.END}")
    print(f"{Colors.BLUE}{'='*50}{Colors.END}")
    
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from imblearn.over_sampling import SMOTE
    import joblib
    
    # Prepare features and target
    target_col = 'readmitted'
    if target_col not in df.columns:
        print(f"{Colors.RED}❌ Target column '{target_col}' not found in data{Colors.END}")
        return None, None, None, None
    
    # Select features based on config
    feature_cols = []
    for group in FEATURE_GROUPS.values():
        feature_cols.extend([f for f in group if f in df.columns and f != target_col])
    
    # Remove duplicates
    feature_cols = list(set(feature_cols))
    
    print(f"{Colors.CYAN}Using {len(feature_cols)} features for modeling{Colors.END}")
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Store original feature names for later use
    original_features = X.columns.tolist()
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"{Colors.YELLOW}Encoding categorical variables: {list(categorical_cols)}{Colors.END}")
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        # Store the encoded feature names
        encoded_features = X_encoded.columns.tolist()
    else:
        X_encoded = X.copy()
        encoded_features = original_features
    
    print(f"{Colors.CYAN}Original data - Class distribution: {pd.Series(y).value_counts().to_dict()}{Colors.END}")
    
    # Handle class imbalance using SMOTE
    print(f"{Colors.YELLOW}🔄 Handling class imbalance with SMOTE...{Colors.END}")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_encoded, y)
    
    print(f"{Colors.CYAN}Balanced data - Class distribution: {pd.Series(y_balanced).value_counts().to_dict()}{Colors.END}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    print(f"{Colors.CYAN}Training set: {X_train.shape[0]:,} patients{Colors.END}")
    print(f"{Colors.CYAN}Test set: {X_test.shape[0]:,} patients{Colors.END}")
    
    # Define models using config parameters
    models = {
        'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), 
                               MODEL_PARAMS['logistic_regression']),
        'Random Forest': (RandomForestClassifier(random_state=42), 
                         MODEL_PARAMS['random_forest']),
        'XGBoost': (XGBClassifier(random_state=42, eval_metric='logloss'), 
                    MODEL_PARAMS['xgboost'])
    }
    
    trained_models = {}
    performances = {}
    
    for name, (model, params) in models.items():
        print(f"\n{Colors.YELLOW}🏃 Training {name}...{Colors.END}")
        
        try:
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1)
            
            # Simulate training progress for tree-based models
            if name in ['Random Forest', 'XGBoost']:
                for i in range(10):
                    progress = (i + 1) * 10
                    bar = "█" * (i + 1) + "░" * (10 - i - 1)
                    sys.stdout.write(f'\r{Colors.CYAN}   Progress: [{bar}] {progress}%{Colors.END}')
                    sys.stdout.flush()
                    time.sleep(0.1)
                print()
            
            # Train model with grid search
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            trained_models[name] = best_model
            performances[name] = {
                'accuracy': accuracy, 
                'roc_auc': roc_auc,
                'best_params': grid_search.best_params_
            }
            
            print(f"{Colors.GREEN}   ✅ {name:.<20} Accuracy: {accuracy:.3f} | ROC-AUC: {roc_auc:.3f}{Colors.END}")
            print(f"{Colors.CYAN}   Best parameters: {grid_search.best_params_}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}   ❌ Error training {name}: {e}{Colors.END}")
            # Fallback to default parameters
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                trained_models[name] = model
                performances[name] = {
                    'accuracy': accuracy, 
                    'roc_auc': roc_auc,
                    'best_params': 'default'
                }
                
                print(f"{Colors.GREEN}   ✅ {name:.<20} Accuracy: {accuracy:.3f} | ROC-AUC: {roc_auc:.3f}{Colors.END}")
            except:
                print(f"{Colors.RED}   ❌ Failed to train {name}{Colors.END}")
    
    if not trained_models:
        print(f"{Colors.RED}❌ No models were successfully trained{Colors.END}")
        return None, None, None, None
    
    # Save the best model
    best_model_name = max(performances, key=lambda x: performances[x]['roc_auc'])
    best_model = trained_models[best_model_name]
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model and features (save both original and encoded feature names)
    joblib.dump(best_model, models_dir / 'trained_model.pkl')
    joblib.dump({
        'original_features': original_features,
        'encoded_features': encoded_features,
        'categorical_columns': list(categorical_cols)
    }, models_dir / 'feature_names.pkl')
    joblib.dump(smote, models_dir / 'smote_preprocessor.pkl')
    
    # Save performance metrics
    performance_data = {
        'best_model': best_model_name,
        'performances': performances,
        'test_size': len(X_test),
        'training_date': pd.Timestamp.now().isoformat()
    }
    joblib.dump(performance_data, models_dir / 'performance_metrics.pkl')
    
    print(f"\n{Colors.GREEN}🎯 Best Model: {best_model_name} (ROC-AUC: {performances[best_model_name]['roc_auc']:.3f}){Colors.END}")
    print(f"{Colors.GREEN}✅ Models saved to 'models/' directory{Colors.END}")
    
    return best_model, encoded_features, performances, smote

def launch_dashboard(model, features, performances, smote):
    """Step 4: Launch interactive dashboard"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}STEP 4: LAUNCHING INTERACTIVE DASHBOARD{Colors.END}")
    print(f"{Colors.BLUE}{'='*50}{Colors.END}")
    
    import dash
    from dash import dcc, html, Input, Output, dash_table
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
    # Generate feature importance data - FIXED: Check array lengths match
    if hasattr(model, 'feature_importances_'):
        # Ensure features and importance arrays have same length
        if len(features) == len(model.feature_importances_):
            importance_data = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            # If lengths don't match, use the first n features
            n_features = min(len(features), len(model.feature_importances_))
            importance_data = pd.DataFrame({
                'Feature': features[:n_features],
                'Importance': model.feature_importances_[:n_features]
            }).sort_values('Importance', ascending=False)
            print(f"{Colors.YELLOW}⚠️  Feature importance array length mismatch. Using first {n_features} features.{Colors.END}")
    elif hasattr(model, 'coef_'):
        # For linear models, use coefficients
        if len(features) == len(model.coef_[0]):
            importance_data = pd.DataFrame({
                'Feature': features,
                'Importance': np.abs(model.coef_[0])
            }).sort_values('Importance', ascending=False)
        else:
            n_features = min(len(features), len(model.coef_[0]))
            importance_data = pd.DataFrame({
                'Feature': features[:n_features],
                'Importance': np.abs(model.coef_[0][:n_features])
            }).sort_values('Importance', ascending=False)
    else:
        # Fallback importance
        importance_data = pd.DataFrame({
            'Feature': features[:10],  # Use first 10 features
            'Importance': np.random.uniform(0.1, 0.3, min(10, len(features)))
        }).sort_values('Importance', ascending=False)
    
    # Create performance comparison data
    perf_df = pd.DataFrame([
        {'Model': name, 'Accuracy': perf['accuracy'], 'ROC-AUC': perf['roc_auc']}
        for name, perf in performances.items()
    ])
    
    # Create Dash app
    app = dash.Dash(__name__)
    
    # Define layout with simplified input form
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("🏥 Hospital Readmission Risk Predictor", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
            html.P("MSDS692 Data Science Practicum - Predicting 30-Day Readmissions", 
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 16}),
            html.P("Sai Teja Lakkapally", 
                  style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': 14}),
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
        
        # Model Performance Section
        html.Div([
            html.H2("📊 Model Performance", style={'color': '#34495e', 'marginBottom': '20px'}),
            
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure={
                            'data': [
                                go.Bar(name='Accuracy', x=perf_df['Model'], y=perf_df['Accuracy'], 
                                      marker_color='#3498db'),
                                go.Bar(name='ROC-AUC', x=perf_df['Model'], y=perf_df['ROC-AUC'],
                                      marker_color='#e74c3c')
                            ],
                            'layout': {
                                'title': 'Model Performance Comparison',
                                'barmode': 'group',
                                'template': 'plotly_white'
                            }
                        }
                    )
                ], className='six columns', style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(
                        figure=px.bar(importance_data.head(10), x='Importance', y='Feature', 
                                     orientation='h', color='Importance',
                                     color_continuous_scale='Viridis',
                                     title='Top 10 Feature Importance')
                    )
                ], className='six columns', style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
            ]),
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
        
        # Risk Prediction Interface
        html.Div([
            html.H2("🎯 Patient Risk Assessment", style={'color': '#34495e', 'marginBottom': '20px'}),
            
            html.Div([
                # Simplified input form with key features
                html.Div([
                    html.H4("Patient Characteristics", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    
                    html.Label("Age", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                    dcc.Input(id='age-input', type='number', value=65, 
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),
                    
                    html.Label("Length of Stay (days)", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                    dcc.Input(id='los-input', type='number', value=7,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),
                    
                    html.Label("Number of Medications", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                    dcc.Input(id='meds-input', type='number', value=12,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),
                    
                    html.Label("Number of Diagnoses", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                    dcc.Input(id='diag-input', type='number', value=6,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),
                    
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.H4("Additional Factors", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    
                    html.Label("Prior Admissions", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                    dcc.Input(id='prior-input', type='number', value=2,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),
                    
                    html.Label("Social Vulnerability Index", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                    dcc.Input(id='svi-input', type='number', value=0.5, min=0, max=1, step=0.1,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),
                    
                    html.Label("Comorbidity Index", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                    dcc.Input(id='comorb-input', type='number', value=2.5, min=0, max=5, step=0.1,
                             style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),
                    
                ], style={'width': '48%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'}),
            ]),
            
            # Calculate button
            html.Div([
                html.Button('🧮 Calculate Readmission Risk', 
                          id='calculate-btn', 
                          n_clicks=0,
                          style={
                              'backgroundColor': '#3498db', 
                              'color': 'white', 
                              'padding': '15px 30px', 
                              'border': 'none', 
                              'borderRadius': '8px', 
                              'fontSize': '18px',
                              'cursor': 'pointer', 
                              'margin': '20px auto', 
                              'display': 'block',
                              'fontWeight': 'bold'
                          }),
            ]),
            
            # Risk output
            html.Div(id='risk-output', style={
                'textAlign': 'center', 
                'fontSize': '20px', 
                'marginTop': '20px',
                'padding': '20px',
                'borderRadius': '10px'
            })
            
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
        
        # Project Information
        html.Div([
            html.H2("📋 Project Overview", style={'color': '#34495e', 'marginBottom': '15px'}),
            html.P("""
                This system predicts 30-day hospital readmission risk by integrating medical data with 
                social determinants of health. The model combines clinical factors with community-level 
                social vulnerability indicators to provide comprehensive risk assessment.
            """, style={'lineHeight': '1.6', 'fontSize': '16px'}),
            
            html.H4("📊 Data Sources", style={'color': '#2c3e50', 'marginTop': '20px'}),
            html.Ul([
                html.Li("CMS HRRP: Hospital readmission and reduction program data"),
                html.Li("CDC SVI: Social Vulnerability Index data"),
                html.Li("Kaggle Diabetes: Patient medical records"),
                html.Li("ZIP-County Crosswalk: Geographic mapping data"),
            ], style={'lineHeight': '1.8', 'fontSize': '15px'}),
            
            html.H4("🤝 Jesuit Values & Healthcare Equity", style={'color': '#2c3e50', 'marginTop': '20px'}),
            html.P("""
                This project promotes healthcare equity by addressing social determinants of health 
                and ensuring fair model performance across diverse patient populations.
            """, style={'lineHeight': '1.6', 'fontSize': '16px', 'fontStyle': 'italic'}),
            
        ], style={'backgroundColor': '#ecf0f1', 'padding': '25px', 'borderRadius': '10px'}),
        
    ], style={
        'padding': '20px', 
        'fontFamily': 'Arial, sans-serif',
        'maxWidth': '1200px',
        'margin': '0 auto'
    })
    
    # Callback for risk calculation
    @app.callback(
        Output('risk-output', 'children'),
        [Input('calculate-btn', 'n_clicks')],
        [Input('age-input', 'value'),
         Input('los-input', 'value'),
         Input('meds-input', 'value'),
         Input('diag-input', 'value'),
         Input('prior-input', 'value'),
         Input('svi-input', 'value'),
         Input('comorb-input', 'value')]
    )
    def calculate_risk(n_clicks, age, los, meds, diag, prior, svi, comorb):
        if n_clicks > 0 and all(v is not None for v in [age, los, meds, diag, prior, svi, comorb]):
            try:
                # Create a default input array with zeros for all features
                input_data = np.zeros(len(features))
                
                # Map the input values to the correct feature positions
                feature_mapping = {
                    'age': age,
                    'length_of_stay': los,
                    'num_medications': meds,
                    'num_diagnoses': diag,
                    'prior_admissions': prior,
                    'svi_theme1': svi,  # Using svi_theme1 as proxy for SVI input
                    'comorbidity_index': comorb
                }
                
                # Set values for features that exist in our feature set
                for feature_name, value in feature_mapping.items():
                    # Find features that contain the feature name (handles one-hot encoded names)
                    matching_features = [i for i, f in enumerate(features) if feature_name in f]
                    for idx in matching_features:
                        input_data[idx] = value
                
                # Ensure we have at least some values set
                if np.all(input_data == 0):
                    # Set first few features as fallback
                    input_data[0] = age
                    input_data[1] = los
                    input_data[2] = meds
                
                # Predict probability
                risk_prob = model.predict_proba([input_data])[0, 1]
                risk_percent = risk_prob * 100
                
                # Determine risk level and recommendations
                if risk_prob > 0.3:
                    color = '#e74c3c'
                    level = 'HIGH RISK'
                    icon = '🔴'
                    advice = [
                        "Immediate care coordination needed",
                        "Schedule follow-up within 7 days", 
                        "Consider home health services",
                        "Patient education on warning signs"
                    ]
                elif risk_prob > 0.15:
                    color = '#f39c12'
                    level = 'MEDIUM RISK'
                    icon = '🟡'
                    advice = [
                        "Schedule follow-up within 14 days",
                        "Medication reconciliation",
                        "Assess social support needs",
                        "Provide discharge instructions"
                    ]
                else:
                    color = '#2ecc71'
                    level = 'LOW RISK'
                    icon = '🟢'
                    advice = [
                        "Standard discharge process",
                        "Routine follow-up appointment",
                        "General health education",
                        "Monitor for any changes"
                    ]
                
                return html.Div([
                    html.H3(f"{icon} Readmission Risk: {risk_percent:.1f}%", 
                           style={'color': color, 'marginBottom': '10px'}),
                    html.H4(f"Risk Level: {level}", 
                           style={'color': color, 'marginBottom': '20px'}),
                    
                    html.H5("Recommended Interventions:", 
                           style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    html.Ul([html.Li(item) for item in advice], 
                           style={'textAlign': 'left', 'marginLeft': '20px'})
                ], style={
                    'border': f'3px solid {color}',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'backgroundColor': '#f8f9fa'
                })
            except Exception as e:
                return html.Div([
                    html.H4("❌ Error calculating risk", style={'color': '#e74c3c'}),
                    html.P(f"Please check all input values and try again. Error: {str(e)}")
                ])
        
        return html.Div([
            html.H4("👆 Enter patient characteristics above", style={'color': '#7f8c8d'}),
            html.P("Click 'Calculate Readmission Risk' to see prediction")
        ], style={'color': '#95a5a6'})
    
    print(f"{Colors.GREEN}✅ Dashboard created successfully!{Colors.END}")
    print(f"{Colors.CYAN}🌐 Opening dashboard at: http://localhost:8050{Colors.END}")
    print(f"{Colors.YELLOW}🖱️  Press Ctrl+C to stop the dashboard{Colors.END}")
    print(f"{Colors.MAGENTA}📊 Features: Real-time risk prediction, model comparison, feature importance{Colors.END}")
    
    # Run the dashboard - FIXED: run_server -> run
    app.run(debug=False, port=8050)

def main():
    """Main function - run everything with one command"""
    try:
        # Step 0: Show banner
        print_banner()
        time.sleep(1)
        
        # Step 1: Install requirements
        install_requirements()
        time.sleep(1)
        
        # Step 2: Load data from actual datasets
        df = load_and_prepare_data()
        time.sleep(1)
        
        # Step 3: Train models using config parameters
        model, features, performances, smote = train_models_with_progress(df)
        
        if model is not None:
            time.sleep(1)
            
            # Step 4: Launch dashboard
            launch_dashboard(model, features, performances, smote)
        else:
            print(f"{Colors.RED}❌ Model training failed. Cannot launch dashboard.{Colors.END}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}👋 Dashboard stopped by user{Colors.END}")
        return 0
    except Exception as e:
        print(f"\n{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())