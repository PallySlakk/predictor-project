import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import ReadmissionModel
from features import FeatureEngineer
from etl import DataETL

class TestReadmissionModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and models"""
        print("Setting up test environment...")
        
        # Generate small synthetic dataset for testing
        np.random.seed(42)
        self.n_samples = 1000
        
        self.X_test = pd.DataFrame({
            'length_of_stay': np.random.randint(1, 20, self.n_samples),
            'num_lab_procedures': np.random.randint(1, 50, self.n_samples),
            'num_procedures': np.random.randint(0, 10, self.n_samples),
            'num_medications': np.random.randint(1, 30, self.n_samples),
            'number_outpatient': np.random.randint(0, 10, self.n_samples),
            'number_emergency': np.random.randint(0, 5, self.n_samples),
            'number_inpatient': np.random.randint(0, 5, self.n_samples),
            'number_diagnoses': np.random.randint(1, 10, self.n_samples),
            'prior_admissions': np.random.randint(0, 5, self.n_samples),
            'comorbidity_index': np.random.randint(0, 5, self.n_samples),
            'utilization_score': np.random.uniform(0, 10, self.n_samples),
            'procedure_intensity': np.random.uniform(0, 5, self.n_samples),
            'RPL_THEME1': np.random.uniform(0, 1, self.n_samples),
            'RPL_THEME2': np.random.uniform(0, 1, self.n_samples),
            'RPL_THEME3': np.random.uniform(0, 1, self.n_samples),
            'RPL_THEME4': np.random.uniform(0, 1, self.n_samples),
            'overall_vulnerability': np.random.uniform(0, 1, self.n_samples),
            'economic_hardship': np.random.uniform(0, 50, self.n_samples),
            'healthcare_access_barrier': np.random.randint(0, 3, self.n_samples),
            'social_isolation_risk': np.random.randint(0, 3, self.n_samples)
        })
        
        # Create imbalanced target (15% readmission rate)
        n_readmitted = int(self.n_samples * 0.15)
        y_test = np.zeros(self.n_samples)
        y_test[:n_readmitted] = 1
        np.random.shuffle(y_test)
        self.y_test = y_test
        
        # Initialize model trainer
        self.model_trainer = ReadmissionModel(random_state=42)
        
    def test_data_loading(self):
        """Test that data loads correctly"""
        print("Testing data loading...")
        etl = DataETL()
        data = etl.load_all_data()
        
        self.assertIsNotNone(data.svi_data)
        self.assertIsNotNone(data.cms_data)
        self.assertIsNotNone(data.kaggle_data)
        self.assertIsNotNone(data.crosswalk_data)
        
        # Check data shapes
        self.assertGreater(len(data.svi_data), 0)
        self.assertGreater(len(data.cms_data), 0)
        self.assertGreater(len(data.kaggle_data), 0)
        
        print("✓ Data loading test passed")
    
    def test_feature_engineering(self):
        """Test feature engineering pipeline"""
        print("Testing feature engineering...")
        
        # Create sample data for feature engineering
        sample_data = pd.DataFrame({
            'length_of_stay': [5, 10, 3],
            'num_lab_procedures': [25, 40, 15],
            'num_procedures': [2, 5, 1],
            'num_medications': [8, 15, 5],
            'number_outpatient': [2, 5, 1],
            'number_emergency': [1, 3, 0],
            'number_inpatient': [1, 2, 0],
            'number_diagnoses': [4, 8, 2],
            'age': [45, 72, 35],
            'gender': ['Male', 'Female', 'Male'],
            'race': ['Caucasian', 'AfricanAmerican', 'Hispanic'],
            'RPL_THEME1': [0.3, 0.8, 0.5],
            'RPL_THEME2': [0.2, 0.7, 0.4],
            'RPL_THEME3': [0.4, 0.9, 0.6],
            'RPL_THEME4': [0.1, 0.6, 0.3],
            'EP_POV': [15, 35, 20],
            'EP_UNEMP': [5, 12, 8],
            'EP_NOHSDP': [10, 25, 15],
            'EP_NOVEH': [5, 15, 8],
            'max_glu_serum': ['None', '>200', 'Norm'],
            'A1Cresult': ['None', '>7', 'Norm'],
            'change': ['No', 'Yes', 'No'],
            'diabetesMed': ['Yes', 'Yes', 'No'],
            'readmitted': ['NO', '<30', 'NO'],
            'readmitted_30': [0, 1, 0]
        })
        
        feature_engineer = FeatureEngineer()
        X, y, feature_names = feature_engineer.prepare_features(sample_data)
        
        # Check that features are created
        self.assertGreater(X.shape[1], 0)
        self.assertEqual(len(y), len(sample_data))
        self.assertIn('comorbidity_index', feature_names)
        self.assertIn('overall_vulnerability', feature_names)
        
        print("✓ Feature engineering test passed")
    
    def test_baseline_models_training(self):
        """Test that baseline models can be trained"""
        print("Testing baseline model training...")
        
        # Use smaller dataset for faster testing
        X_train = self.X_test.iloc[:200]
        y_train = self.y_test[:200]
        
        self.model_trainer.train_baseline_models(X_train, y_train, cv_folds=3)
        
        # Check that models are trained
        self.assertIn('logistic_regression', self.model_trainer.models)
        self.assertIn('random_forest', self.model_trainer.models)
        
        # Check that results are stored
        self.assertIn('logistic_regression', self.model_trainer.results)
        self.assertIn('random_forest', self.model_trainer.results)
        
        print("✓ Baseline models training test passed")
    
    def test_advanced_models_training(self):
        """Test that advanced models can be trained"""
        print("Testing advanced model training...")
        
        # Use smaller dataset for faster testing
        X_train = self.X_test.iloc[:200]
        y_train = self.y_test[:200]
        
        self.model_trainer.train_advanced_models(X_train, y_train, cv_folds=2)
        
        # Check that models are trained
        self.assertIn('xgboost', self.model_trainer.models)
        self.assertIn('lightgbm', self.model_trainer.models)
        
        print("✓ Advanced models training test passed")
    
    def test_model_evaluation(self):
        """Test model evaluation functionality"""
        print("Testing model evaluation...")
        
        # Train a simple model first
        X_train = self.X_test.iloc[:300]
        y_train = self.y_test[:300]
        X_test = self.X_test.iloc[300:400]
        y_test = self.y_test[300:400]
        
        self.model_trainer.train_baseline_models(X_train, y_train, cv_folds=2)
        results = self.model_trainer.evaluate_models(X_test, y_test)
        
        # Check evaluation results
        for model_name, result in results.items():
            self.assertIn('roc_auc', result)
            self.assertIn('pr_auc', result)
            self.assertIn('y_pred_proba', result)
            
            # Scores should be between 0 and 1
            self.assertGreaterEqual(result['roc_auc'], 0)
            self.assertLessEqual(result['roc_auc'], 1)
        
        print("✓ Model evaluation test passed")
    
    def test_class_imbalance_handling(self):
        """Test class imbalance handling methods"""
        print("Testing class imbalance handling...")
        
        feature_engineer = FeatureEngineer()
        
        # Test SMOTE
        X_resampled, y_resampled = feature_engineer.handle_imbalance(
            self.X_test, self.y_test, method='smote'
        )
        
        self.assertEqual(len(X_resampled), len(y_resampled))
        
        # Test undersampling
        X_undersampled, y_undersampled = feature_engineer.handle_imbalance(
            self.X_test, self.y_test, method='undersample'
        )
        
        self.assertEqual(len(X_undersampled), len(y_undersampled))
        
        print("✓ Class imbalance handling test passed")
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        print("Testing feature importance...")
        
        # Train a model that has feature_importances_
        X_train = self.X_test.iloc[:200]
        y_train = self.y_test[:200]
        
        self.model_trainer.train_baseline_models(X_train, y_train, cv_folds=2)
        
        # Evaluate to trigger feature importance calculation
        results = self.model_trainer.evaluate_models(
            self.X_test.iloc[200:300], self.y_test[200:300]
        )
        
        # Check if feature importance is calculated for tree-based models
        if 'random_forest' in self.model_trainer.feature_importance:
            importance_df = self.model_trainer.feature_importance['random_forest']
            self.assertGreater(len(importance_df), 0)
            self.assertIn('feature', importance_df.columns)
            self.assertIn('importance', importance_df.columns)
        
        print("✓ Feature importance test passed")
    
    def test_model_saving_loading(self):
        """Test model saving and loading functionality"""
        print("Testing model saving and loading...")
        
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Train a model and save it
            X_train = self.X_test.iloc[:100]
            y_train = self.y_test[:100]
            
            self.model_trainer.train_baseline_models(X_train, y_train, cv_folds=2)
            self.model_trainer.save_models(temp_dir)
            
            # Check that model files are created
            import glob
            model_files = glob.glob(f"{temp_dir}/*.pkl")
            self.assertGreater(len(model_files), 0)
            
            # Test loading (create new instance to test loading)
            new_trainer = ReadmissionModel()
            new_trainer.load_models(temp_dir)
            
            # Check that models are loaded
            self.assertGreater(len(new_trainer.models), 0)
            
            print("✓ Model saving and loading test passed")
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)

def run_tests():
    """Run all tests"""
    print("Running Hospital Readmission Model Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReadmissionModels)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 50)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_tests()