import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, classification_report, 
                           confusion_matrix, precision_recall_curve, roc_curve)
from sklearn.calibration import calibration_curve
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class ReadmissionModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        
    def train_baseline_models(self, X_train, y_train, cv_folds=5):
        """Train baseline models with cross-validation"""
        print("Training baseline models...")
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state, 
                class_weight='balanced',
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced_subsample',
                n_estimators=100
            )
        }
        
        # Cross-validation scoring
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scoring = {'roc_auc': 'roc_auc', 'average_precision': 'average_precision'}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores_auc = cross_val_score(model, X_train, y_train, 
                                          cv=cv, scoring='roc_auc', n_jobs=-1)
            cv_scores_ap = cross_val_score(model, X_train, y_train, 
                                         cv=cv, scoring='average_precision', n_jobs=-1)
            
            # Train final model
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Store results
            self.results[name] = {
                'cv_roc_auc_mean': cv_scores_auc.mean(),
                'cv_roc_auc_std': cv_scores_auc.std(),
                'cv_ap_mean': cv_scores_ap.mean(),
                'cv_ap_std': cv_scores_ap.std(),
                'model': model
            }
            
            print(f"{name} - ROC-AUC: {cv_scores_auc.mean():.4f} (±{cv_scores_auc.std():.4f})")
            print(f"{name} - PR-AUC: {cv_scores_ap.mean():.4f} (±{cv_scores_ap.std():.4f})")
        
        return self
    
    def train_advanced_models(self, X_train, y_train, cv_folds=5):
        """Train advanced models with hyperparameter tuning"""
        print("Training advanced models...")
        
        # Calculate class weight for imbalance
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        # Define models and parameter grids
        models_config = {
            'xgboost': {
                'model': XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1],
                    'scale_pos_weight': [scale_pos_weight]
                }
            },
            'lightgbm': {
                'model': LGBMClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1],
                    'scale_pos_weight': [scale_pos_weight]
                }
            }
        }
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, config in models_config.items():
            print(f"Training {name} with hyperparameter tuning...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store best model
            best_model = grid_search.best_estimator_
            self.models[name] = best_model
            
            # Cross-validation with best model
            cv_scores_auc = cross_val_score(best_model, X_train, y_train, 
                                          cv=cv, scoring='roc_auc', n_jobs=-1)
            cv_scores_ap = cross_val_score(best_model, X_train, y_train, 
                                         cv=cv, scoring='average_precision', n_jobs=-1)
            
            self.results[name] = {
                'cv_roc_auc_mean': cv_scores_auc.mean(),
                'cv_roc_auc_std': cv_scores_auc.std(),
                'cv_ap_mean': cv_scores_ap.mean(),
                'cv_ap_std': cv_scores_ap.std(),
                'best_params': grid_search.best_params_,
                'model': best_model
            }
            
            print(f"{name} - Best params: {grid_search.best_params_}")
            print(f"{name} - ROC-AUC: {cv_scores_auc.mean():.4f} (±{cv_scores_auc.std():.4f})")
            print(f"{name} - PR-AUC: {cv_scores_ap.mean():.4f} (±{cv_scores_ap.std():.4f})")
        
        return self
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("Evaluating models on test set...")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            # Store results
            evaluation_results[name] = {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred,
                'model': model
            }
            
            # Print results
            print(f"{name} Test Results:")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  PR-AUC: {pr_auc:.4f}")
            print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted']))
            
            # Store feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        # Determine best model
        self._select_best_model(evaluation_results)
        
        return evaluation_results
    
    def _select_best_model(self, evaluation_results):
        """Select the best model based on ROC-AUC"""
        best_roc_auc = 0
        best_model_name = None
        
        for name, results in evaluation_results.items():
            if results['roc_auc'] > best_roc_auc:
                best_roc_auc = results['roc_auc']
                best_model_name = name
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"Best model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
    
    def shap_analysis(self, X, feature_names, model_name=None):
        """Perform SHAP analysis for model interpretability"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        print(f"Performing SHAP analysis for {model_name}...")
        
        # Create explainer based on model type
        if model_name in ['random_forest', 'xgboost', 'lightgbm']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # For binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # For linear models
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Create feature importance DataFrame
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        return explainer, shap_values, shap_importance
    
    def fairness_analysis(self, X_test, y_test, sensitive_features):
        """Analyze model fairness across subgroups"""
        print("Performing fairness analysis...")
        
        model = self.best_model
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        fairness_results = {}
        
        for feature_name, feature_values in sensitive_features.items():
            print(f"Analyzing fairness for {feature_name}...")
            
            subgroup_metrics = {}
            unique_values = feature_values.unique()
            
            for value in unique_values:
                mask = feature_values == value
                subgroup_y_true = y_test[mask]
                subgroup_y_pred = y_pred[mask]
                subgroup_y_proba = y_pred_proba[mask]
                
                if len(subgroup_y_true) > 0 and len(np.unique(subgroup_y_true)) > 1:
                    roc_auc = roc_auc_score(subgroup_y_true, subgroup_y_proba)
                    pr_auc = average_precision_score(subgroup_y_true, subgroup_y_proba)
                    
                    subgroup_metrics[value] = {
                        'n_samples': len(subgroup_y_true),
                        'roc_auc': roc_auc,
                        'pr_auc': pr_auc,
                        'readmission_rate': subgroup_y_true.mean()
                    }
            
            fairness_results[feature_name] = subgroup_metrics
        
        return fairness_results
    
    def plot_model_comparison(self, evaluation_results):
        """Plot comparison of model performance"""
        models = list(evaluation_results.keys())
        roc_scores = [evaluation_results[model]['roc_auc'] for model in models]
        pr_scores = [evaluation_results[model]['pr_auc'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC-AUC comparison
        bars1 = ax1.bar(models, roc_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax1.set_title('Model Comparison - ROC-AUC', fontsize=14)
        ax1.set_ylabel('ROC-AUC Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # PR-AUC comparison
        bars2 = ax2.bar(models, pr_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax2.set_title('Model Comparison - PR-AUC', fontsize=14)
        ax2.set_ylabel('PR-AUC Score')
        ax2.set_ylim(0, 1)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, filepath='models/'):
        """Save trained models"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f'{filepath}/{name}_model.pkl')
        
        # Save best model separately
        if self.best_model:
            joblib.dump(self.best_model, f'{filepath}/best_model.pkl')
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='models/'):
        """Load trained models"""
        import glob
        
        model_files = glob.glob(f'{filepath}/*.pkl')
        
        for file in model_files:
            name = file.split('/')[-1].replace('_model.pkl', '')
            if name != 'best':
                self.models[name] = joblib.load(file)
        
        # Load best model
        best_model_file = f'{filepath}/best_model.pkl'
        if os.path.exists(best_model_file):
            self.best_model = joblib.load(best_model_file)
            self.best_model_name = 'best'
        
        print("Models loaded successfully")

# Usage example
if __name__ == "__main__":
    from etl import DataETL
    from features import FeatureEngineer
    
    # Load and prepare data
    etl = DataETL()
    data = etl.run_pipeline()
    
    feature_engineer = FeatureEngineer()
    X, y, feature_names = feature_engineer.prepare_features(data)
    
    # Handle imbalance
    X_balanced, y_balanced = feature_engineer.handle_imbalance(X, y, method='smote')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    # Train models
    model_trainer = ReadmissionModel()
    model_trainer.train_baseline_models(X_train, y_train)
    model_trainer.train_advanced_models(X_train, y_train)
    
    # Evaluate models
    results = model_trainer.evaluate_models(X_test, y_test)
    
    # SHAP analysis
    explainer, shap_values, shap_importance = model_trainer.shap_analysis(
        X_test.iloc[:1000], feature_names  # Use subset for performance
    )
    
    print("Model training and evaluation completed!")