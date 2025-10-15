"""
Utility functions for Hospital Readmission Prediction Project
MSDS692 - Data Science Practicum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import json
import os
from datetime import datetime
import logging

# Set up logging
def setup_logging(log_file='hospital_readmission.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Data validation functions
def validate_dataframe(df, expected_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content
    """
    logger = logging.getLogger(__name__)
    
    # Check if DataFrame is not empty
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    # Check minimum rows
    if len(df) < min_rows:
        logger.error(f"DataFrame has only {len(df)} rows, minimum required: {min_rows}")
        return False
    
    # Check expected columns if provided
    if expected_columns:
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
    
    logger.info(f"Data validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True

def check_class_balance(y, positive_class=1, warning_threshold=0.1):
    """
    Check class balance and return metrics
    """
    from collections import Counter
    
    class_counts = Counter(y)
    total_samples = len(y)
    positive_ratio = class_counts[positive_class] / total_samples
    
    metrics = {
        'total_samples': total_samples,
        'class_counts': dict(class_counts),
        'positive_ratio': positive_ratio,
        'is_balanced': warning_threshold <= positive_ratio <= (1 - warning_threshold),
        'imbalance_ratio': max(class_counts.values()) / min(class_counts.values())
    }
    
    return metrics

# Visualization utilities
def plot_roc_curves(models_dict, X_test, y_test, figsize=(10, 8)):
    """
    Plot ROC curves for multiple models
    """
    plt.figure(figsize=figsize)
    
    for name, model in models_dict.items():
        # Get predicted probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_pr_curves(models_dict, X_test, y_test, figsize=(10, 8)):
    """
    Plot Precision-Recall curves for multiple models
    """
    plt.figure(figsize=figsize)
    
    for name, model in models_dict.items():
        # Get predicted probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        plt.plot(recall, precision, label=f'{name} (AP = {pr_auc:.3f})', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Model Comparison', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_importance(feature_importance, feature_names, top_n=20, figsize=(12, 8)):
    """
    Plot feature importance
    """
    # Create DataFrame for sorting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=figsize)
    bars = plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
    
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importance', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center')
    
    plt.tight_layout()
    return plt.gcf()

# Model evaluation utilities
def evaluate_model_performance(model, X_test, y_test, model_name='Model'):
    """
    Comprehensive model evaluation
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Add probability-based metrics if available
    if y_pred_proba is not None:
        metrics.update({
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        })
    
    # Classification report
    metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    
    return metrics

def compare_models_performance(models_dict, X_test, y_test):
    """
    Compare performance of multiple models
    """
    results = []
    
    for name, model in models_dict.items():
        metrics = evaluate_model_performance(model, X_test, y_test, name)
        results.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics.get('roc_auc', 'N/A')}",
            'PR-AUC': f"{metrics.get('pr_auc', 'N/A')}"
        })
    
    return pd.DataFrame(results)

# Configuration management
class Config:
    """Configuration management for the project"""
    
    # Data paths
    DATA_DIR = 'data'
    RAW_DATA_DIR = 'data/raw'
    CLEAN_DATA_DIR = 'data/clean'
    PROCESSED_DATA_DIR = 'data/processed'
    MODEL_DIR = 'models'
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Feature settings
    NUMERIC_FEATURES = [
        'length_of_stay', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses', 'prior_admissions',
        'comorbidity_index', 'utilization_score', 'procedure_intensity'
    ]
    
    CATEGORICAL_FEATURES = [
        'gender', 'race', 'age_group', 'los_category', 'vulnerability_level'
    ]
    
    SVI_FEATURES = [
        'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4',
        'RPL_TOTAL', 'EP_POV', 'EP_UNEMP', 'EP_PCI', 'EP_NOHSDP'
    ]
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.CLEAN_DATA_DIR, 
            cls.PROCESSED_DATA_DIR, cls.MODEL_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logging.info("Project directories created")

# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    
    # Setup directories
    Config.setup_directories()
    
    logger.info("Utility functions loaded successfully")