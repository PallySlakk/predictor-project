"""
Hospital Readmission Prediction Package
MSDS692 - Data Science Practicum

A comprehensive machine learning pipeline for predicting 30-day hospital readmissions
using medical data and social determinants of health.
"""

__version__ = '1.0.0'
__author__ = 'Sai Teja Lakkapally'
__email__ = 'slakkapally@regis.edu'

# Import main classes for easy access
from .etl import DataETL
from .features import FeatureEngineer
from .model import ReadmissionModel
from .dashboard import ReadmissionDashboard
from .utils import Config, setup_logging

# Define what gets imported with "from src import *"
__all__ = [
    'DataETL',
    'FeatureEngineer', 
    'ReadmissionModel',
    'ReadmissionDashboard',
    'Config',
    'setup_logging'
]

# Package metadata
PACKAGE_INFO = {
    'name': 'hospital_readmission_predictor',
    'version': __version__,
    'description': 'Predict 30-day hospital readmissions using ML and SDOH',
    'author': __author__,
    'email': __email__,
    'license': 'MIT',
    'keywords': ['healthcare', 'machine learning', 'readmission', 'SDOH']
}

print(f"Hospital Readmission Predictor v{__version__}")
print("Successfully imported package modules")