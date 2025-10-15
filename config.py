# Configuration settings for the project
DATA_PATHS = {
    'svi_data': 'data/raw/CDC_SVI_2020.csv',
    'cms_data': 'data/raw/CMS_HRRP_2022.csv', 
    'kaggle_data': 'data/raw/kaggle_diabetes.csv',
    'crosswalk': 'data/raw/ZIP_COUNTY_122022.csv'
}

MODEL_PARAMS = {
    'logistic_regression': {
        'C': [0.1, 1.0, 10.0],
        'class_weight': ['balanced', None],
        'max_iter': [1000]
    },
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'class_weight': ['balanced', 'balanced_subsample']
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'scale_pos_weight': [1, 3, 5]
    }
}

FEATURE_GROUPS = {
    'medical_features': [
        'length_of_stay', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses', 'prior_admissions',
        'comorbidity_index'
    ],
    'demographic_features': [
        'age', 'gender', 'race'
    ],
    'sdoh_features': [
        'svi_theme1', 'svi_theme2', 'svi_theme3', 'svi_theme4',
        'rpl_theme1', 'rpl_theme2', 'rpl_theme3', 'rpl_theme4',
        'income_per_capita', 'poverty_rate', 'no_high_school_diploma',
        'unemployment_rate', 'housing_burden', 'no_vehicle'
    ]
}