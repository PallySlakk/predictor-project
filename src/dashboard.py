import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import shap
import base64
import io

class ReadmissionDashboard:
    def __init__(self, model, feature_names, data_processor):
        self.model = model
        self.feature_names = feature_names
        self.data_processor = data_processor
        self.app = dash.Dash(__name__)
        
        # Sample data for demonstration
        self.sample_data = self._create_sample_data()
        
        self.setup_layout()
        self.setup_callbacks()
    
    def _create_sample_data(self):
        """Create sample data for dashboard demonstration"""
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'feature': self.feature_names[:20],
            'importance': np.random.uniform(0, 1, 20),
            'impact': np.random.uniform(-0.2, 0.2, 20),
            'category': np.random.choice(['Medical', 'SDOH', 'Demographic'], 20)
        }).sort_values('importance', ascending=False)
        
        return sample_data
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Hospital Readmission Risk Predictor", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
            
            # Overview section
            html.Div([
                html.H2("Project Overview", style={'color': '#34495e'}),
                html.P("""
                    This dashboard presents a machine learning model that predicts 30-day hospital 
                    readmissions by integrating clinical data with social determinants of health (SDOH). 
                    The model helps identify high-risk patients and understand factors driving readmission risk.
                """, style={'fontSize': '16px', 'lineHeight': '1.6'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Model Performance Section
            html.Div([
                html.H2("Model Performance", style={'color': '#34495e'}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3("0.78", style={'color': '#27ae60', 'margin': '0'}),
                            html.P("ROC-AUC Score", style={'margin': '0'})
                        ], className='metric-box', style={
                            'backgroundColor': 'white', 
                            'padding': '20px', 
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], className='three columns'),
                    
                    html.Div([
                        html.Div([
                            html.H3("0.34", style={'color': '#e74c3c', 'margin': '0'}),
                            html.P("PR-AUC Score", style={'margin': '0'})
                        ], className='metric-box', style={
                            'backgroundColor': 'white', 
                            'padding': '20px', 
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], className='three columns'),
                    
                    html.Div([
                        html.Div([
                            html.H3("72%", style={'color': '#3498db', 'margin': '0'}),
                            html.P("Recall", style={'margin': '0'})
                        ], className='metric-box', style={
                            'backgroundColor': 'white', 
                            'padding': '20px', 
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], className='three columns'),
                    
                    html.Div([
                        html.Div([
                            html.H3("68%", style={'color': '#9b59b6', 'margin': '0'}),
                            html.P("Precision", style={'margin': '0'})
                        ], className='metric-box', style={
                            'backgroundColor': 'white', 
                            'padding': '20px', 
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], className='three columns'),
                ], className='row', style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
                
                # Performance charts
                html.Div([
                    html.Div([
                        dcc.Graph(id='roc-curve', figure=self._create_roc_curve())
                    ], className='six columns'),
                    
                    html.Div([
                        dcc.Graph(id='pr-curve', figure=self._create_pr_curve())
                    ], className='six columns'),
                ], className='row'),
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Feature Importance Section
            html.Div([
                html.H2("Feature Importance Analysis", style={'color': '#34495e'}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id='feature-importance', figure=self._create_feature_importance())
                    ], className='six columns'),
                    
                    html.Div([
                        dcc.Graph(id='shap-summary', figure=self._create_shap_summary())
                    ], className='six columns'),
                ], className='row'),
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # SDOH Impact Section
            html.Div([
                html.H2("Social Determinants of Health Impact", style={'color': '#34495e'}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id='sdoh-impact', figure=self._create_sdoh_impact())
                    ], className='six columns'),
                    
                    html.Div([
                        dcc.Graph(id='vulnerability-analysis', figure=self._create_vulnerability_analysis())
                    ], className='six columns'),
                ], className='row'),
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Risk Prediction Interface
            html.Div([
                html.H2("Patient Risk Assessment", style={'color': '#34495e'}),
                
                html.Div([
                    html.Div([
                        html.Label("Length of Stay"),
                        dcc.Slider(id='los-slider', min=1, max=30, value=5, 
                                  marks={i: str(i) for i in range(1, 31, 5)}),
                        
                        html.Label("Number of Prior Admissions"),
                        dcc.Slider(id='prior-admissions-slider', min=0, max=10, value=2),
                        
                        html.Label("Comorbidity Index"),
                        dcc.Slider(id='comorbidity-slider', min=0, max=5, value=2),
                    ], className='six columns'),
                    
                    html.Div([
                        html.Label("Social Vulnerability Index"),
                        dcc.Slider(id='svi-slider', min=0, max=1, value=0.5, step=0.1,
                                  marks={0: 'Low', 0.5: 'Medium', 1: 'High'}),
                        
                        html.Label("Economic Hardship Score"),
                        dcc.Slider(id='economic-slider', min=0, max=100, value=30),
                        
                        html.Label("Healthcare Access Barrier"),
                        dcc.Slider(id='access-slider', min=0, max=3, value=1),
                    ], className='six columns'),
                ], className='row'),
                
                html.Div([
                    html.Button('Calculate Risk', id='calculate-risk', n_clicks=0,
                               style={'backgroundColor': '#3498db', 'color': 'white', 
                                      'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px',
                                      'cursor': 'pointer', 'fontSize': '16px'}),
                ], style={'textAlign': 'center', 'marginTop': '20px'}),
                
                html.Div(id='risk-output', style={'marginTop': '20px', 'textAlign': 'center'})
                
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Fairness Analysis
            html.Div([
                html.H2("Fairness and Equity Analysis", style={'color': '#34495e'}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id='demographic-fairness', figure=self._create_demographic_fairness())
                    ], className='six columns'),
                    
                    html.Div([
                        dcc.Graph(id='regional-fairness', figure=self._create_regional_fairness())
                    ], className='six columns'),
                ], className='row'),
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px'}),
            
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'})
    
    def _create_roc_curve(self):
        """Create ROC curve figure"""
        # Sample data for demonstration
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Simulated ROC curve
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='XGBoost', line=dict(color='#3498db', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', 
                                line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title='ROC Curve (AUC = 0.78)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_pr_curve(self):
        """Create Precision-Recall curve figure"""
        # Sample data for demonstration
        recall = np.linspace(0, 1, 100)
        precision = 0.3 + 0.5 * (1 - recall)  # Simulated PR curve
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='XGBoost', 
                                line=dict(color='#e74c3c', width=3)))
        
        fig.update_layout(
            title='Precision-Recall Curve (AUC = 0.34)',
            xaxis_title='Recall',
            yaxis_title='Precision',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_feature_importance(self):
        """Create feature importance chart"""
        top_features = self.sample_data.head(10)
        
        fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                    color='category', color_discrete_map={
                        'Medical': '#3498db',
                        'SDOH': '#e74c3c', 
                        'Demographic': '#2ecc71'
                    })
        
        fig.update_layout(
            title='Top 10 Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def _create_shap_summary(self):
        """Create SHAP summary plot"""
        # Sample SHAP values for demonstration
        features = self.feature_names[:15]
        shap_values = np.random.uniform(-0.2, 0.2, 15)
        
        fig = go.Figure()
        
        # Positive impacts
        pos_mask = shap_values > 0
        fig.add_trace(go.Bar(
            y=[features[i] for i in range(len(features)) if pos_mask[i]],
            x=[shap_values[i] for i in range(len(features)) if pos_mask[i]],
            orientation='h',
            name='Increases Risk',
            marker_color='#e74c3c'
        ))
        
        # Negative impacts  
        neg_mask = shap_values < 0
        fig.add_trace(go.Bar(
            y=[features[i] for i in range(len(features)) if neg_mask[i]],
            x=[shap_values[i] for i in range(len(features)) if neg_mask[i]],
            orientation='h',
            name='Decreases Risk',
            marker_color='#3498db'
        ))
        
        fig.update_layout(
            title='SHAP Feature Impact',
            xaxis_title='SHAP Value (Impact on Prediction)',
            yaxis_title='Features',
            template='plotly_white',
            barmode='relative',
            showlegend=True
        )
        
        return fig
    
    def _create_sdoh_impact(self):
        """Create SDOH impact visualization"""
        sdoh_data = pd.DataFrame({
            'factor': ['Income', 'Education', 'Housing', 'Transportation', 'Social Support'],
            'impact': [0.15, 0.12, 0.08, 0.06, 0.09],
            'prevalence': [25, 18, 12, 8, 15]
        })
        
        fig = px.scatter(sdoh_data, x='prevalence', y='impact', size='impact', 
                        color='factor', hover_name='factor',
                        size_max=30, color_discrete_sequence=px.colors.qualitative.Set2)
        
        fig.update_layout(
            title='SDOH Factors: Impact vs Prevalence',
            xaxis_title='Prevalence in Population (%)',
            yaxis_title='Impact on Readmission Risk',
            template='plotly_white'
        )
        
        return fig
    
    def _create_vulnerability_analysis(self):
        """Create vulnerability analysis chart"""
        vulnerability_data = pd.DataFrame({
            'vulnerability_level': ['Low', 'Medium', 'High'],
            'readmission_rate': [8.2, 12.5, 18.7],
            'sample_size': [45000, 35000, 20000]
        })
        
        fig = px.bar(vulnerability_data, x='vulnerability_level', y='readmission_rate',
                    color='vulnerability_level', color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c'])
        
        fig.update_layout(
            title='Readmission Rates by Social Vulnerability Level',
            xaxis_title='Social Vulnerability Level',
            yaxis_title='Readmission Rate (%)',
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _create_demographic_fairness(self):
        """Create demographic fairness analysis"""
        fairness_data = pd.DataFrame({
            'group': ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
            'roc_auc': [0.78, 0.76, 0.77, 0.79, 0.75],
            'selection_rate': [12.5, 15.2, 14.8, 11.3, 13.7]
        })
        
        fig = px.scatter(fairness_data, x='selection_rate', y='roc_auc', 
                        size=fairness_data['roc_auc']*10, color='group',
                        hover_name='group', size_max=20)
        
        fig.update_layout(
            title='Model Performance Across Demographic Groups',
            xaxis_title='Selection Rate (%)',
            yaxis_title='ROC-AUC Score',
            template='plotly_white'
        )
        
        return fig
    
    def _create_regional_fairness(self):
        """Create regional fairness analysis"""
        regional_data = pd.DataFrame({
            'region': ['Northeast', 'Midwest', 'South', 'West'],
            'performance_gap': [0.02, -0.01, 0.03, -0.02],
            'readmission_rate': [13.2, 14.5, 15.8, 12.7]
        })
        
        fig = px.bar(regional_data, x='region', y='performance_gap',
                    color='performance_gap', color_continuous_scale='RdBu')
        
        fig.update_layout(
            title='Regional Performance Gaps',
            xaxis_title='Region',
            yaxis_title='Performance Gap vs Overall',
            template='plotly_white'
        )
        
        return fig
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            Output('risk-output', 'children'),
            [Input('calculate-risk', 'n_clicks')],
            [Input('los-slider', 'value'),
             Input('prior-admissions-slider', 'value'),
             Input('comorbidity-slider', 'value'),
             Input('svi-slider', 'value'),
             Input('economic-slider', 'value'),
             Input('access-slider', 'value')]
        )
        def calculate_risk(n_clicks, los, prior_admissions, comorbidity, svi, economic, access):
            if n_clicks > 0:
                # Simplified risk calculation for demonstration
                base_risk = 0.1
                risk_score = (base_risk + 
                            los * 0.02 + 
                            prior_admissions * 0.05 + 
                            comorbidity * 0.04 +
                            svi * 0.15 +
                            economic * 0.001 +
                            access * 0.03)
                
                risk_score = min(risk_score, 0.95)  # Cap at 95%
                
                risk_percentage = risk_score * 100
                
                if risk_score > 0.3:
                    color = '#e74c3c'
                    level = 'HIGH'
                    recommendation = "Consider care coordination and follow-up interventions"
                elif risk_score > 0.15:
                    color = '#f39c12' 
                    level = 'MEDIUM'
                    recommendation = "Schedule follow-up appointment and patient education"
                else:
                    color = '#2ecc71'
                    level = 'LOW'
                    recommendation = "Standard discharge process"
                
                return html.Div([
                    html.H3(f"Risk Score: {risk_percentage:.1f}%", 
                           style={'color': color, 'marginBottom': '10px'}),
                    html.H4(f"Risk Level: {level}", style={'color': color}),
                    html.P(recommendation, style={'fontSize': '16px', 'marginTop': '10px'})
                ], style={
                    'padding': '20px', 
                    'backgroundColor': '#f8f9fa', 
                    'borderRadius': '10px',
                    'border': f'2px solid {color}'
                })
            
            return html.Div("Adjust the sliders and click 'Calculate Risk' to see prediction.")
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)

# Usage
if __name__ == "__main__":
    # For demonstration, create a mock model and processor
    class MockModel:
        def predict_proba(self, X):
            return np.random.uniform(0, 1, (len(X), 2))
    
    class MockProcessor:
        def transform(self, X):
            return X
    
    # Create and run dashboard
    feature_names = [
        'length_of_stay', 'prior_admissions', 'comorbidity_index',
        'overall_vulnerability', 'economic_hardship', 'utilization_score',
        'num_medications', 'number_diagnoses', 'age', 'procedure_intensity',
        'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4'
    ]
    
    dashboard = ReadmissionDashboard(
        model=MockModel(),
        feature_names=feature_names,
        data_processor=MockProcessor()
    )
    
    print("Starting dashboard on http://localhost:8050")
    dashboard.run_server(debug=True)