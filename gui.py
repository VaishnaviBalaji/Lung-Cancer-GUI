import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import urllib.request
import io
from datetime import date
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Immunotherapy Response Prediction", layout="wide")

@st.cache_data  # Use cache_data for caching in Streamlit Cloud
def load_data():
    return pd.read_csv('data.csv')  # Ensure data.csv is in the root directory

df = load_data()

def preprocess_data(df):
    df = df.dropna(subset=['Immunotherapy Response'])
    response_mapping = {'PR': 1, 'SD': 1, 'PD': 0}
    df['Immunotherapy Response'] = df['Immunotherapy Response'].map(response_mapping)
    df = df[df['Immunotherapy Response'].notna()]
    
    categorical_cols = ['Gender', 'EAS Regimen', 'Intent', 'Histology Type']
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col]).codes
    
    numerical_cols = ['age', 'T', 'N', 'M', 'EAS Performance status']
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    return df

def create_pdf(input_data, prediction, probability, feature_importance, shap_values, input_df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    logo_url = "https://images.squarespace-cdn.com/content/v1/5da2ef343b1ad128da926c53/1571314611701-SGDM0RAIYQQSKRKOHG32/curenetics_logo_t.png?format=1500w"
    logo_data = urllib.request.urlopen(logo_url).read()
    logo_img = Image(BytesIO(logo_data), width=2*inch, height=1*inch)
    
    logo_table = Table([[logo_img]], colWidths=[6*inch])
    logo_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'RIGHT')]))
    story.append(logo_table)
    
    story.append(Spacer(1, 12))
    story.append(Paragraph("Lung Cancer Immunotherapy Response Prediction Report", styles['Title']))
    
    today= date.today()
    formatted_date = today.strftime("%B %d, %Y")
    date_paragraph = Paragraph(f"Date: {formatted_date}", styles['Heading2'])
    story.append(date_paragraph)
    
    if 'clinician_info' in st.session_state and st.session_state.clinician_info is not None:
        story.append(Paragraph("Clinician Information:", styles['Heading2']))
        for key, value in st.session_state.clinician_info.items():
            story.append(Paragraph(f"{key.capitalize()}: {value}", styles['Normal']))
        story.append(Spacer(1, 15))
    
    # Add input data and prediction results...
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main_app():
    st.title('Lung Cancer Immunotherapy Response Prediction')
    
    # Sidebar content...
    
    df_processed = preprocess_data(df)
    
    X = df_processed.drop(['Immunotherapy Response', 'ICD code'], axis=1)
    y = df_processed['Immunotherapy Response']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0.1, 0.9)
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                       n_iter=100, cv=5,
                                       scoring='f1', n_jobs=-1,
                                       random_state=42)
                                       
    random_search.fit(X_train, y_train)
    
    model = random_search.best_estimator_
    
    # Continue with prediction and SHAP analysis...

if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
