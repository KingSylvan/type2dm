import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime
import sqlite3
import json
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="T2DM Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    .feature-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Database functions
def init_db():
    """Initialize SQLite database for storing predictions"""
    conn = sqlite3.connect('db/predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  age INTEGER,
                  sex TEXT,
                  bmi REAL,
                  waist REAL,
                  smoking_status TEXT,
                  alcohol_freq TEXT,
                  moderate_mins INTEGER,
                  vigorous_mins INTEGER,
                  sleep_hrs REAL,
                  fruit_intake INTEGER,
                  veg_intake INTEGER,
                  red_meat_intake INTEGER,
                  predicted_risk REAL,
                  risk_category TEXT,
                  lifestyle_score INTEGER)''')
    conn.commit()
    conn.close()

def save_prediction(user_data, risk_prob, risk_category, lifestyle_score):
    """Save prediction to database"""
    conn = sqlite3.connect('db/predictions.db')
    c = conn.cursor()
    c.execute('''INSERT INTO predictions 
                 (timestamp, age, sex, bmi, waist, smoking_status, alcohol_freq,
                  moderate_mins, vigorous_mins, sleep_hrs, fruit_intake, veg_intake,
                  red_meat_intake, predicted_risk, risk_category, lifestyle_score)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               user_data['age'], user_data['sex'], user_data['bmi'], user_data['waist'],
               user_data['smoking_status'], user_data['alcohol_freq'],
               user_data['moderate_mins'], user_data['vigorous_mins'], user_data['sleep_hrs'],
               user_data['fruit_intake'], user_data['veg_intake'], user_data['red_meat_intake'],
               risk_prob, risk_category, lifestyle_score))
    conn.commit()
    conn.close()

def fetch_recent_predictions(limit=10):
    """Fetch recent predictions from database"""
    conn = sqlite3.connect('db/predictions.db')
    df = pd.read_sql_query(f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    return df

# Feature engineering functions
def calculate_lifestyle_score(user_data):
    """Calculate composite lifestyle score (0-10 points)"""
    score = 0
    
    # BMI score (0-2 points)
    if user_data['bmi'] < 25:
        score += 2
    elif user_data['bmi'] < 30:
        score += 1
    
    # Physical activity score (0-2 points)
    total_activity = user_data['moderate_mins'] + user_data['vigorous_mins'] * 2
    if total_activity >= 150:
        score += 2
    elif total_activity >= 75:
        score += 1
    
    # Smoking score (0-2 points)
    if user_data['smoking_status'] == 'Never':
        score += 2
    elif user_data['smoking_status'] == 'Previous':
        score += 1
    
    # Diet score (0-2 points)
    diet_score = (user_data['fruit_intake'] + user_data['veg_intake']) / 2
    if diet_score > 3:
        score += 2
    elif diet_score > 1.5:
        score += 1
    
    # Sleep score (0-2 points)
    if 7 <= user_data['sleep_hrs'] <= 9:
        score += 2
    elif 6 <= user_data['sleep_hrs'] <= 10:
        score += 1
    
    return score

def get_risk_category(risk_prob):
    """Categorize risk based on probability"""
    if risk_prob >= 0.5:
        return "High"
    elif risk_prob >= 0.4:
        return "Medium"
    else:
        return "Low"

def get_risk_color(risk_category):
    """Get color for risk category"""
    colors = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}
    return colors.get(risk_category, "#000000")

# Load the trained pipeline
MODEL_PATH = 'models/t2dm_RF_d12_F1-0.149.joblib'
try:
    # Try loading with different protocols for compatibility
    import pickle
    
    # First try with joblib
    try:
        model_bundle = joblib.load(MODEL_PATH)
    except Exception as joblib_error:
        # If joblib fails, try with pickle
        st.warning(f"Joblib loading failed: {joblib_error}. Trying alternative loading method...")
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_bundle = pickle.load(f)
        except Exception as pickle_error:
            raise Exception(f"Both joblib and pickle loading failed. Joblib: {joblib_error}, Pickle: {pickle_error}")
    
    model_pipeline = model_bundle['pipeline']
    model_threshold = model_bundle.get('threshold', 0.5)
    
    # Test the model with dummy data to ensure compatibility
    dummy_data = pd.DataFrame({
        'Pregnancies': [1],
        'Glucose': [100],
        'BloodPressure': [70],
        'SkinThickness': [20],
        'Insulin': [80],
        'BMI': [25.0],
        'DiabetesPedigreeFunction': [0.5],
        'Age': [30]
    })
    
    # Try a prediction to verify compatibility
    _ = model_pipeline.predict_proba(dummy_data)
    st.success("✅ Model loaded successfully and compatibility verified!")
    
except Exception as e:
    model_pipeline = None
    st.error(f"❌ Could not load model pipeline: {e}")
    st.error("📋 **Possible solutions:**")
    st.error("1. Model was trained with a different scikit-learn version")
    st.error("2. Try retraining the model with current environment versions")
    st.error("3. Check if all required packages are installed with correct versions")

# Main app
def main():
    # Initialize database
    init_db()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Type 2 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #666;'>
            Early detection and prevention of Type 2 Diabetes Mellitus using machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("""
        **Model Type:** Random Forest Classifier
        **Training Data:** UK Biobank (synthetic)
        **Target Population:** Adults 40-69 years
        **Features:** Lifestyle, demographic, and genetic factors
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Your Information")
        
        # Create form for user input
        with st.form("t2dm_prediction_form"):
            # Demographics section
            st.subheader("👤 Demographics")
            col1_demo, col2_demo = st.columns(2)
            
            with col1_demo:
                age = st.slider("Age (years)", 40, 69, 50, help="Age range: 40-69 years")
                sex = st.selectbox("Sex", ["Female", "Male"], help="Biological sex")
            
            with col2_demo:
                bmi = st.number_input("BMI (kg/m²)", 18.0, 50.0, 25.0, 0.1, 
                                    help="Body Mass Index: Underweight <18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese ≥30")
                waist = st.number_input("Waist Circumference (cm)", 60.0, 150.0, 90.0, 0.5,
                                      help="Measure around the narrowest part of your waist")
            
            # Lifestyle section
            st.subheader("🏃‍♂️ Lifestyle Factors")
            col1_life, col2_life = st.columns(2)
            
            with col1_life:
                smoking_status = st.selectbox("Smoking Status", 
                                            ["Never", "Previous", "Current"],
                                            help="Current smoking status")
                alcohol_freq = st.selectbox("Alcohol Frequency",
                                          ["Never", "Occasional", "Monthly", "Weekly_1_2", "Weekly_3_4", "Daily"],
                                          help="How often do you consume alcohol?")
            
            with col2_life:
                moderate_mins = st.number_input("Moderate Activity (mins/week)", 0, 1000, 150,
                                              help="Moderate intensity physical activity (e.g., brisk walking)")
                vigorous_mins = st.number_input("Vigorous Activity (mins/week)", 0, 1000, 75,
                                              help="Vigorous intensity physical activity (e.g., running, cycling)")
            
            # Health behaviors section
            st.subheader("💤 Health Behaviors")
            col1_health, col2_health = st.columns(2)
            
            with col1_health:
                sleep_hrs = st.number_input("Sleep Duration (hours/night)", 4.0, 12.0, 7.5, 0.25,
                                          help="Average hours of sleep per night")
            
            with col2_health:
                fruit_intake = st.selectbox("Fruit Intake", [0, 1, 2, 3, 4, 5, 6],
                                          format_func=lambda x: f"{x} servings/day",
                                          help="Daily fruit consumption")
            
            # Diet section
            st.subheader("🥗 Dietary Habits")
            col1_diet, col2_diet = st.columns(2)
            
            with col1_diet:
                veg_intake = st.selectbox("Vegetable Intake", [0, 1, 2, 3, 4, 5, 6],
                                        format_func=lambda x: f"{x} servings/day",
                                        help="Daily vegetable consumption")
            
            with col2_diet:
                red_meat_intake = st.selectbox("Red Meat Intake", [0, 1, 2, 3, 4, 5],
                                             format_func=lambda x: f"{x} servings/week",
                                             help="Weekly red meat consumption")
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict T2DM Risk", 
                                            help="Click to get your personalized risk assessment")
    
    with col2:
        st.header("📊 Results")
        
        if submitted:
            # Prepare user data
            user_data = {
                'age': age,
                'sex': 1 if sex == 'Male' else 0,
                'bmi': bmi,
                'waist': waist,
                'smoking_status': {"Never": 0, "Previous": 1, "Current": 2}.get(smoking_status, 3),
                'alcohol_freq': {"Daily": 1, "Weekly_3_4": 2, "Weekly_1_2": 3, "Monthly": 4, "Occasional": 5, "Never": 6}.get(alcohol_freq, 0),
                'moderate_mins': moderate_mins,
                'vigorous_mins': vigorous_mins,
                'sleep_hrs': sleep_hrs,
                'fruit_intake': fruit_intake,
                'veg_intake': veg_intake,
                'red_meat_intake': red_meat_intake
            }
            
            # Calculate lifestyle score
            lifestyle_score = calculate_lifestyle_score({
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'waist': waist,
                'smoking_status': smoking_status,
                'alcohol_freq': alcohol_freq,
                'moderate_mins': moderate_mins,
                'vigorous_mins': vigorous_mins,
                'sleep_hrs': sleep_hrs,
                'fruit_intake': fruit_intake,
                'veg_intake': veg_intake,
                'red_meat_intake': red_meat_intake
            })
            
            # Model-based prediction
            if model_pipeline is not None:
                # Create DataFrame with correct columns and order
                input_df = pd.DataFrame([user_data])
                # Fill missing columns if any (for robustness)
                for col in model_pipeline.feature_names_in_:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[model_pipeline.feature_names_in_]
                try:
                    risk_prob = float(model_pipeline.predict_proba(input_df)[0, 1])
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    risk_prob = 0.0
            else:
                st.error("Model pipeline not loaded. Showing 0% risk.")
                risk_prob = 0.0
            
            # Clamp risk probability
            risk_prob = min(max(risk_prob, 0.01), 0.99)
            
            # Get risk category
            risk_category = get_risk_category(risk_prob)
            risk_color = get_risk_color(risk_category)
            
            # Display results
            st.markdown(f"""
            <div class="metric-card">
                <h3>Your T2DM Risk Assessment</h3>
                <p style="font-size: 2rem; font-weight: bold; color: {risk_color};">
                    {risk_prob:.1%}
                </p>
                <p style="font-size: 1.2rem; font-weight: bold; color: {risk_color};">
                    Risk Level: {risk_category}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Lifestyle score
            st.markdown(f"""
            <div class="metric-card">
                <h4>Lifestyle Score: {lifestyle_score}/10</h4>
                <p>Higher scores indicate healthier lifestyle choices</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Save to database
            save_prediction(user_data, risk_prob, risk_category, lifestyle_score)
            
            # Recommendations
            st.subheader("💡 Personalized Recommendations")
            if risk_category == "High":
                st.error("""
                **High Risk - Immediate Action Recommended:**
                • Consult with your healthcare provider
                • Consider lifestyle modification programs
                • Monitor blood glucose levels regularly
                • Focus on weight management and physical activity
                """)
            elif risk_category == "Medium":
                st.warning("""
                **Medium Risk - Preventive Measures:**
                • Increase physical activity to 150+ minutes/week
                • Improve diet with more fruits and vegetables
                • Maintain healthy weight
                • Regular health check-ups
                """)
            else:
                st.success("""
                **Low Risk - Maintain Healthy Habits:**
                • Continue current healthy lifestyle
                • Regular physical activity
                • Balanced diet
                • Annual health check-ups
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>This tool is for educational and research purposes only. 
        Always consult healthcare professionals for medical advice.</p>
        <p>Built with ❤️ using Streamlit and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
    """
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 2rem;'>
        <p style='margin: 0; color: #262730; font-size: 14px;'>
        <strong>🚀 Powered by Sylvanus Chinedu Egbosiuba</strong><br>
        <a href='mailto:egbosiubasylvanus@gmail.com' style='color: #ff6b6b; text-decoration: none;'>
        📧 egbosiubasylvanus@gmail.com</a>
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main() 