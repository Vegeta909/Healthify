import streamlit as st
import pandas as pd
import numpy as np
import os
from prediction import predict_diabetes, predict_heart_attack, MODELS_DIR

# Set page configuration
st.set_page_config(
    page_title="HEALTHIFY - Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .high-risk {
        background-color: #FFEBEE;
        border: 2px solid #C62828;
        color: #C62828;
    }
    .low-risk {
        background-color: #E8F5E9;
        border: 2px solid #2E7D32;
        color: #2E7D32;
    }
    .result-box h3 {
        color: inherit;
        font-weight: bold;
    }
    .result-box p {
        color: #333333;
        font-size: 1.1rem;
    }
    .stTable {
        width: 100%;
    }
    .model-comparison {
        margin-top: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Display header
try:
    st.image("static/images/Healthi-fy.png", width=250)
except:
    st.title("HEALTHIFY")
    
st.markdown("<h1 class='main-header'>Smart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "HEALTHIFY is a Big Data-driven application that predicts diabetes and heart attack risk "
    "using machine learning models trained on real-world health datasets. "
    "This application uses Apache Spark for distributed computing and machine learning."
)

st.sidebar.title("Technologies Used")
st.sidebar.markdown(
    """
    - Apache Spark
    - PySpark MLlib
    - Streamlit
    - Python
    """
)

# Add model status section to diagnose model loading issues
st.sidebar.title("Model Status")
model_paths = [
    os.path.join(MODELS_DIR, "diabetes_model_lr"),
    os.path.join(MODELS_DIR, "diabetes_model_rf"),
    os.path.join(MODELS_DIR, "heart_model_lr"),
    os.path.join(MODELS_DIR, "heart_model_rf"),
    os.path.join(MODELS_DIR, "best_models.txt")
]

for path in model_paths:
    if os.path.exists(path):
        st.sidebar.success(f"✅ {os.path.basename(path)} exists")
    else:
        st.sidebar.error(f"❌ {os.path.basename(path)} missing")

# Create tabs
tab1, tab2 = st.tabs(["Diabetes Prediction", "Heart Attack Prediction"])

# Diabetes Prediction Tab
with tab1:
    st.markdown("<h2 class='sub-header'>Diabetes Prediction</h2>", unsafe_allow_html=True)
    st.write("Enter the patient's details to predict diabetes risk.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age (Years)", 0, 120, 30)
    
    if st.button("Predict Diabetes Risk"):
        with st.spinner("Analyzing data..."):
            # Create feature vector
            features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            
            # Make prediction
            try:
                results = predict_diabetes(features)
                
                # Create a comparison table
                st.subheader("Model Predictions")
                
                # Check if we have machine learning model results
                has_ml_results = (
                    results.get("logistic_regression", {}).get("prediction") is not None or 
                    results.get("random_forest", {}).get("prediction") is not None
                )
                
                if has_ml_results:
                    # Create a dataframe for model comparison
                    comparison_data = {
                        "Model": [], 
                        "Prediction": [], 
                        "Risk Level": [], 
                        "Probability": []
                    }
                    
                    # Add logistic regression results if available
                    if results.get("logistic_regression", {}).get("prediction") is not None:
                        lr_prediction = results["logistic_regression"]["prediction"]
                        lr_probability = results["logistic_regression"]["probability"]
                        comparison_data["Model"].append("Logistic Regression")
                        comparison_data["Prediction"].append("Positive" if lr_prediction == 1 else "Negative")
                        comparison_data["Risk Level"].append("High Risk" if lr_prediction == 1 else "Low Risk")
                        comparison_data["Probability"].append(f"{lr_probability:.1%}")
                    
                    # Add random forest results if available
                    if results.get("random_forest", {}).get("prediction") is not None:
                        rf_prediction = results["random_forest"]["prediction"]
                        rf_probability = results["random_forest"]["probability"]
                        comparison_data["Model"].append("Random Forest")
                        comparison_data["Prediction"].append("Positive" if rf_prediction == 1 else "Negative")
                        comparison_data["Risk Level"].append("High Risk" if rf_prediction == 1 else "Low Risk")
                        comparison_data["Probability"].append(f"{rf_probability:.1%}")
                    
                    # Create and display the comparison dataframe
                    comparison_df = pd.DataFrame(comparison_data)
                    st.markdown("<div class='model-comparison'>", unsafe_allow_html=True)
                    st.table(comparison_df)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Determine consensus result
                    high_risk_count = sum(1 for model_name in ["logistic_regression", "random_forest"] 
                                       if model_name in results 
                                       and results[model_name].get("prediction") == 1)
                    
                    # Calculate average probability from available models
                    probabilities = [results[model_name]["probability"] 
                                   for model_name in ["logistic_regression", "random_forest"]
                                   if model_name in results and results[model_name].get("probability") is not None]
                    avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0.5
                    
                    # Display the final recommendation based on consensus
                    if high_risk_count > 0 or avg_probability >= 0.3:
                        st.markdown(
                            f"""
                            <div class='result-box high-risk'>
                                <h3>⚠️ High Risk of Diabetes</h3>
                                <p>The models predict an average {avg_probability:.1%} probability of diabetes.</p>
                                <p>Recommendation: Further clinical tests are recommended.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class='result-box low-risk'>
                                <h3>✅ Low Risk of Diabetes</h3>
                                <p>The models predict an average {avg_probability:.1%} probability of diabetes.</p>
                                <p>Recommendation: Maintain a healthy lifestyle and regular check-ups.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                else:
                    # Use fallback results if ML models failed
                    fallback_result = results.get("fallback", {}).get("prediction", 0)
                    fallback_probability = results.get("fallback", {}).get("probability", 0.5)
                    
                    st.warning("Machine learning models could not be loaded. Using fallback prediction rules.")
                    
                    if fallback_result == 1 or fallback_probability >= 0.3:
                        st.markdown(
                            f"""
                            <div class='result-box high-risk'>
                                <h3>⚠️ High Risk of Diabetes (Fallback Prediction)</h3>
                                <p>The fallback prediction indicates a {fallback_probability:.1%} probability of diabetes.</p>
                                <p>Recommendation: Further clinical tests are recommended.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class='result-box low-risk'>
                                <h3>✅ Low Risk of Diabetes (Fallback Prediction)</h3>
                                <p>The fallback prediction indicates a {fallback_probability:.1%} probability of diabetes.</p>
                                <p>Recommendation: Maintain a healthy lifestyle and regular check-ups.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please make sure the models are properly exported.")

# Heart Attack Prediction Tab
with tab2:
    st.markdown("<h2 class='sub-header'>Heart Attack Risk Prediction</h2>", unsafe_allow_html=True)
    st.write("Enter the patient's details to predict heart attack risk.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (Years)", 0, 120, 40, key="heart_age")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        obesity = st.selectbox("Obesity", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        alcohol = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        physical_activity = st.selectbox("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diet_score = st.slider("Diet Score", 0, 10, 5)
        cholesterol = st.number_input("Cholesterol Level", 100, 300, 200)
        triglycerides = st.number_input("Triglyceride Level", 50, 300, 150)
        ldl = st.number_input("LDL Level", 50, 200, 100)
        hdl = st.number_input("HDL Level", 20, 100, 50)
    
    with col3:
        systolic_bp = st.number_input("Systolic BP", 90, 200, 120)
        diastolic_bp = st.number_input("Diastolic BP", 60, 120, 80)
        pollution = st.selectbox("Air Pollution Exposure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        family_history = st.selectbox("Family History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        stress = st.slider("Stress Level", 1, 10, 5)
        healthcare = st.selectbox("Healthcare Access", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    
    if st.button("Predict Heart Attack Risk"):
        with st.spinner("Analyzing data..."):
            # Create feature vector
            features = [age, gender, diabetes, hypertension, obesity, smoking, alcohol, 
                       physical_activity, diet_score, cholesterol, triglycerides, ldl, hdl, 
                       systolic_bp, diastolic_bp, pollution, family_history, stress, healthcare]
            
            # Make prediction
            try:
                results = predict_heart_attack(features)
                
                # Create a comparison table
                st.subheader("Model Predictions")
                
                # Check if we have machine learning model results
                has_ml_results = (
                    results.get("logistic_regression", {}).get("prediction") is not None or 
                    results.get("random_forest", {}).get("prediction") is not None
                )
                
                if has_ml_results:
                    # Create a dataframe for model comparison
                    comparison_data = {
                        "Model": [], 
                        "Prediction": [], 
                        "Risk Level": [], 
                        "Probability": []
                    }
                    
                    # Add logistic regression results if available
                    if results.get("logistic_regression", {}).get("prediction") is not None:
                        lr_prediction = results["logistic_regression"]["prediction"]
                        lr_probability = results["logistic_regression"]["probability"]
                        comparison_data["Model"].append("Logistic Regression")
                        comparison_data["Prediction"].append("Positive" if lr_prediction == 1 else "Negative")
                        comparison_data["Risk Level"].append("High Risk" if lr_prediction == 1 else "Low Risk")
                        comparison_data["Probability"].append(f"{lr_probability:.1%}")
                    
                    # Add random forest results if available
                    if results.get("random_forest", {}).get("prediction") is not None:
                        rf_prediction = results["random_forest"]["prediction"]
                        rf_probability = results["random_forest"]["probability"]
                        comparison_data["Model"].append("Random Forest")
                        comparison_data["Prediction"].append("Positive" if rf_prediction == 1 else "Negative")
                        comparison_data["Risk Level"].append("High Risk" if rf_prediction == 1 else "Low Risk")
                        comparison_data["Probability"].append(f"{rf_probability:.1%}")
                    
                    # Create and display the comparison dataframe
                    comparison_df = pd.DataFrame(comparison_data)
                    st.markdown("<div class='model-comparison'>", unsafe_allow_html=True)
                    st.table(comparison_df)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Determine consensus result
                    high_risk_count = sum(1 for model_name in ["logistic_regression", "random_forest"] 
                                       if model_name in results 
                                       and results[model_name].get("prediction") == 1)
                    
                    # Calculate average probability from available models
                    probabilities = [results[model_name]["probability"] 
                                   for model_name in ["logistic_regression", "random_forest"]
                                   if model_name in results and results[model_name].get("probability") is not None]
                    avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0.5
                    
                    # Display the final recommendation based on consensus
                    if high_risk_count > 0 or avg_probability >= 0.3:
                        st.markdown(
                            f"""
                            <div class='result-box high-risk'>
                                <h3>⚠️ High Risk of Heart Attack</h3>
                                <p>The models predict an average {avg_probability:.1%} probability of heart attack risk.</p>
                                <p>Recommendation: Immediate consultation with a cardiologist is advised.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class='result-box low-risk'>
                                <h3>✅ Low Risk of Heart Attack</h3>
                                <p>The models predict an average {avg_probability:.1%} probability of heart attack risk.</p>
                                <p>Recommendation: Continue with regular health check-ups and maintain a healthy lifestyle.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                else:
                    # Use fallback results if ML models failed
                    fallback_result = results.get("fallback", {}).get("prediction", 0)
                    fallback_probability = results.get("fallback", {}).get("probability", 0.5)
                    
                    st.warning("Machine learning models could not be loaded. Using fallback prediction rules.")
                    
                    if fallback_result == 1 or fallback_probability >= 0.3:
                        st.markdown(
                            f"""
                            <div class='result-box high-risk'>
                                <h3>⚠️ High Risk of Heart Attack (Fallback Prediction)</h3>
                                <p>The fallback prediction indicates a {fallback_probability:.1%} probability of heart attack risk.</p>
                                <p>Recommendation: Immediate consultation with a cardiologist is advised.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class='result-box low-risk'>
                                <h3>✅ Low Risk of Heart Attack (Fallback Prediction)</h3>
                                <p>The fallback prediction indicates a {fallback_probability:.1%} probability of heart attack risk.</p>
                                <p>Recommendation: Continue with regular health check-ups and maintain a healthy lifestyle.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please make sure the models are properly exported.")