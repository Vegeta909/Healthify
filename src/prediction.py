from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.ml.linalg import Vectors
import os
import sys

HEALTHIFY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(HEALTHIFY_DIR, "models")

# Now use these paths in your functions
def initialize_spark():
    """Initialize Spark session"""
    return SparkSession.builder \
        .appName("HEALTHIFY - Prediction") \
        .config("spark.driver.memory", "2g") \
        .config("spark.python.worker.timeout", "600") \
        .getOrCreate()

def create_feature_vector(spark, features):
    """Create a DataFrame with feature vector"""
    # Create DataFrame with feature vector
    data = [(Vectors.dense(features),)]
    return spark.createDataFrame(data, ["features"])

def fallback_diabetes_prediction(features):
    """Rule-based fallback prediction when Spark fails"""
    try:
        # Extract important features
        pregnancies = features[0]  # Number of pregnancies
        glucose = features[1]      # Glucose level
        blood_pressure = features[2]  # Blood pressure
        skin_thickness = features[3]  # Skin thickness
        insulin = features[4]      # Insulin level
        bmi = features[5]          # BMI
        dpf = features[6]          # Diabetes pedigree function
        age = features[7]          # Age
        
        # Check if these are default values from the UI
        is_default_input = (pregnancies == 0 and glucose == 120 and blood_pressure == 70 and 
                         skin_thickness == 20 and insulin == 80 and 
                         bmi == 25.0 and dpf == 0.5 and age == 30)
        
        if is_default_input:
            print("Default values detected, providing conservative estimate")
            # For default values, give a conservative estimate
            return 0, 0.20
    except Exception as e:
        print(f"Error extracting features: {e}")
        # If any error in feature extraction, give conservative estimate
        return 0, 0.20
    
    # Start with a base probability
    probability = 0.1
    
    # Glucose has the strongest correlation with diabetes
    if glucose <= 90:
        probability += 0.05
    elif glucose <= 110:
        probability += 0.15
    elif glucose <= 125:
        probability += 0.3
    elif glucose <= 140:
        probability += 0.45
    elif glucose <= 160:
        probability += 0.55
    elif glucose <= 180:
        probability += 0.65
    else:
        probability += 0.75
    
    # BMI impact
    if bmi < 18.5:
        probability += 0.05  # Underweight - slight increase
    elif bmi < 25:
        probability += 0.0   # Normal weight - no change
    elif bmi < 30:
        probability += 0.1   # Overweight - moderate increase
    elif bmi < 35:
        probability += 0.15  # Obesity class I - significant increase
    elif bmi < 40:
        probability += 0.2   # Obesity class II - high increase
    else:
        probability += 0.25  # Obesity class III - very high increase
    
    # Age risk adjustment
    if age < 35:
        probability -= 0.05  # Lower risk for young age
    elif age < 45:
        probability += 0.0   # Neutral for middle age
    elif age < 55:
        probability += 0.1   # Increased risk for age 45-54
    elif age < 65:
        probability += 0.15  # Higher risk for age 55-64
    else:
        probability += 0.2   # Highest risk for age 65+
    
    # Diabetes Pedigree Function impact (family history)
    if dpf > 0.8:
        probability += 0.15  # Strong family history
    elif dpf > 0.5:
        probability += 0.07  # Moderate family history
    
    # Blood pressure impact
    if blood_pressure > 140:
        probability += 0.1   # High blood pressure increases risk
    
    # Insulin levels impact
    if insulin < 15 and glucose > 125:
        probability += 0.2   # Low insulin with high glucose suggests insulin resistance
    
    # Ensure probability is between 0.05 and 0.95
    probability = max(0.05, min(0.95, probability))
    
    # Determine result based on final probability
    result = 1 if probability >= 0.5 else 0
    
    return result, probability

def fallback_heart_prediction(features):
    """Rule-based fallback prediction for heart attack risk"""
    try:
        # Extract important features
        age = features[0] if len(features) > 0 else 50
        gender = features[1] if len(features) > 1 else 0     # 1 for male, 0 for female
        diabetes = features[2] if len(features) > 2 else 0   # 1 for yes, 0 for no
        hypertension = features[3] if len(features) > 3 else 0  # 1 for yes, 0 for no
        obesity = features[4] if len(features) > 4 else 0    # 1 for yes, 0 for no
        smoking = features[5] if len(features) > 5 else 0    # 1 for yes, 0 for no
        alcohol = features[6] if len(features) > 6 else 0    # 1 for yes, 0 for no
        physical_activity = features[7] if len(features) > 7 else 0  # 1 for yes, 0 for no
        diet_score = features[8] if len(features) > 8 else 5  # Diet score
        
        # Check if these are default values from the UI
        is_default_input = (age == 40 and gender == 0 and 
                         diabetes == 0 and hypertension == 0 and
                         obesity == 0 and smoking == 0 and
                         alcohol == 0 and physical_activity == 0 and
                         diet_score == 5)
        
        if is_default_input:
            print("Default values detected, providing conservative estimate")
            # For default values, give a conservative estimate
            return 0, 0.20
        
        # Get cholesterol related values if available
        cholesterol = features[9] if len(features) > 9 else 200
        hdl = features[12] if len(features) > 12 else 50
        ldl = features[11] if len(features) > 11 else 100
        
        # Get blood pressure if available
        systolic_bp = features[13] if len(features) > 13 else 120
        diastolic_bp = features[14] if len(features) > 14 else 80
        
        # Get other risk factors
        family_history = features[16] if len(features) > 16 else 0  # 1 for yes, 0 for no
        stress = features[17] if len(features) > 17 else 5  # 1-10 scale
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Default values if extraction fails
        return 0, 0.20
    
    # Start with a base probability
    probability = 0.15
    
    # Age impact (one of the strongest predictors)
    if age < 40:
        probability -= 0.05
    elif age < 50:
        probability += 0.05
    elif age < 60:
        probability += 0.15
    elif age < 70:
        probability += 0.25
    else:
        probability += 0.3
    
    # Gender impact (males have higher risk)
    if gender == 1:  # Male
        probability += 0.1
    
    # Diabetes impact
    if diabetes == 1:
        probability += 0.15
    
    # Hypertension impact
    if hypertension == 1:
        probability += 0.15
    
    # Obesity impact
    if obesity == 1:
        probability += 0.1
    
    # Smoking impact (major risk factor)
    if smoking == 1:
        probability += 0.2
    
    # Cholesterol impact
    if cholesterol > 240:
        probability += 0.15
    elif cholesterol > 200:
        probability += 0.1
    
    # HDL impact (good cholesterol - higher is better)
    if hdl < 40:
        probability += 0.1
    elif hdl > 60:
        probability -= 0.05
    
    # LDL impact (bad cholesterol)
    if ldl > 160:
        probability += 0.15
    elif ldl > 130:
        probability += 0.1
    
    # Blood pressure impact
    if systolic_bp >= 140 or diastolic_bp >= 90:
        probability += 0.15
    elif systolic_bp >= 130 or diastolic_bp >= 85:
        probability += 0.1
    
    # Family history impact
    if family_history == 1:
        probability += 0.15
    
    # Stress level impact
    if stress > 7:
        probability += 0.1
    
    # Physical activity impact (protective)
    if physical_activity == 1:
        probability -= 0.1
    
    # Ensure probability is between 0.05 and 0.95
    probability = max(0.05, min(0.95, probability))
    
    # Determine result based on probability
    result = 1 if probability >= 0.5 else 0
    
    return result, probability

def predict_diabetes(features):
    """Predict diabetes using both logistic regression and random forest models"""
    # Results dictionary with default values
    results = {
        "logistic_regression": {"prediction": None, "probability": None},
        "random_forest": {"prediction": None, "probability": None},
        "fallback": {"prediction": None, "probability": None}
    }
    
    try:
        # Initialize Spark
        spark = initialize_spark()
        
        try:
            # Create feature vector once for both models
            df = create_feature_vector(spark, features)
            
            # Try logistic regression model
            try:
                lr_model_path = os.path.join(MODELS_DIR, "diabetes_model_lr")
                print(f"Loading LR model from: {lr_model_path}")
                lr_model = LogisticRegressionModel.load(lr_model_path)
                lr_prediction = lr_model.transform(df)
                lr_result = lr_prediction.select("prediction").collect()[0][0]
                lr_prob = lr_prediction.select("probability").collect()[0][0][1]
                
                # Use a lower threshold for high risk
                lr_adjusted_result = 1 if lr_prob >= 0.3 else 0
                
                results["logistic_regression"]["prediction"] = int(lr_adjusted_result)
                results["logistic_regression"]["probability"] = float(lr_prob)
                print("Logistic Regression prediction successful")
            except Exception as e:
                print(f"Error with logistic regression model: {str(e)}")
            
            # Try random forest model
            try:
                rf_model_path = os.path.join(MODELS_DIR, "diabetes_model_rf")
                print(f"Loading RF model from: {rf_model_path}")
                rf_model = RandomForestClassificationModel.load(rf_model_path)
                rf_prediction = rf_model.transform(df)
                rf_result = rf_prediction.select("prediction").collect()[0][0]
                rf_prob = rf_prediction.select("probability").collect()[0][0][1]
                
                # Use a lower threshold for high risk
                rf_adjusted_result = 1 if rf_prob >= 0.3 else 0
                
                results["random_forest"]["prediction"] = int(rf_adjusted_result)
                results["random_forest"]["probability"] = float(rf_prob)
                print("Random Forest prediction successful")
            except Exception as e:
                print(f"Error with random forest model: {str(e)}")
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
        
        finally:
            # Stop Spark session
            try:
                spark.stop()
                print("Spark session stopped")
            except:
                print("Error stopping Spark session")
            
    except Exception as e:
        print(f"Spark initialization failed: {str(e)}")
    
    # If both model predictions failed, use fallback
    if results["logistic_regression"]["prediction"] is None and results["random_forest"]["prediction"] is None:
        try:
            print("Using fallback prediction method")
            fallback_result, fallback_prob = fallback_diabetes_prediction(features)
            results["fallback"]["prediction"] = fallback_result
            results["fallback"]["probability"] = fallback_prob
            print(f"Fallback prediction: {fallback_result}, {fallback_prob}")
        except Exception as e:
            print(f"Error in fallback prediction: {str(e)}")
            # Set some default values as absolute last resort
            results["fallback"]["prediction"] = 0
            results["fallback"]["probability"] = 0.5
            print("Using default values due to error in fallback")
    
    return results

def predict_heart_attack(features):
    """Predict heart attack risk using both logistic regression and random forest models"""
    # Results dictionary with default values
    results = {
        "logistic_regression": {"prediction": None, "probability": None},
        "random_forest": {"prediction": None, "probability": None},
        "fallback": {"prediction": None, "probability": None}
    }
    
    try:
        # Initialize Spark
        spark = initialize_spark()
        
        try:
            # Create feature vector once for both models
            df = create_feature_vector(spark, features)
            
            # Try logistic regression model
            try:
                lr_model_path = os.path.join(MODELS_DIR, "heart_model_lr")
                print(f"Loading LR model from: {lr_model_path}")
                lr_model = LogisticRegressionModel.load(lr_model_path)
                lr_prediction = lr_model.transform(df)
                lr_result = lr_prediction.select("prediction").collect()[0][0]
                lr_prob = lr_prediction.select("probability").collect()[0][0][1]
                
                # Use a lower threshold for high risk
                lr_adjusted_result = 1 if lr_prob >= 0.3 else 0
                
                results["logistic_regression"]["prediction"] = int(lr_adjusted_result)
                results["logistic_regression"]["probability"] = float(lr_prob)
                print("Logistic Regression prediction successful")
            except Exception as e:
                print(f"Error with logistic regression model: {str(e)}")
            
            # Try random forest model
            try:
                rf_model_path = os.path.join(MODELS_DIR, "heart_model_rf")
                print(f"Loading RF model from: {rf_model_path}")
                rf_model = RandomForestClassificationModel.load(rf_model_path)
                rf_prediction = rf_model.transform(df)
                rf_result = rf_prediction.select("prediction").collect()[0][0]
                rf_prob = rf_prediction.select("probability").collect()[0][0][1]
                
                # Use a lower threshold for high risk
                rf_adjusted_result = 1 if rf_prob >= 0.3 else 0
                
                results["random_forest"]["prediction"] = int(rf_adjusted_result)
                results["random_forest"]["probability"] = float(rf_prob)
                print("Random Forest prediction successful")
            except Exception as e:
                print(f"Error with random forest model: {str(e)}")
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
        
        finally:
            # Stop Spark session
            try:
                spark.stop()
                print("Spark session stopped")
            except:
                print("Error stopping Spark session")
            
    except Exception as e:
        print(f"Spark initialization failed: {str(e)}")
    
    # If both model predictions failed, use fallback
    if results["logistic_regression"]["prediction"] is None and results["random_forest"]["prediction"] is None:
        try:
            print("Using fallback prediction method")
            fallback_result, fallback_prob = fallback_heart_prediction(features)
            results["fallback"]["prediction"] = fallback_result
            results["fallback"]["probability"] = fallback_prob
            print(f"Fallback prediction: {fallback_result}, {fallback_prob}")
        except Exception as e:
            print(f"Error in fallback prediction: {str(e)}")
            # Set some default values as absolute last resort
            results["fallback"]["prediction"] = 0
            results["fallback"]["probability"] = 0.5
            print("Using default values due to error in fallback")
    
    return results