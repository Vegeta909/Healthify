from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler, StandardScaler
import os
import sys

import os

# Set HADOOP_HOME environment variable - use environment variables instead of hardcoded paths
hadoop_home = os.environ.get('HADOOP_HOME', '')
java_home = os.environ.get('JAVA_HOME', '')

# Add Hadoop bin to PATH only if environment variable exists
if hadoop_home:
    hadoop_bin = os.path.join(hadoop_home, 'bin')
    os.environ['PATH'] = hadoop_bin + os.pathsep + os.environ['PATH']
    # Disable Hadoop native libraries warning
    os.environ['HADOOP_OPTS'] = "-Djava.library.path=" + os.path.join(hadoop_home, 'bin')


# Verify Java is accessible
import subprocess
try:
    result = subprocess.run(['java', '-version'], capture_output=True, text=True)
    java_output = result.stderr if result.stderr else result.stdout
    print("Java detected:", java_output.split('\n')[0])
except Exception as e:
    print(f"Error checking Java: {e}")
    sys.exit(1)



def initialize_spark():
    """Initialize Spark session"""
    return SparkSession.builder \
        .appName("HEALTHIFY - Data Processing") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def load_diabetes_data(spark):
    """Load diabetes dataset from CSV"""
    print("Loading diabetes dataset...")
    return spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .csv("data/diabetes.csv")

def load_heart_data(spark):
    """Load heart attack dataset from CSV"""
    print("Loading heart attack dataset...")
    return spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .csv("data/heart_attack_prediction_india.csv")

def process_diabetes_data(df):
    """Process diabetes dataset"""
    print("Processing diabetes dataset...")
    
    # Display basic statistics
    print(f"Total records: {df.count()}")
    print("Schema:")
    df.printSchema()
    
    # Replace zeros with column means for specific columns
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for column in zero_cols:
        # Calculate mean excluding zeros
        mean_val = df.filter(col(column) != 0).agg(mean(column)).collect()[0][0]
        print(f"Mean {column}: {mean_val}")
        
        # Replace zeros with mean
        df = df.withColumn(column, when(col(column) == 0, mean_val).otherwise(col(column)))
    
    # Feature engineering
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Create feature vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled")
    df = assembler.transform(df)
    
    # Standardize features
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", 
                           withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # Select relevant columns
    df = df.select("features", "Outcome")
    
    return df, scaler_model

def process_heart_data(df):
    """Process heart attack dataset"""
    print("Processing heart attack dataset...")
    
    # Display basic statistics
    print(f"Total records: {df.count()}")
    print("Schema:")
    df.printSchema()
    
    # Drop unnecessary columns
    drop_cols = ['Patient_ID', 'State_Name']
    df = df.drop(*drop_cols)
    
    # Handle missing values
    df = df.na.drop()
    
    # Convert Gender to numeric (1 for Male, 0 for Female)
    df = df.withColumn("Gender", when(col("Gender") == "Male", 1).otherwise(0))
    
    # Feature engineering
    feature_cols = [c for c in df.columns if c != 'Heart_Attack_Risk']
    
    # Create feature vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled")
    df = assembler.transform(df)
    
    # Standardize features
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", 
                           withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # Select relevant columns
    df = df.select("features", "Heart_Attack_Risk")
    
    return df, scaler_model

def save_processed_data(df, name):
    """Save processed data as parquet files to preserve Vector types"""
    output_path = f"data/processed/{name}"
    print(f"Saving processed data to {output_path}")
    
    # Save as parquet to preserve the Vector type
    df.write.mode("overwrite").parquet(output_path)



def main():
    """Main function to process both datasets"""
    # Initialize Spark
    spark = initialize_spark()
    
    # Create directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Process diabetes data
    diabetes_df = load_diabetes_data(spark)
    processed_diabetes, diabetes_scaler = process_diabetes_data(diabetes_df)
    save_processed_data(processed_diabetes, "diabetes")
    
    # Process heart attack data
    heart_df = load_heart_data(spark)
    processed_heart, heart_scaler = process_heart_data(heart_df)
    save_processed_data(processed_heart, "heart")
    
    # Show sample of processed data
    print("\nSample of processed diabetes data:")
    processed_diabetes.show(5)
    
    print("\nSample of processed heart attack data:")
    processed_heart.show(5)
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
