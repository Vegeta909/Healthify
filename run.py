import os
import subprocess
import time

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {text} ".center(80, "="))
    print("="*80 + "\n")

def run_command(command, description):
    """Run a shell command and print its output"""
    print_header(description)
    print(f"Running: {command}\n")
    start_time = time.time()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    end_time = time.time()
    
    if process.returncode == 0:
        print(f"\n✅ {description} completed successfully in {end_time - start_time:.2f} seconds")
    else:
        print(f"\n❌ {description} failed with return code {process.returncode}")
        exit(1)

def main():
    """Run the complete HEALTHIFY pipeline"""
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Process data
    run_command("python src/data_processing.py", "Data Processing")
    
    # Train models
    run_command("python src/model_training.py", "Model Training")
    
    # Start Streamlit app
    run_command("streamlit run src/app.py", "Starting Streamlit App")

if __name__ == "__main__":
    main()
