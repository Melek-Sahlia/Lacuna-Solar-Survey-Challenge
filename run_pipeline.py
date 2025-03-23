import os
import sys
import subprocess
import time
import argparse

def run_command(command, description=None):
    """Run a command and print its output"""
    if description:
        print(f"\n{description}")
    
    print(f"Running: {command}")
    start_time = time.time()
    
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    end_time = time.time()
    
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
        return False
    
    print(f"Command completed in {end_time - start_time:.2f} seconds")
    return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Boil and Pan Detection Pipeline')
    parser.add_argument('--prepare-only', action='store_true', help='Only prepare the data')
    parser.add_argument('--train-only', action='store_true', help='Only train the model')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate the model')
    parser.add_argument('--predict-only', action='store_true', help='Only run predictions')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    return parser.parse_args()

def main():
    """Run the complete pipeline for boil and pan detection"""
    args = parse_args()
    
    # Set Python executable (use the current one)
    python_exec = sys.executable
    
    # Check if specific steps are requested
    prepare_only = args.prepare_only
    train_only = args.train_only
    evaluate_only = args.evaluate_only
    predict_only = args.predict_only
    
    # If no specific step is requested, run the full pipeline
    run_all = not (prepare_only or train_only or evaluate_only or predict_only)
    
    print("Starting Boil and Pan Detection Pipeline")
    
    # Step 1: Prepare data
    if run_all or prepare_only:
        success = run_command(f"{python_exec} prepare_data.py", "Step 1: Preparing data")
        if not success:
            print("Data preparation failed. Exiting.")
            return
            
        success = run_command(f"{python_exec} split_data.py", "Step 1b: Splitting data")
        if not success:
            print("Data splitting failed. Exiting.")
            return
    
    # Step 2: Train model
    if run_all or train_only:
        # Update train.py with command line arguments
        cmd = f"{python_exec} train.py"
        if args.epochs != 20:
            cmd += f" --epochs {args.epochs}"
        if args.batch_size != 8:
            cmd += f" --batch-size {args.batch_size}"
            
        success = run_command(cmd, "Step 2: Training model")
        if not success:
            print("Model training failed. Exiting.")
            return
    
    # Step 3: Evaluate model
    if run_all or evaluate_only:
        success = run_command(f"{python_exec} evaluate.py", "Step 3: Evaluating model")
        if not success:
            print("Model evaluation failed. Exiting.")
            return
    
    # Step 4: Generate predictions
    if run_all or predict_only:
        success = run_command(f"{python_exec} predict.py", "Step 4: Generating predictions")
        if not success:
            print("Prediction generation failed. Exiting.")
            return
    
    print("\nPipeline completed successfully!")
    
    # Print submission file information if it exists
    submission_file = 'submission.csv'
    if os.path.exists(submission_file):
        file_size = os.path.getsize(submission_file) / 1024  # KB
        print(f"Submission file '{submission_file}' created ({file_size:.2f} KB)")
        print("You can now submit this file.")

if __name__ == "__main__":
    main() 