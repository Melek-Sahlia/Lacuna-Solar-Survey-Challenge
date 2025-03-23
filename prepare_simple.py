import os
import pandas as pd
import numpy as np
import cv2
import ast
from tqdm import tqdm
import glob
import shutil

def create_directory_structure():
    """Create the required directory structure if it doesn't exist"""
    os.makedirs('images/train', exist_ok=True)
    os.makedirs('images/val', exist_ok=True)
    os.makedirs('images/test', exist_ok=True)
    os.makedirs('labels/train', exist_ok=True)
    os.makedirs('labels/val', exist_ok=True)
    os.makedirs('labels/counts/train', exist_ok=True)
    os.makedirs('labels/counts/val', exist_ok=True)
    print("Directory structure created.")

def main():
    # Create directory structure
    create_directory_structure()
    
    # Load CSV data
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('Test.csv')
    
    # Process training data
    image_ids = train_df['ID'].unique()
    
    for image_id in tqdm(image_ids, desc="Processing training data"):
        # Get all entries for this image
        img_data = train_df[train_df['ID'] == image_id]
        
        # Create count files
        count_file_path = os.path.join('labels', 'counts', 'train', f"{image_id}.txt")
        with open(count_file_path, 'w') as f:
            # For boil - find the largest boil count value
            boil_count = img_data['boil_nbr'].max() if len(img_data) > 0 else 0
            f.write(f"0 {int(boil_count)}\n")
            
            # For pan - find the largest pan count value
            pan_count = img_data['pan_nbr'].max() if len(img_data) > 0 else 0
            f.write(f"1 {int(pan_count)}\n")
    
    # Process test data
    # Make sure test images are copied to the right location
    test_images = glob.glob('images/*.jpg')
    for img_path in tqdm(test_images, desc="Copying test images"):
        img_id = os.path.basename(img_path).split('.')[0]
        if img_id in test_df['ID'].values:
            dest_path = os.path.join('images', 'test', os.path.basename(img_path))
            shutil.copy(img_path, dest_path)
    
    print("Basic data preparation completed.")
    print("Note: This simplified script only creates count labels. For full functionality, run prepare_data.py.")

if __name__ == "__main__":
    main() 