import os
import pandas as pd
import numpy as np
import cv2
import ast
from tqdm import tqdm
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

def convert_polygon_to_bbox(polygon, img_width, img_height):
    """Convert a polygon to a bounding box format [x_center, y_center, width, height]"""
    try:
        # Parse polygon string if needed
        if isinstance(polygon, str):
            polygon = ast.literal_eval(polygon)
        
        # Extract x, y coordinates
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        
        # Compute bounding box
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Normalize to image dimensions
        x_center = (x_min + x_max) / (2 * img_width)
        y_center = (y_min + y_max) / (2 * img_height)
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return [x_center, y_center, width, height]
    except Exception as e:
        print(f"Error converting polygon: {e}")
        return None

def process_train_data(train_df, img_dir='images/train'):
    """Process training data to create YOLO format labels"""
    unique_images = train_df['ID'].unique()
    
    for image_id in tqdm(unique_images, desc="Processing training data"):
        # Get all annotations for this image
        img_annotations = train_df[train_df['ID'] == image_id]
        
        # Save label information
        label_file_path = os.path.join('labels/train', f"{image_id}.txt")
        count_file_path = os.path.join('labels/counts/train', f"{image_id}.txt")
        
        # Get image dimensions (assuming they're in the same directory)
        img_path = os.path.join(img_dir, image_id + '.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_height, img_width = img.shape[:2]
            else:
                print(f"Warning: Could not read image {img_path}")
                continue
        else:
            print(f"Warning: Image file not found {img_path}")
            continue
        
        # Process annotations
        with open(label_file_path, 'w') as label_file, open(count_file_path, 'w') as count_file:
            # Process boil annotations
            boil_annotations = img_annotations[img_annotations['boil_nbr'] > 0]
            boil_count = sum(boil_annotations['boil_nbr'])
            
            # Write boil count to count file
            count_file.write(f"0 {boil_count}\n")
            
            # Process pan annotations
            pan_annotations = img_annotations[img_annotations['pan_nbr'] > 0]
            pan_count = sum(pan_annotations['pan_nbr'])
            
            # Write pan count to count file
            count_file.write(f"1 {pan_count}\n")
            
            # Write bounding box information
            for _, row in img_annotations.iterrows():
                polygon = row['polygon']
                if pd.isna(polygon) or polygon is None or polygon == '':
                    continue
                
                bbox = convert_polygon_to_bbox(polygon, img_width, img_height)
                if bbox is None:
                    continue
                
                # Write to label file (class_id x_center y_center width height)
                if row['boil_nbr'] > 0:
                    # Class 0 for boil
                    label_file.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                
                if row['pan_nbr'] > 0:
                    # Class 1 for pan
                    label_file.write(f"1 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

def prepare_test_data(test_df):
    """Ensure test images are in the correct location"""
    # Verify test directory exists
    os.makedirs('images/test', exist_ok=True)
    
    for _, row in tqdm(test_df.iterrows(), desc="Processing test data"):
        image_id = row['ID']
        
        # Try to find the image in the source directory
        source_paths = [
            os.path.join('images', f"{image_id}.jpg"),
            os.path.join('images/test', f"{image_id}.jpg")
        ]
        
        dest_path = os.path.join('images/test', f"{image_id}.jpg")
        
        if os.path.exists(dest_path):
            continue  # Already in the right place
        
        for source_path in source_paths:
            if os.path.exists(source_path):
                shutil.copy(source_path, dest_path)
                break

def main():
    # Create directory structure
    create_directory_structure()
    
    # Load CSV data
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('Test.csv')
    
    # Process train data
    process_train_data(train_df)
    
    # Process test data
    prepare_test_data(test_df)
    
    print("Data preparation completed.")

if __name__ == "__main__":
    main() 