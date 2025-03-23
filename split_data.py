import os
import glob
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

def main():
    """
    Split the data into train and validation sets
    """
    print("Splitting data into train and validation sets...")
    
    # Get all images in the train directory
    train_images = glob.glob('images/train/*.jpg')
    
    # Extract image IDs
    image_ids = [os.path.basename(img_path).split('.')[0] for img_path in train_images]
    
    # Split into train and validation sets (80% train, 20% validation)
    train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    
    print(f"Number of training images: {len(train_ids)}")
    print(f"Number of validation images: {len(val_ids)}")
    
    # Create validation directory if it doesn't exist
    os.makedirs('images/val', exist_ok=True)
    os.makedirs('labels/val', exist_ok=True)
    os.makedirs('labels/counts/val', exist_ok=True)
    
    # Move validation images and labels
    for val_id in tqdm(val_ids, desc="Moving validation data"):
        # Move image
        src_img = os.path.join('images/train', f"{val_id}.jpg")
        dst_img = os.path.join('images/val', f"{val_id}.jpg")
        
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.copy(src_img, dst_img)
        
        # Move label
        src_label = os.path.join('labels/train', f"{val_id}.txt")
        dst_label = os.path.join('labels/val', f"{val_id}.txt")
        
        if os.path.exists(src_label) and not os.path.exists(dst_label):
            shutil.copy(src_label, dst_label)
        
        # Move count label
        src_count = os.path.join('labels/counts/train', f"{val_id}.txt")
        dst_count = os.path.join('labels/counts/val', f"{val_id}.txt")
        
        if os.path.exists(src_count) and not os.path.exists(dst_count):
            shutil.copy(src_count, dst_count)
    
    print("Data splitting completed.")

if __name__ == "__main__":
    main() 