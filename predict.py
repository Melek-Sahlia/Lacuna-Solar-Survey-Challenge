import os
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import glob
from tqdm import tqdm

# Configuration
IMAGE_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

def load_image(image_path, img_size=(640, 640)):
    """Load and preprocess an image for prediction"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return img / 255.0

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """Apply non-max suppression to boxes with overlap >= iou_threshold"""
    indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=100, iou_threshold=iou_threshold
    )
    return [boxes[i] for i in indices], [scores[i] for i in indices]

def process_predictions(detection_output, count_output, img_shape, confidence_threshold=0.3):
    """Process raw model predictions into detections and counts"""
    # Process detection predictions
    boxes = []
    scores = []
    classes = []
    
    for i, detection in enumerate(detection_output):
        confidence = detection[4]
        if confidence >= confidence_threshold:
            cx, cy, w, h = detection[:4]
            # Convert to pixel coordinates
            x1 = int((cx - w/2) * img_shape[1])
            y1 = int((cy - h/2) * img_shape[0])
            x2 = int((cx + w/2) * img_shape[1])
            y2 = int((cy + h/2) * img_shape[0])
            
            # Determine class (assuming 0 = boil, 1 = pan)
            class_id = 0 if i % 2 == 0 else 1
            
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
            classes.append(class_id)
    
    # Apply NMS for each class
    boil_indices = [i for i, c in enumerate(classes) if c == 0]
    pan_indices = [i for i, c in enumerate(classes) if c == 1]
    
    boil_boxes = [boxes[i] for i in boil_indices]
    boil_scores = [scores[i] for i in boil_indices]
    
    pan_boxes = [boxes[i] for i in pan_indices]
    pan_scores = [scores[i] for i in pan_indices]
    
    # Apply NMS
    if boil_boxes:
        boil_boxes, boil_scores = non_max_suppression(boil_boxes, boil_scores, iou_threshold=IOU_THRESHOLD)
    
    if pan_boxes:
        pan_boxes, pan_scores = non_max_suppression(pan_boxes, pan_scores, iou_threshold=IOU_THRESHOLD)
    
    # Process count predictions
    boil_count = max(0, int(round(count_output[0])))
    pan_count = max(0, int(round(count_output[1])))
    
    # If detection count doesn't match prediction count, use the larger one
    boil_count = max(boil_count, len(boil_boxes))
    pan_count = max(pan_count, len(pan_boxes))
    
    return {
        'boil_boxes': boil_boxes,
        'boil_scores': boil_scores,
        'pan_boxes': pan_boxes,
        'pan_scores': pan_scores,
        'boil_count': boil_count,
        'pan_count': pan_count
    }

def main():
    # Load model
    model = tf.keras.models.load_model('boil_pan_detector.h5')
    
    # Load test dataset information
    test_df = pd.read_csv('Test.csv')
    
    # Initialize results dataframe
    results = []
    
    # Process each test image
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test images"):
        image_id = row['ID']
        
        # Find the image file
        image_path = glob.glob(f'images/test/{image_id}.jpg')
        if not image_path:
            print(f"Warning: Could not find image for ID {image_id}")
            # Add zero counts
            results.append({'ID': f"{image_id}_boil", 'Target': 0})
            results.append({'ID': f"{image_id}_pan", 'Target': 0})
            continue
        
        # Load and preprocess image
        img = load_image(image_path[0], img_size=IMAGE_SIZE)
        
        # Make prediction
        detection_output, count_output = model.predict(np.expand_dims(img, axis=0))
        
        # Process predictions
        predictions = process_predictions(
            detection_output[0], 
            count_output[0], 
            IMAGE_SIZE, 
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        # Add to results
        results.append({'ID': f"{image_id}_boil", 'Target': predictions['boil_count']})
        results.append({'ID': f"{image_id}_pan", 'Target': predictions['pan_count']})
        
        # Optional: Draw and save detection visualization
        visualization_dir = 'visualizations'
        os.makedirs(visualization_dir, exist_ok=True)
        
        viz_img = cv2.resize(cv2.imread(image_path[0]), IMAGE_SIZE)
        
        # Draw boil boxes
        for box, score in zip(predictions['boil_boxes'], predictions['boil_scores']):
            x1, y1, x2, y2 = box
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(viz_img, f"Boil: {score:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw pan boxes
        for box, score in zip(predictions['pan_boxes'], predictions['pan_scores']):
            x1, y1, x2, y2 = box
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(viz_img, f"Pan: {score:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add count text
        cv2.putText(viz_img, f"Boil count: {predictions['boil_count']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(viz_img, f"Pan count: {predictions['pan_count']}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(visualization_dir, f"{image_id}.jpg"), viz_img)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('submission.csv', index=False)
    
    print(f"Predictions completed. Results saved to submission.csv")

if __name__ == "__main__":
    main() 