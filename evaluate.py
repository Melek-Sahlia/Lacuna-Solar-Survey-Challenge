import os
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

def load_ground_truth(image_id):
    """Load ground truth counts for an image ID"""
    # Try to find the count file
    count_path = os.path.join('labels', 'counts', 'val', f"{image_id}.txt")
    
    boil_count = 0
    pan_count = 0
    
    if os.path.exists(count_path):
        with open(count_path, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        class_id, count = int(parts[0]), int(parts[1])
                        if class_id == 0:
                            boil_count = count
                        elif class_id == 1:
                            pan_count = count
    
    return {
        'boil_count': boil_count,
        'pan_count': pan_count
    }

def main():
    # Load the model
    model = tf.keras.models.load_model('boil_pan_detector.h5')
    
    # Find validation images
    val_images = glob.glob('images/val/*.jpg')
    if not val_images:
        print("No validation images found. Please make sure validation data is prepared.")
        return
    
    # Initialize metrics
    gt_boil_counts = []
    pred_boil_counts = []
    gt_pan_counts = []
    pred_pan_counts = []
    
    # Process each validation image
    for image_path in tqdm(val_images, desc="Evaluating on validation set"):
        image_id = os.path.basename(image_path).split('.')[0]
        
        # Load ground truth
        gt = load_ground_truth(image_id)
        
        # Load and preprocess image
        img = load_image(image_path, img_size=IMAGE_SIZE)
        
        # Make prediction
        detection_output, count_output = model.predict(np.expand_dims(img, axis=0))
        
        # Process predictions
        predictions = process_predictions(
            detection_output[0], 
            count_output[0], 
            IMAGE_SIZE, 
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        # Store results for evaluation
        gt_boil_counts.append(gt['boil_count'])
        pred_boil_counts.append(predictions['boil_count'])
        gt_pan_counts.append(gt['pan_count'])
        pred_pan_counts.append(predictions['pan_count'])
        
        # Optional: Save visualization
        visualization_dir = 'val_visualizations'
        os.makedirs(visualization_dir, exist_ok=True)
        
        viz_img = cv2.resize(cv2.imread(image_path), IMAGE_SIZE)
        
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
        cv2.putText(viz_img, f"Boil count: {predictions['boil_count']} (GT: {gt['boil_count']})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(viz_img, f"Pan count: {predictions['pan_count']} (GT: {gt['pan_count']})", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(visualization_dir, f"{image_id}.jpg"), viz_img)
    
    # Calculate metrics
    boil_mae = mean_absolute_error(gt_boil_counts, pred_boil_counts)
    boil_rmse = np.sqrt(mean_squared_error(gt_boil_counts, pred_boil_counts))
    
    pan_mae = mean_absolute_error(gt_pan_counts, pred_pan_counts)
    pan_rmse = np.sqrt(mean_squared_error(gt_pan_counts, pred_pan_counts))
    
    # Overall metrics
    all_gt = gt_boil_counts + gt_pan_counts
    all_pred = pred_boil_counts + pred_pan_counts
    overall_mae = mean_absolute_error(all_gt, all_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_gt, all_pred))
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Boil Count MAE: {boil_mae:.4f}")
    print(f"Boil Count RMSE: {boil_rmse:.4f}")
    print(f"Pan Count MAE: {pan_mae:.4f}")
    print(f"Pan Count RMSE: {pan_rmse:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    
    # Save metrics to a file
    metrics = {
        'boil_mae': boil_mae,
        'boil_rmse': boil_rmse,
        'pan_mae': pan_mae,
        'pan_rmse': pan_rmse,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse
    }
    
    with open('evaluation_metrics.txt', 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    # Plot prediction vs. ground truth
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(gt_boil_counts, pred_boil_counts, alpha=0.5)
    plt.plot([0, max(gt_boil_counts)], [0, max(gt_boil_counts)], 'r--')
    plt.xlabel('Ground Truth Boil Count')
    plt.ylabel('Predicted Boil Count')
    plt.title('Boil Count: Prediction vs. Ground Truth')
    
    plt.subplot(1, 2, 2)
    plt.scatter(gt_pan_counts, pred_pan_counts, alpha=0.5)
    plt.plot([0, max(gt_pan_counts)], [0, max(gt_pan_counts)], 'r--')
    plt.xlabel('Ground Truth Pan Count')
    plt.ylabel('Predicted Pan Count')
    plt.title('Pan Count: Prediction vs. Ground Truth')
    
    plt.tight_layout()
    plt.savefig('evaluation_plots.png')
    print("Evaluation completed. Results saved to evaluation_metrics.txt and evaluation_plots.png")

if __name__ == "__main__":
    main() 