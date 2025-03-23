import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import argparse

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Default configuration
DEFAULT_IMAGE_SIZE = (640, 640)
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 20
DEFAULT_NUM_CLASSES = 2  # boil and pan
DEFAULT_LEARNING_RATE = 1e-4
MAX_BOXES = 100  # Maximum number of boxes to detect

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, label_paths, batch_size=8, img_size=(640, 640), augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.n = len(self.image_paths)
        
    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.label_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X = np.zeros((len(batch_x), self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        y_boxes = np.zeros((len(batch_x), MAX_BOXES, 5), dtype=np.float32)  # Fixed size for boxes
        y_counts = np.zeros((len(batch_x), 2), dtype=np.float32)  # Fixed size for counts
        
        for i, (img_path, label_path) in enumerate(zip(batch_x, batch_y)):
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            X[i] = img / 255.0
            
            # Load labels
            
            # Get the base filename without extension
            base_filename = os.path.basename(label_path)
            id_part = os.path.splitext(base_filename)[0]
            
            # Load bounding box annotations
            box_count = 0
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.read().strip().split('\n')
                    for line in lines:
                        if line and box_count < MAX_BOXES:  # Limit to MAX_BOXES
                            class_id, x_center, y_center, width, height = map(float, line.split())
                            y_boxes[i, box_count] = [x_center, y_center, width, height, class_id]
                            box_count += 1
            
            # Load count annotations (from counts directory)
            count_path = os.path.join('labels', 'counts', 'train', f"{id_part}.txt")
            if os.path.exists(count_path):
                with open(count_path, 'r') as f:
                    lines = f.read().strip().split('\n')
                    for line in lines:
                        if line:
                            parts = line.split()
                            if len(parts) >= 2:
                                class_id, count = int(parts[0]), int(parts[1])
                                if class_id < 2:  # Only boil and pan
                                    y_counts[i, class_id] = count
        
        # Return tensors directly, not in a list
        return X, (y_boxes, y_counts)

def create_tf_dataset(image_paths, label_paths, batch_size=8, img_size=(640, 640), augment=False):
    """Create a TensorFlow dataset from paths"""
    # Create the generator
    generator = DataGenerator(image_paths, label_paths, batch_size, img_size, augment)
    
    # Create a tf.data.Dataset from the generator
    def data_gen():
        for i in range(len(generator)):
            yield generator[i]
    
    # Define output signature for the dataset
    output_signature = (
        tf.TensorSpec(shape=(None, img_size[0], img_size[1], 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None, MAX_BOXES, 5), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
        )
    )
    
    dataset = tf.data.Dataset.from_generator(
        data_gen,
        output_signature=output_signature
    )
    
    return dataset

def build_model(input_shape=(640, 640, 3), num_classes=2, learning_rate=1e-4):
    """
    Build a multi-task model for object detection and counting
    """
    # Base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Detection head - completely different approach
    x = base_model.output
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='detection_conv1')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='detection_conv2')(x)
    x = Flatten(name='detection_flatten')(x)
    detection_output = Dense(MAX_BOXES * 5, activation='linear', name='detection_dense')(x)
    detection_output = Reshape((MAX_BOXES, 5), name='bbox_output')(detection_output)
    
    # Counting head
    y = GlobalAveragePooling2D(name='counting_gap')(base_model.output)
    y = Dense(256, activation='relu', name='counting_dense1')(y)
    y = Dense(128, activation='relu', name='counting_dense2')(y)
    count_output = Dense(num_classes, activation='linear', name='count_output')(y)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=[detection_output, count_output])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={
            'bbox_output': 'mse',  # For detection
            'count_output': 'mse'   # For counting
        },
        loss_weights={
            'bbox_output': 1.0,
            'count_output': 1.0
        }
    )
    
    return model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Boil and Pan Detection Model')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, 
                       help=f'Batch size for training (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, 
                       help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE, 
                       help=f'Learning rate (default: {DEFAULT_LEARNING_RATE})')
    parser.add_argument('--img-size', type=int, default=DEFAULT_IMAGE_SIZE[0], 
                       help=f'Image size (default: {DEFAULT_IMAGE_SIZE[0]})')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Apply arguments
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    IMAGE_SIZE = (args.img_size, args.img_size)
    NUM_CLASSES = DEFAULT_NUM_CLASSES
    
    print(f"Training with:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Image size: {IMAGE_SIZE}")
    print(f"  - Max boxes per image: {MAX_BOXES}")
    
    # Get all training images and labels
    train_images = glob.glob('images/train/*.jpg')
    train_labels = [f"labels/train/{os.path.basename(img_path).replace('.jpg', '.txt')}" 
                    for img_path in train_images]
    
    # Get validation images and labels
    val_images = glob.glob('images/val/*.jpg')
    val_labels = [f"labels/val/{os.path.basename(img_path).replace('.jpg', '.txt')}" 
                  for img_path in val_images]
    
    # If validation set is empty, create it from train set
    if not val_images:
        print("No validation images found. Creating validation set from training data...")
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42
        )
    
    print(f"Training on {len(train_images)} images, validating on {len(val_images)} images")
    
    # Create TF datasets instead of generators
    train_dataset = create_tf_dataset(train_images, train_labels, batch_size=BATCH_SIZE, img_size=IMAGE_SIZE, augment=True)
    val_dataset = create_tf_dataset(val_images, val_labels, batch_size=BATCH_SIZE, img_size=IMAGE_SIZE, augment=False)
    
    # Build model
    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'model_checkpoint.h5', 
            save_best_only=True, 
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save model
    model.save('boil_pan_detector.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['count_output_loss'], label='Count Train Loss')
    plt.plot(history.history['val_count_output_loss'], label='Count Val Loss')
    plt.title('Count Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    print("Training completed!")

if __name__ == "__main__":
    main() 