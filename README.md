# Boil and Pan Detection Pipeline

This project provides a complete pipeline for detecting and counting boils and pans in aerial images. The pipeline includes data preparation, model training, evaluation, and prediction.

## Directory Structure

```
.
├── images/
│   ├── train/     # Training images
│   ├── val/       # Validation images
│   └── test/      # Test images
├── labels/
│   ├── train/     # Training labels for bounding boxes
│   ├── val/       # Validation labels for bounding boxes
│   └── counts/    # Count labels for boils and pans
├── prepare_data.py     # Script to prepare the data
├── split_data.py       # Script to split data into train and validation sets
├── train.py            # Script to train the model
├── evaluate.py         # Script to evaluate the model
├── predict.py          # Script to generate predictions on test data
└── run_pipeline.py     # Main script to run the entire pipeline
```

## Requirements

The required Python packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Quick Start

To run the entire pipeline in one go:

```bash
python run_pipeline.py
```

This will:
1. Prepare the data by converting polygons to bounding boxes
2. Split the data into training and validation sets
3. Train a model
4. Evaluate the model on the validation set
5. Generate predictions on the test set
6. Create a submission file in the required format

## Step-by-Step Execution

If you prefer to run the pipeline step by step:

### 1. Prepare the data

```bash
python prepare_data.py
python split_data.py
```

This will convert the CSV data to the YOLO format and split it into training and validation sets.

### 2. Train the model

```bash
python train.py --epochs 20 --batch-size 8
```

Optional arguments:
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 8)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--img-size`: Image size (default: 640)

### 3. Evaluate the model

```bash
python evaluate.py
```

This will evaluate the model on the validation set and produce evaluation metrics.

### 4. Generate predictions

```bash
python predict.py
```

This will generate predictions on the test set and create a submission file.

## Model Architecture

The model uses a ResNet50 backbone pre-trained on ImageNet with two heads:
1. Detection head: Predicts bounding boxes for boils and pans
2. Counting head: Predicts the count of boils and pans in each image

## Customization

You can customize various aspects of the pipeline:

### Training Parameters

Modify the training parameters in `train.py` or use command line arguments:

```bash
python train.py --epochs 30 --batch-size 16 --learning-rate 0.0005
```

### Model Architecture

To modify the model architecture, edit the `build_model` function in `train.py`.

### Running Specific Steps

Use the following arguments with `run_pipeline.py`:

```bash
# Only prepare the data
python run_pipeline.py --prepare-only

# Only train the model
python run_pipeline.py --train-only

# Only evaluate the model
python run_pipeline.py --evaluate-only

# Only generate predictions
python run_pipeline.py --predict-only
```

## Output

After running the pipeline, you'll get the following outputs:
- Trained model: `boil_pan_detector.h5`
- Training history plot: `training_history.png`
- Evaluation metrics: `evaluation_metrics.txt`
- Evaluation plots: `evaluation_plots.png`
- Submission file: `submission.csv`
- Visualizations: in the `visualizations/` and `val_visualizations/` directories 