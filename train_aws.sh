#!/bin/bash

# Exit on error
set -e

# Default values
BATCH_SIZE=32
INSTANCE_NAME="FLAN-MoE-Training"
DATA_PATH="data/"
OUTPUT_PATH="models/"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Setup environment
echo "Setting up training environment..."

# Install required packages
pip install --no-cache-dir \
    torch \
    transformers \
    wandb \
    tqdm \
    numpy \
    boto3

# Create necessary directories
mkdir -p $DATA_PATH
mkdir -p $OUTPUT_PATH

# Download data if using S3
if [[ $DATA_PATH == s3://* ]]; then
    echo "Downloading data from S3..."
    aws s3 sync $DATA_PATH ./data/
    DATA_PATH="./data/"
fi

# Set environment variables
export WANDB_API_KEY="your-wandb-key-here"  # Replace with your key
export DATA_PATH=$DATA_PATH
export AWS_INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)

# Start training
echo "Starting training with batch size $BATCH_SIZE..."
python aws_train.py --batch_size $BATCH_SIZE

# Upload results to S3 if output path is S3
if [[ $OUTPUT_PATH == s3://* ]]; then
    echo "Uploading results to S3..."
    aws s3 sync final_model $OUTPUT_PATH
fi

echo "Training complete!" 