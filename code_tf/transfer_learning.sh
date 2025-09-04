#!/bin/bash

# Exit on any error
set -e

# === Configuration Variables ===
MODEL_PATH="../code_tf/aws_ckpt/model.keras"
DATA_PATH="../data/20241118_data_splited.mat"
MODEL_DIR="../code_tf/aws_ckpt"
NAMES_TO_TRAIN=["conv2d_25","conv2d_26","conv2d_27","outQF_logits","conv2d_28","conv2d_29","outDF_logits"]
TRAIN_SUBSET=8000
BATCH=32
EPOCHS=50
LEARNING_RATE=1e-5
DECAY_RATE=0.4
PATIENCE=10
IS_AWS=True

SCALE_FL=1e5
SCALE_OP0=10
SCALE_OP1=1
SCALE_DF=1
SCALE_QF=1
SCALE_RE=1

# === Run transfer learning ===
echo "Launching transfer learning with the following parameters:"

python transfer_learning.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_dir $MODEL_DIR \
    --is_aws $IS_AWS \
    --names_to_train $NAMES_TO_TRAIN \
    --batch $BATCH \
    --epochs $EPOCHS \
    --learningRate $LEARNING_RATE \
    --decayRate $DECAY_RATE \
    --patience $PATIENCE \
    --scaleFL $SCALE_FL \
    --scaleOP0 $SCALE_OP0 \
    --scaleOP1 $SCALE_OP1 \
    --scaleDF $SCALE_DF \
    --scaleQF $SCALE_QF \
    --scaleRE $SCALE_RE \
    