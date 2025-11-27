#!/bin/bash

TASK=qrecc
DATA_PATH=/data/../nlp_data # /home/user/InfoCQR/datasets
OUTPUT_PATH=/data/../nlp_data/preprocessed

python data_preprocessing.py \
  --task ${TASK} \
  --data_path ${DATA_PATH} \
  --output_path ${OUTPUT_PATH} \
  --max_passage_length 384 \
  --test_collection_path ${DATA_PATH}/${TASK}
  
