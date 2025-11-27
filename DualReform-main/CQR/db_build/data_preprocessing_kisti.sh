#!/bin/bash

TASK=kisti
# TASK=topiocqa
DATA_PATH=/data2/../nlp_data # /home/user/InfoCQR/datasets
OUTPUT_PATH=/data2/../nlp_data/kisti/preprocessed

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# /data2/../nlp_data/topiocqa/collection-paragraph-topiocqa
python data_preprocessing_kisti.py \
  --task ${TASK} \
  --data_path ${DATA_PATH} \
  --output_path ${OUTPUT_PATH} \
  --max_passage_length 384 \
  --test_collection_path ${DATA_PATH}/${TASK}
  
