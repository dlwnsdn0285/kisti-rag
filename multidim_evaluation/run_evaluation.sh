#!/bin/sh

cd /data/jeonghwan/ragchecker

mkdir -p ./ragchecker_result/temp

INPUT_DIR="/data/jeonghwan/ragchecker/data_formatted"

echo "========================================="
echo "Starting RAG Evaluation Process"
echo "Input Directory: $INPUT_DIR"
echo "========================================="
echo ""

for INPUT_PATH in "$INPUT_DIR"/*_formatted.json; do
    if [ -f "$INPUT_PATH" ]; then
        echo "Processing: $INPUT_PATH"
        python RAGChecker_eval.py "$INPUT_PATH"
        echo "------------------------"
        echo ""
    fi
done

echo "========================================="
echo "All evaluations completed!"
echo "========================================="