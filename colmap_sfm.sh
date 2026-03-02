#!/usr/bin/env bash
set -e

DATASET_DIR=dataset_example/image_data/mpi/S1/scene_1/raw_images      # images in one folder
WORK_DIR=./sfm_work
DB_PATH=$WORK_DIR/database.db
SPARSE_DIR=$WORK_DIR/sparse
LOG_DIR=$WORK_DIR/logs
mkdir -p "$WORK_DIR" "$SPARSE_DIR" "$LOG_DIR"

# 1) Feature extraction
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$DATASET_DIR" \
  --ImageReader.single_camera 0 \
  --SiftExtraction.use_gpu 1

# 2) Matching
colmap exhaustive_matcher \
  --database_path "$DB_PATH" \
  --SiftMatching.use_gpu 1

# 3) Sparse reconstruction
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$DATASET_DIR" \
  --output_path "$SPARSE_DIR" \
  2>&1 | tee "$LOG_DIR/mapper.log"

# # 4) Convert model to TXT for parsing
# # usually model is in sparse/0
# colmap model_converter \
#   --input_path "$SPARSE_DIR/0" \
#   --output_path "$SPARSE_DIR/0_txt" \
#   --output_type TXT

echo "Done. Read cameras.txt and images.txt in $SPARSE_DIR/0_txt"