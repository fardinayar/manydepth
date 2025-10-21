#!/bin/bash

# Script to run depth inference with both teacher and student models
# on test data and save numpy predictions with proper naming

# Note: Removed set -e to allow script to continue processing all samples even if one fails

# Configuration
TEST_DATA_DIR="test_data"
OUTPUT_DIR="depth_predictions"
WEIGHTS_FOLDER="outs/kitti/vits_518/mdp/models/weights_4"
DEPTH_ANYTHING_ENCODER="vits"
MIN_DEPTH=0.1
MAX_DEPTH=400.0

# Create output directories
mkdir -p "${OUTPUT_DIR}/teacher"
mkdir -p "${OUTPUT_DIR}/student"

echo "Starting depth inference on test data..."
echo "Test data directory: ${TEST_DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Weights folder: ${WEIGHTS_FOLDER}"
echo ""

# Function to run inference for a single sample
run_inference() {
    local sample_dir="$1"
    local sample_name=$(basename "$sample_dir")
    
    echo "Processing sample: ${sample_name}"
    
    # Find target and reference images
    local target_img=$(find "$sample_dir" -name "target_*.png" | head -1)
    local reference_img=$(find "$sample_dir" -name "reference_*.png" | head -1)
    
    if [[ -z "$target_img" || -z "$reference_img" ]]; then
        echo "Warning: Could not find target or reference image in ${sample_dir}"
        return 1
    fi
    
    echo "  Target image: $(basename "$target_img")"
    echo "  Reference image: $(basename "$reference_img")"
    
    # Run student model inference (multi-frame, no poses)
    echo "  Running student model inference..."
    python manydepth/scripts/save_pointcloud_dual.py \
        --target_image "$target_img" \
        --lookup_frame "$reference_img" \
        --weights_folder "$WEIGHTS_FOLDER" \
        --depth_anything_encoder "$DEPTH_ANYTHING_ENCODER" \
        --min_depth "$MIN_DEPTH" \
        --max_depth "$MAX_DEPTH" \
        --output_dir "${OUTPUT_DIR}/student" \
        --coordinate_system camera
    
    # Rename the output files to include sample name
    local target_basename=$(basename "$target_img" .png)
    if [[ -f "${OUTPUT_DIR}/student/${target_basename}_depth.npy" ]]; then
        mv "${OUTPUT_DIR}/student/${target_basename}_depth.npy" "${OUTPUT_DIR}/student/${sample_name}_student_depth.npy"
    fi
    if [[ -f "${OUTPUT_DIR}/student/${target_basename}_disp.npy" ]]; then
        mv "${OUTPUT_DIR}/student/${target_basename}_disp.npy" "${OUTPUT_DIR}/student/${sample_name}_student_disp.npy"
    fi
    
    # Run teacher model inference (monocular, single frame)
    echo "  Running teacher model inference..."
    python manydepth/scripts/save_pointcloud_dual.py \
        --target_image "$target_img" \
        --lookup_frame "$target_img" \
        --weights_folder "$WEIGHTS_FOLDER" \
        --depth_anything_encoder "$DEPTH_ANYTHING_ENCODER" \
        --min_depth "$MIN_DEPTH" \
        --max_depth "$MAX_DEPTH" \
        --output_dir "${OUTPUT_DIR}/teacher" \
        --coordinate_system camera \
        --teacher_mode
    
    # Rename the output files to include sample name
    if [[ -f "${OUTPUT_DIR}/teacher/${target_basename}_depth.npy" ]]; then
        mv "${OUTPUT_DIR}/teacher/${target_basename}_depth.npy" "${OUTPUT_DIR}/teacher/${sample_name}_teacher_depth.npy"
    fi
    if [[ -f "${OUTPUT_DIR}/teacher/${target_basename}_disp.npy" ]]; then
        mv "${OUTPUT_DIR}/teacher/${target_basename}_disp.npy" "${OUTPUT_DIR}/teacher/${sample_name}_teacher_disp.npy"
    fi
    
    echo "  Completed ${sample_name}"
    echo ""
}

# Check if test data directory exists
if [[ ! -d "$TEST_DATA_DIR" ]]; then
    echo "Error: Test data directory '$TEST_DATA_DIR' not found!"
    exit 1
fi

# Check if weights folder exists
if [[ ! -d "$WEIGHTS_FOLDER" ]]; then
    echo "Error: Weights folder '$WEIGHTS_FOLDER' not found!"
    exit 1
fi

# Process all sample directories
sample_count=0
for sample_dir in "${TEST_DATA_DIR}"/sample_*; do
    if [[ -d "$sample_dir" ]]; then
        run_inference "$sample_dir"
        ((sample_count++))
    fi
done

echo "Completed processing ${sample_count} samples"
echo ""
echo "Output files saved to:"
echo "  Student model: ${OUTPUT_DIR}/student/"
echo "  Teacher model: ${OUTPUT_DIR}/teacher/"
echo ""
echo "File naming convention:"
echo "  Student: {sample_name}_student_depth.npy, {sample_name}_student_disp.npy"
echo "  Teacher: {sample_name}_teacher_depth.npy, {sample_name}_teacher_disp.npy"
