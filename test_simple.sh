DATA_PERCENTS=(0.01 0.05 0.10 0.20 0.50 1.00)

for PERCENT in "${DATA_PERCENTS[@]}"; do
    python manydepth/train.py \
        --data_path kitti_data/ \
        --png \
        --g2s \
        --data_percent "$PERCENT" \
        --log_dir "outs/kitti_dp${PERCENT}"
done