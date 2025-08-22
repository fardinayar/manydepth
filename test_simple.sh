DATA_PERCENTS=(0.01 0.05 0.10 0.20 0.50 1.00)

for PERCENT in "${DATA_PERCENTS[@]}"; do
    python manydepth/train.py \
        --data_path kitti_data/ \
        --png \
        --g2s \
        --data_percent "$PERCENT" \
        --log_dir "outs/kitti_dp${PERCENT}"
done

# CUDA_VISIBLE_DEVICES=1 python manydepth/evaluate_depth_mda.py --data_path /mnt/e/cityscapes/ --load_weights_folder outs/kitti/orig_Res/mdp/models/weights_4  --eval_mono --eval_split cityscapes
# python manydepth/scripts/video_demo.py --image_folder our_data/extracted_frames/GX010001_PhoenixPark ---weights_folder outs/gopro/mdp/models/weights_29 --mask_folder our_data/extracted_frames/GX010001_PhoenixPark/