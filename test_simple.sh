DATA_PERCENTS=(1 5 10 20 50)

for PERCENT in "${DATA_PERCENTS[@]}"; do
     CUDA_VISIBLE_DEVICES=0 python manydepth/train.py \
        --data_path kitti_data/ \
        --png \
        --g2s \
        --data_percent "$PERCENT" \
        --log_dir "outs/kitti_dp_${PERCENT}" \
        
done

# CUDA_VISIBLE_DEVICES=1 python manydepth/evaluate_depth_mda.py --data_path /mnt/e/cityscapes/ --load_weights_folder outs/kitti/orig_Res/mdp/models/weights_4 --eval_split cityscapes
