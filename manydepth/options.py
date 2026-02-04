# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="ManyDepth options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark",
                                          "cityscapes_preprocessed"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test",
                                          "cityscapes_preprocessed", "gopro"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=182)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=630)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.0)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0])
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=300.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=5)
        self.parser.add_argument("--pytorch_random_seed",
                                 default=None,
                                 type=int)
        self.parser.add_argument("--max_grad_norm",
                                 type=float,
                                 help="maximum gradient norm for clipping (0 to disable)",
                                 default=1)
        self.parser.add_argument("--warmup_steps",
                                 type=int,
                                 help="number of warmup steps for learning rate (0 to disable)",
                                 default=1000)

        # ABLATION options
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument('--num_matching_frames',
                                 help='Sets how many previous frames to load to build the cost'
                                      'volume',
                                 type=int,
                                 default=1)
        self.parser.add_argument("--disable_motion_masking",
                                 help="If set, will not apply consistency loss in regions where"
                                      "the cost volume is deemed untrustworthy",
                                 action="store_true")
        self.parser.add_argument("--no_matching_augmentation",
                                 action='store_true',
                                 help="If set, will not apply static camera augmentation or "
                                      "zero cost volume augmentation during training")
        self.parser.add_argument("--no_temporal_fusion",
                                 action='store_true',
                                 help="If set, will not use temporal fusion in the depth decoder")
        self.parser.add_argument("--no_lora",
                                 action='store_true',
                                 help="If set, will not use LoRA in the depth decoder and encoder")
        self.parser.add_argument("--no_consistency_loss",
                                 action='store_true',
                                 help="If set, will not use consistency loss in the depth decoder")
        self.parser.add_argument("--no_loss_dynamic_weight",
                                 action='store_true',
                                 help="If set, will not use dynamic weight for the loss")
        

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--mono_weights_folder",
                                 type=str)
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=50)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        self.parser.add_argument("--save_intermediate_models",
                                 help="if set, save the model each time we log to tensorboard",
                                 action='store_true')

        # EVALUATION options
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=["eigen", "eigen_benchmark", "benchmark", "odom_9",
                                          "odom_10", "cityscapes"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)

        self.parser.add_argument('--static_camera',
                                 action='store_true',
                                 help='If set, during evaluation the current frame will also be'
                                      'used as the lookup frame, to simulate a static camera')
        self.parser.add_argument('--eval_teacher',
                                 action='store_true',
                                 help='If set, the teacher network will be evaluated')
        
        # DEPTH_ANYTHING options
        self.parser.add_argument('--depth_anything_encoder',
                                 type=str,
                                 choices=["vits", "vitb", "vitl", "vitg"],
                                 default="vits")
        
        self.parser.add_argument('--depth_anything_checkpoint',
                                 type=str,
                                 default='checkpoints')
        
        self.parser.add_argument('--encoder_lr_coef',
                                 type=float,
                                 default=1.0)
        
        self.parser.add_argument('--fusion_lr_coef',
                                 type=float,
                                 default=4.0,
                                 help='Learning rate multiplier for feature fusion parameters')
        
        self.parser.add_argument("--g2s",
                         help="use g2s loss",
                         action="store_true")
        
        self.parser.add_argument('--data_percent',
                                 type=float,
                                 default=100.0)
        
        self.parser.add_argument('--pose_from_scratch',
                                 action='store_true',
                                 help='If set, the pose encoder and decoder will be initialized randomly')
        
        self.parser.add_argument('--gradient_accumulation_steps',
                                 type=int,
                                 default=1,
                                 help='Number of gradient accumulation steps')
        
        # Cost Volume Feature Fusion options
        self.parser.add_argument('--use_cost_volume_fusion',
                                 help='Use cost volume feature fusion instead of attention-based fusion',
                                 action='store_true')
        self.parser.add_argument('--cost_volume_depth_bins',
                                 type=int,
                                 default=96,
                                 help='Number of depth bins for cost volume')
        self.parser.add_argument('--cost_volume_depth_min',
                                 type=float,
                                 default=0.1,
                                 help='Minimum depth for cost volume')
        self.parser.add_argument('--cost_volume_depth_max',
                                 type=float,
                                 default=100.0,
                                 help='Maximum depth for cost volume')
        # Pose is mandatory for cost volume fusion; keep no option flag
        
        # Feature fusion passes
        self.parser.add_argument('--num_passes',
                                 type=int,
                                 default=2,
                                 help='Number of fusion passes for attention-based feature fusion')
        
        # Register tokens for attention-based fusion
        self.parser.add_argument('--num_register_tokens',
                                 type=int,
                                 default=4,
                                 help='Number of register tokens for attention sink in MultiFrameFeatureFusion')
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
