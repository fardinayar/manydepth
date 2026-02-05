# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import sys
import tempfile

import configargparse
from configargparse import YAMLConfigFileParser
_CONFIGARGPARSE_AVAILABLE = True

try:
    import yaml
except ImportError:
    yaml = None

from config import TrainConfig, load_yaml_with_extends

file_dir = os.path.dirname(__file__)


def _preprocess_config_argv(argv: list) -> list:
    """
    Resolve 'extends' in YAML config files so base config is inherited in-file.
    For each -c/--config <path>, load path with extends; if the file had 'extends',
    write merged YAML to a temp file and replace the path in argv with the temp path.
    Returns list of temp file paths to keep (caller may delete after parse).
    """
    kept_temps = []
    i = 0
    while i < len(argv):
        if argv[i] in ("-c", "--config") and i + 1 < len(argv):
            path = argv[i + 1]
            if os.path.isfile(path) and yaml is not None:
                try:
                    merged, had_extends = load_yaml_with_extends(path)
                    if had_extends:
                        fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="manydepth_config_")
                        os.close(fd)
                        with open(temp_path, "w") as f:
                            yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
                        argv[i + 1] = temp_path
                        kept_temps.append(temp_path)
                except Exception:
                    pass
            i += 2
            continue
        i += 1
    return kept_temps


def _inject_saved_config_argv(argv: list) -> list:
    """
    When --load_weights_folder is set, load opt.json from that run and prepend it
    as a config file so params are loaded from the run's config. User -c and CLI
    override the saved config. Returns list of temp file paths to delete after parse.
    """
    kept_temps = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--load_weights_folder",) and i + 1 < len(argv):
            folder = os.path.expanduser(argv[i + 1])
            opt_path = os.path.join(folder, "opt.json")
            if not os.path.isfile(opt_path):
                opt_path = os.path.join(os.path.dirname(folder), "opt.json")
            if os.path.isfile(opt_path) and yaml is not None:
                try:
                    cfg = TrainConfig.from_json(opt_path)
                    merged = cfg.to_dict()
                    fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="manydepth_saved_config_")
                    os.close(fd)
                    with open(temp_path, "w") as f:
                        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
                    argv[1:1] = ["-c", temp_path]
                    kept_temps.append(temp_path)
                except Exception:
                    pass
            break
        i += 1
    return kept_temps


def _get_parser():
    if _CONFIGARGPARSE_AVAILABLE and YAMLConfigFileParser is not None:
        parser = configargparse.ArgumentParser(
            description="ManyDepth options",
            config_file_parser_class=YAMLConfigFileParser,
            default_config_files=[],
        )
        parser.add_argument("-c", "--config", is_config_file=True, help="path to YAML config file")
    else:
        parser = configargparse.ArgumentParser(description="ManyDepth options")
        parser.add_argument("-c", "--config", type=str, default=None, help="path to YAML config file (requires PyYAML)")

    # PATHS
    parser.add_argument("--data_path", type=str, help="path to the training data",
                        default=os.path.join(file_dir, "kitti_data"))
    parser.add_argument("--log_dir", type=str, help="log directory",
                        default=os.path.join(os.path.expanduser("~"), "tmp"))

    # TRAINING options
    parser.add_argument("--model_name", type=str, help="the name of the folder to save the model in", default="mdp")
    parser.add_argument("--split", type=str, help="which training split to use",
                        choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "cityscapes_preprocessed"],
                        default="eigen_zhou")
    parser.add_argument("--dataset", type=str, help="dataset to train on", default="kitti",
                        choices=["kitti", "kitti_odom", "cityscapes_preprocessed", "gopro"])
    parser.add_argument("--png", help="if set, trains from raw KITTI png files (instead of jpgs)", action="store_true")
    parser.add_argument("--height", type=int, help="input image height", default=182)
    parser.add_argument("--width", type=int, help="input image width", default=630)
    parser.add_argument("--disparity_smoothness", type=float, help="disparity smoothness weight", default=0.0)
    parser.add_argument("--scales", nargs="+", type=int, help="scales used in the loss", default=[0])
    parser.add_argument("--max_depth", type=float, help="maximum depth", default=300.0)
    parser.add_argument("--frame_ids", nargs="+", type=int, help="frames to load", default=[0, -1, 1])

    # OPTIMIZATION options
    parser.add_argument("--batch_size", type=int, help="batch size", default=12)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=5)
    parser.add_argument("--pytorch_random_seed", default=None, type=int)
    parser.add_argument("--max_grad_norm", type=float, help="maximum gradient norm for clipping (0 to disable)", default=1)
    parser.add_argument("--warmup_steps", type=int, help="number of warmup steps for learning rate (0 to disable)", default=1000)

    # ABLATION options
    parser.add_argument("--avg_reprojection", help="if set, uses average reprojection loss", action="store_true")
    parser.add_argument("--disable_automasking", help="if set, doesn't do auto-masking", action="store_true")
    parser.add_argument("--weights_init", type=str, help="pretrained or scratch", default="pretrained",
                        choices=["pretrained", "scratch"])
    parser.add_argument("--num_matching_frames", type=int, help="Sets how many previous frames to load to build the cost volume", default=1)
    parser.add_argument("--disable_motion_masking", help="If set, will not apply consistency loss in regions where the cost volume is deemed untrustworthy", action="store_true")
    parser.add_argument("--no_matching_augmentation", action="store_true",
                        help="If set, will not apply static camera augmentation or zero cost volume augmentation during training")
    parser.add_argument("--no_temporal_fusion", action="store_true",
                        help="If set, will not use temporal fusion in the depth decoder")
    parser.add_argument("--no_lora", action="store_true",
                        help="If set, will not use LoRA in the depth decoder and encoder")
    parser.add_argument("--no_consistency_loss", action="store_true",
                        help="If set, will not use consistency loss in the depth decoder")
    parser.add_argument("--no_loss_dynamic_weight", action="store_true",
                        help="If set, will not use dynamic weight for the loss")

    # SYSTEM options
    parser.add_argument("--no_cuda", help="if set disables CUDA", action="store_true")
    parser.add_argument("--num_workers", type=int, help="number of dataloader workers", default=12)

    # LOADING options
    parser.add_argument("--load_weights_folder", type=str, help="name of model to load")
    parser.add_argument("--mono_weights_folder", type=str)
    parser.add_argument("--models_to_load", nargs="+", type=str, help="models to load",
                        default=["encoder", "depth", "pose_encoder", "pose"])

    # LOGGING options
    parser.add_argument("--log_frequency", type=int, help="number of batches between each tensorboard log", default=50)
    parser.add_argument("--save_frequency", type=int, help="number of epochs between each save", default=1)
    parser.add_argument("--save_intermediate_models", help="if set, save the model each time we log to tensorboard", action="store_true")

    # EVALUATION options
    parser.add_argument("--disable_median_scaling", help="if set disables median scaling in evaluation", action="store_true")
    parser.add_argument("--pred_depth_scale_factor", type=float, help="if set multiplies predictions by this number", default=1)
    parser.add_argument("--ext_disp_to_eval", type=str, help="optional path to a .npy disparities file to evaluate")
    parser.add_argument("--eval_split", type=str, default="eigen",
                        choices=["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "cityscapes"],
                        help="which split to run eval on")
    parser.add_argument("--save_pred_disps", help="if set saves predicted disparities", action="store_true")
    parser.add_argument("--no_eval", help="if set disables evaluation", action="store_true")
    parser.add_argument("--eval_eigen_to_benchmark", action="store_true",
                        help="if set assume we are loading eigen results from npy but we want to evaluate using the new benchmark.")
    parser.add_argument("--eval_teacher", action="store_true", help="If set, the teacher network will be evaluated")

    # DEPTH_ANYTHING options
    parser.add_argument("--depth_anything_encoder", type=str, choices=["vits", "vitb", "vitl", "vitg"], default="vits")
    parser.add_argument("--depth_anything_checkpoint_dir", type=str, default="checkpoints",
                        help="directory containing Depth Anything checkpoints (depth_anything_v2_<encoder>.pth)")
    parser.add_argument("--encoder_lr_coef", type=float, default=1.0)
    parser.add_argument("--fusion_lr_coef", type=float, default=4.0, help="Learning rate multiplier for feature fusion parameters")
    parser.add_argument("--g2s", help="use g2s loss", action="store_true")
    parser.add_argument("--data_percent", type=float, default=100.0)
    parser.add_argument("--pose_from_scratch", action="store_true",
                        help="If set, the pose encoder and decoder will be initialized randomly")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")

    # Cost Volume Feature Fusion options
    parser.add_argument("--use_cost_volume_fusion", action="store_true",
                        help="Use cost volume feature fusion instead of attention-based fusion")
    parser.add_argument("--cost_volume_depth_bins", type=int, default=96, help="Number of depth bins for cost volume")
    parser.add_argument("--cost_volume_depth_min", type=float, default=0.1, help="Minimum depth for cost volume")
    parser.add_argument("--cost_volume_depth_max", type=float, default=100.0, help="Maximum depth for cost volume")
    parser.add_argument("--num_passes", type=int, default=2, help="Number of fusion passes for attention-based feature fusion")
    parser.add_argument("--num_register_tokens", type=int, default=4,
                        help="Number of register tokens for attention sink in MultiFrameFeatureFusion")

    # Ablation: LoRA (encoder/decoder)
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank for encoder/decoder")
    parser.add_argument("--lora_alpha", type=float, default=4.0, help="LoRA alpha for encoder/decoder")
    parser.add_argument("--lora_dropout", type=float, default=0.4, help="LoRA dropout (training)")

    # Ablation: feature fusion
    parser.add_argument("--fusion_neighborhood_size", nargs="+", type=int, default=[3, 15],
                        help="Attention neighborhood (height width) or single int for square; empty for global")
    parser.add_argument("--fusion_num_scales", type=int, default=4, help="Per-scale LoRA in fusion block")
    parser.add_argument("--fusion_lora_rank", type=int, default=32, help="Fusion block LoRA rank")
    parser.add_argument("--fusion_lora_alpha", type=float, default=4.0, help="Fusion block LoRA alpha")
    parser.add_argument("--fusion_dropout", type=float, default=0.0, help="Fusion block dropout")
    parser.add_argument("--fusion_drop_path", type=float, default=0.0, help="Fusion stochastic depth")
    parser.add_argument("--cost_volume_fusion_dropout", type=float, default=0.2, help="Cost volume fusion dropout")

    # Ablation: scheduler
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="LR scheduler decay factor")
    parser.add_argument("--scheduler_step_epochs", type=int, default=2, help="Step LR every N epochs")

    # Ablation: pose encoder
    parser.add_argument("--pose_encoder_num_layers", type=int, default=18, help="ResNet layers for pose encoder")

    return parser


class MonodepthOptions:
    def __init__(self):
        self.parser = _get_parser()

    def parse(self) -> TrainConfig:
        # Resolve 'extends' in config file(s) so base is inherited in-file, not via CLI
        _temp_configs = _preprocess_config_argv(sys.argv)
        # When load_weights_folder is set, inject that run's opt.json as first config (overrideable by -c/CLI)
        _temp_configs.extend(_inject_saved_config_argv(sys.argv))
        try:
            args = self.parser.parse_args()
        finally:
            for p in _temp_configs:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        d = vars(args)
        # When using plain argparse + --config, load YAML first and override with CLI
        if not _CONFIGARGPARSE_AVAILABLE and d.get("config"):
            try:
                cfg = TrainConfig.from_yaml(d["config"])
                for k, v in d.items():
                    if k == "config" or v is None:
                        continue
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                return cfg
            except Exception:
                pass
        # Convert fusion_neighborhood_size list to tuple/None
        if d.get("fusion_neighborhood_size") is not None:
            ne = d["fusion_neighborhood_size"]
            if len(ne) == 0:
                d["fusion_neighborhood_size"] = None
            elif len(ne) == 1:
                d["fusion_neighborhood_size"] = (int(ne[0]), int(ne[0]))
            else:
                d["fusion_neighborhood_size"] = tuple(int(x) for x in ne)
        return TrainConfig.from_dict(d)
