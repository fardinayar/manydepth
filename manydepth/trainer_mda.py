# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import subprocess
import sys
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
os.environ.setdefault("MPLCONFIGDIR", "/tmp/manydepth_matplotlib")

import numpy as np
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import readlines, sec_to_hm_str
from layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss
import loralib as lora

import datasets, networks
import matplotlib
matplotlib.use("Agg")  # headless backend for batch/PBS (avoids hang when no DISPLAY)
import matplotlib.pyplot as plt
from PIL import Image
from networks.replace_with_lora import replace_mlp_with_lora, replace_conv_with_loraconv
_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.prediction_output_path = os.path.join("output_preds", self.opt.model_name)

        # checking height and width are multiples of 14
        assert self.opt.height % 14 == 0, "'height' must be a multiple of 14"
        assert self.opt.width % 14 == 0, "'width' must be a multiple of 14"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.epoch = 0
        self.step = 0
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2
        
        # Gradient accumulation parameters
        self.gradient_accumulation_steps = self.opt.gradient_accumulation_steps
        self.effective_batch_size = self.opt.batch_size * self.gradient_accumulation_steps
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.effective_batch_size}")

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"
        if self.opt.num_matching_frames != 1:
            raise ValueError("num_matching_frames must be 1 with the current temporal fusion block")

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        self.models["encoder"] = networks.ManyDepthAnythingEncoder(
            encoder_name=self.opt.depth_anything_encoder,
            checkpoint_dir=self.opt.depth_anything_checkpoint_dir,
        )
        if not self.opt.no_lora:
            print("Using LoRA in the encoder")
            self.models['encoder'] = replace_mlp_with_lora(
                self.models["encoder"], r=self.opt.lora_rank, lora_alpha=self.opt.lora_alpha, lora_dropout=self.opt.lora_dropout
            )
            lora.mark_only_lora_as_trainable(self.models['encoder'])
            activated_pos_params = 0
            activated_pos_tensors = 0
            for name, param in self.models["encoder"].named_parameters():
                if name.endswith("pos_embed"):
                    param.requires_grad = True
                    activated_pos_params += param.numel()
                    activated_pos_tensors += 1
            print(
                "Activated gradient for encoder positional embedding: "
                f"{activated_pos_tensors} tensor(s), {activated_pos_params} parameter(s)"
            )
        else:
            print("Training full encoder parameters without LoRA")
            for param in self.models["encoder"].parameters():
                param.requires_grad = True

        if self.opt.use_cls_scale_shift:
            activated_cls_params = 0
            activated_cls_tensors = 0
            for name, param in self.models["encoder"].named_parameters():
                if name.endswith("cls_token"):
                    param.requires_grad = True
                    activated_cls_params += param.numel()
                    activated_cls_tensors += 1
            print(
                "Activated gradient for encoder CLS token: "
                f"{activated_cls_tensors} tensor(s), {activated_cls_params} parameter(s)"
            )

        
        self.models["encoder"].to(self.device)

        model_config = networks.MODEL_CONFIGS[self.opt.depth_anything_encoder]
        
        if self.opt.no_temporal_fusion:
            print("Disabling temporal fusion in the depth decoder")
        
        self.models["depth"] = networks.ManyDepthAnythingDecoder(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            features=model_config['features'],
            patch_h=self.opt.height // 14,
            patch_w=self.opt.width // 14,
            temporal_fusion=not self.opt.no_temporal_fusion,
            num_passes=self.opt.num_passes,
            num_register_tokens=self.opt.num_register_tokens,
            fusion_neighborhood_size=self.opt.fusion_neighborhood_size,
            fusion_num_scales=self.opt.fusion_num_scales,
            fusion_independent_blocks=self.opt.fusion_independent_blocks,
            fusion_mode=self.opt.fusion_mode,
            fusion_lora_rank=self.opt.fusion_lora_rank,
            fusion_lora_alpha=self.opt.fusion_lora_alpha,
            fusion_dropout=self.opt.fusion_dropout,
            fusion_drop_path=self.opt.fusion_drop_path,
            fusion_separate_norms=self.opt.fusion_separate_norms,
            use_cls_scale_shift=self.opt.use_cls_scale_shift,
        )

        depth_anything_path = os.path.join(
            self.opt.depth_anything_checkpoint_dir,
            f'depth_anything_v2_{self.opt.depth_anything_encoder}.pth'
        )
        depthanything_weights = torch.load(depth_anything_path, map_location='cpu')
        depthanything_weights_decoder = {}
        for key, value in depthanything_weights.items():
            if "depth_head" in key:
                depthanything_weights_decoder.update({
                    key.replace('depth_head.', ''): value
                })

        self.models["depth"].load_state_dict(depthanything_weights_decoder, strict=False)
        
        if not self.opt.no_lora:
            print("Using LoRA in the depth decoder")
            self.models['depth'] = replace_conv_with_loraconv(
                self.models["depth"], r=self.opt.lora_rank, lora_alpha=self.opt.lora_alpha, lora_dropout=self.opt.lora_dropout
            )
            lora.mark_only_lora_as_trainable(self.models['depth'])
        print("Enable gradient for output convolution and CLS scale-shift head")

        for name, param in self.models['depth'].named_parameters():
            if 'cls_scale_shift' in name:
                param.requires_grad = True
            
        for name, p in self.models['depth'].named_parameters():
            if 'multi_frame_feature_fusion' in name:
                p.requires_grad = True

        self.models["depth"].to(self.device)

        encoder_params = []
        encoder_token_params = []
        for name, param in self.models["encoder"].named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(("cls_token", "pos_embed")):
                encoder_token_params.append(param)
            else:
                encoder_params.append(param)
        if (encoder_params or encoder_token_params) and (
            self.opt.encoder_lr_coef != 0.0 or self.opt.use_cls_scale_shift
        ):
            enc_lr = (
                self.opt.encoder_lr_coef * self.opt.learning_rate
                if self.opt.encoder_lr_coef != 0.0
                else self.opt.learning_rate
            )
            if encoder_params:
                self.parameters_to_train.append({"params": encoder_params, "lr": enc_lr})
            if encoder_token_params:
                encoder_token_lr = enc_lr * self.opt.encoder_token_lr_coef
                self.parameters_to_train.append({
                    "params": encoder_token_params,
                    "lr": encoder_token_lr,
                })
                print(f"Encoder CLS/positional embedding learning rate: {encoder_token_lr}")

        depth_params = []
        fusion_params = []
        for name, param in self.models["depth"].named_parameters():
            if not param.requires_grad:
                continue
            if 'multi_frame_feature_fusion' in name:
                fusion_params.append(param)
            else:
                depth_params.append(param)

        if depth_params:
            self.parameters_to_train.append({'params': depth_params, 'lr': self.opt.learning_rate})
        if fusion_params:
            self.parameters_to_train.append({'params': fusion_params, 'lr': self.opt.learning_rate * self.opt.fusion_lr_coef})
        # Print total number of learnable parameters
        n_encoder = sum(p.numel() for p in self.models["encoder"].parameters() if p.requires_grad)
        n_fusion  = sum(p.numel() for p in fusion_params)
        n_depth   = sum(p.numel() for p in depth_params)
        print(f"Total learnable parameters in encoder: {n_encoder}")
        print(f"Total learnable parameters in depth decoder: {n_depth}")
        print(f"Total learnable parameters in feature fusion: {n_fusion}")
        print(f"Feature fusion learning rate: {self.opt.learning_rate * self.opt.fusion_lr_coef}")
        encoder, decoder = networks.get_da_encoder_decoder(
            encoder_name=self.opt.depth_anything_encoder,
            checkpoint_dir=self.opt.depth_anything_checkpoint_dir,
        )
        self.models["mono_encoder"] = encoder
        self.models["mono_encoder"].to(self.device).eval()

        self.models["mono_depth"] = decoder
        self.models["mono_depth"].to(self.device).eval()
        

        self.models["pose_encoder"] = \
            networks.ResnetEncoder(self.opt.pose_encoder_num_layers, self.opt.pose_weights_init == "pretrained",
                                    num_input_images=self.num_pose_frames)
        self.models["pose"] = \
            networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                    num_input_features=1,
                                    num_frames_to_predict_for=2)

        if not self.opt.pose_from_scratch:
            print("Using pretrained pose encoder and decoder")
            pose_suffix = "R{}".format(self.opt.pose_encoder_num_layers)
            pose_encoder_path = os.path.join("checkpoints", "pose_encoder_{}.pth".format(pose_suffix))
            pose_decoder_path = os.path.join("checkpoints", "pose_{}.pth".format(pose_suffix))
            if not os.path.isfile(pose_encoder_path) or not os.path.isfile(pose_decoder_path):
                raise FileNotFoundError(
                    "Pretrained pose checkpoints for {} were not found. "
                    "Expected {} and {}. Set pose_from_scratch=true to train this pose encoder from scratch.".format(
                        pose_suffix, pose_encoder_path, pose_decoder_path
                    )
                )
            pose_encoder_pretrained_weights = torch.load(pose_encoder_path, map_location='cpu')
            self.models["pose_encoder"].load_state_dict(pose_encoder_pretrained_weights, strict=False)
            
            pose_decoder_pretrained_weights = torch.load(pose_decoder_path, map_location='cpu')
            self.models["pose"].load_state_dict(pose_decoder_pretrained_weights, strict=False)

        
        self.models["pose_encoder"].to(self.device)
        self.models["pose"].to(self.device)
        

        self.parameters_to_train.append({'params': self.models["pose_encoder"].parameters(), 'lr': self.opt.learning_rate})
        self.parameters_to_train.append({'params': self.models["pose"].parameters(), 'lr': self.opt.learning_rate})

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        
        if self.opt.dataset not in datasets_dict:
            raise ValueError(
                "Unknown dataset '{}'. Available datasets: {}".format(
                    self.opt.dataset, ", ".join(sorted(datasets_dict))
                )
            )
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        
        
        percent = self.opt.data_percent
        print(f"Using {percent} percent of the training data")
        if percent <= 0.0 or percent > 100.0:
            raise ValueError("data_percent must be in the range (0, 100]")
        if percent < 100.0:
            full_len = len(train_filenames)
            keep = max(1, int(full_len * percent / 100))
            seed = self.opt.pytorch_random_seed if self.opt.pytorch_random_seed is not None else 1
            rng = random.Random(seed)
            train_filenames = list(train_filenames)
            rng.shuffle(train_filenames)
            subset = train_filenames[:keep]
            repeats, remainder = divmod(full_len, keep)
            train_filenames = subset * repeats + subset[:remainder]
        
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        num_train_batches = num_train_samples // self.opt.batch_size
        if num_train_batches < 1:
            raise ValueError(
                "Training set has {} samples, which is smaller than batch_size={} with drop_last=True".format(
                    num_train_samples, self.opt.batch_size
                )
            )
        steps_per_epoch = math.ceil(num_train_batches / self.gradient_accumulation_steps)
        self.num_total_steps = steps_per_epoch * self.opt.num_epochs
        print('Total number of steps: ', self.num_total_steps, "Total number of epochs:", self.opt.num_epochs)
        
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate, weight_decay=1e-4)

        # Warmup configuration
        self.warmup_steps = self.opt.warmup_steps
        if self.opt.cosine_min_lr_factor < 0.0 or self.opt.cosine_min_lr_factor > 1.0:
            raise ValueError("cosine_min_lr_factor must be in the range [0, 1]")
        cosine_steps = max(1, self.num_total_steps - self.warmup_steps)
        min_lr_factor = self.opt.cosine_min_lr_factor

        def cosine_lr_factor(step):
            progress = min(float(step) / float(cosine_steps), 1.0)
            return min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.model_lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.model_optimizer, lr_lambda=cosine_lr_factor
        )
        print(
            "Using cosine LR scheduler over {} steps with min factor {:.4f}".format(
                cosine_steps, min_lr_factor
            )
        )
        # Store base learning rates for each param group for warmup
        self.base_lrs = [group['lr'] for group in self.model_optimizer.param_groups]
        if self.warmup_steps > 0:
            print(f"Using learning rate warmup for {self.warmup_steps} steps")
            # Start with very small learning rate
            for param_group in self.model_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 1e-6
        
        self.g2s = self.opt.g2s

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=True, img_ext=img_ext, load_gps=self.g2s)
        
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=False, img_ext=img_ext, load_gps=self.g2s)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        
        print(f"Total number of steps: {self.num_total_steps}")
        
        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()
        
        
        self.save_opts()

    def g2s_weight(self):
        maximum_steps = self.opt.g2s_weight_factor * ((self.num_total_steps // self.opt.num_epochs))
        if maximum_steps <= 0:
            return 1
        return (self.step / maximum_steps) ** 3 if self.step <= maximum_steps else 1
    
    def get_warmup_factor(self):
        """Calculate warmup factor for learning rate
        Returns a factor between 0 and 1 based on current step
        """
        if self.warmup_steps <= 0 or self.step >= self.warmup_steps:
            return 1.0
        return float(self.step) / float(self.warmup_steps)

    def channel_gate_stats(self):
        """Return effective bounded channel-gate stats for console logging."""
        fusion_blocks = getattr(self.models["depth"], "multi_frame_feature_fusion", None)
        if not fusion_blocks:
            return None
        gate_tensors = [
            torch.sigmoid(block.channel_gates.detach().float()).reshape(-1)
            for block in fusion_blocks
            if getattr(block, "channel_gates", None) is not None
        ]
        if not gate_tensors:
            return None
        gates = torch.cat(gate_tensors)
        return gates.mean().item(), gates.min().item(), gates.max().item()

    def patch_gate_stats(self):
        """Return patch-gate weight/bias stats for console logging."""
        fusion_blocks = getattr(self.models["depth"], "multi_frame_feature_fusion", None)
        if not fusion_blocks:
            return None
        weight_tensors = [
            gate.weight.detach().float().reshape(-1)
            for block in fusion_blocks
            if getattr(block, "patch_gates", None) is not None
            for gate in block.patch_gates
        ]
        bias_tensors = [
            gate.bias.detach().float().reshape(-1)
            for block in fusion_blocks
            if getattr(block, "patch_gates", None) is not None
            for gate in block.patch_gates
            if gate.bias is not None
        ]
        if not weight_tensors:
            return None
        weights = torch.cat(weight_tensors)
        if bias_tensors:
            biases = torch.cat(bias_tensors)
            return (
                weights.mean().item(), weights.min().item(), weights.max().item(),
                biases.mean().item(), biases.min().item(), biases.max().item(),
            )
        return (
            weights.mean().item(), weights.min().item(), weights.max().item(),
            0.0, 0.0, 0.0,
        )

    def patch_gate_activation_stats(self):
        """Return post-activation patch-gate stats from the latest batch."""
        fusion_blocks = getattr(self.models["depth"], "multi_frame_feature_fusion", None)
        if not fusion_blocks:
            return None
        activation_tensors = [
            activation.detach().float().reshape(-1)
            for block in fusion_blocks
            for activation in getattr(block, "_last_patch_gate_activations", ())
            if activation is not None
        ]
        if not activation_tensors:
            return None
        activations = torch.cat(activation_tensors)
        percentiles = torch.quantile(
            activations, activations.new_tensor([0.1, 0.5, 0.9])
        )
        return (
            activations.mean().item(),
            activations.min().item(),
            percentiles[0].item(),
            percentiles[1].item(),
            percentiles[2].item(),
            activations.max().item(),
        )

    

    def set_train(self):
        """Convert all models to training mode
        """
        for k, m in self.models.items():
            if k not in ['mono_encoder', 'mono_depth']:
                m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.start_time = time.time()
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
                self.save_model()
                if self.opt.eval_after_each_epoch:
                    self._run_eval_on_test_set(save_folder)
        self.save_opts()  # save final config at end of run

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        if not self.opt.no_cuda:
            torch.cuda.synchronize()

        # Initialize gradient accumulation variables
        accumulated_loss = 0.0
        accumulated_mono_loss = 0.0
        accumulated_scale = 0.0 if self.g2s else None
        accumulation_count = 0

        num_batches = len(self.train_loader)
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses, mono_losses = self.process_batch(inputs, is_train=True)
            
            # Scale by the actual accumulation window size; the last window can be partial.
            window_start = (batch_idx // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
            window_size = min(self.gradient_accumulation_steps, num_batches - window_start)
            scaled_loss = losses["loss"] / window_size
            scaled_loss.backward()
            
            # Accumulate losses for logging
            accumulated_loss += losses["loss"].item()
            accumulated_mono_loss += mono_losses["loss"].item()
            if self.g2s and "scale" in losses:
                accumulated_scale += losses["scale"].item()
            accumulation_count += 1

            # Only update optimizer and scheduler after accumulating gradients.
            should_step = (
                (batch_idx + 1) % self.gradient_accumulation_steps == 0
                or (batch_idx + 1) == num_batches
            )
            if should_step:
                # Apply gradient clipping if enabled (max_grad_norm > 0)
                if self.opt.max_grad_norm > 0:
                    params = [
                        p
                        for group in self.model_optimizer.param_groups
                        for p in group['params']
                        if p.requires_grad
                    ]
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params,
                        self.opt.max_grad_norm
                    )
                    # Store gradient norm for logging
                    losses["grad_norm"] = grad_norm
                    # Log gradient norm for monitoring
                    if self.step % self.opt.log_frequency == 0:
                        print(f"Gradient norm (clipped): {grad_norm:.4f}")
                else:
                    # Calculate gradient norm without clipping for monitoring
                    if self.step % self.opt.log_frequency == 0:
                        total_norm = 0
                        for group in self.parameters_to_train:
                            for p in group['params']:
                                if p.grad is not None:
                                    param_norm = p.grad.data.norm(2)
                                    total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                        losses["grad_norm"] = total_norm
                        print(f"Gradient norm (no clipping): {total_norm:.4f}")
                
                self.model_optimizer.step()
                self.model_optimizer.zero_grad()
                
                # Increment step counter only when optimizer updates
                self.step += 1
                
                # Apply warmup or regular scheduler
                if self.warmup_steps > 0 and self.step <= self.warmup_steps:
                    # During warmup: linearly increase LR from near-zero to base LR
                    warmup_factor = self.get_warmup_factor()
                    for i, param_group in enumerate(self.model_optimizer.param_groups):
                        param_group['lr'] = self.base_lrs[i] * warmup_factor
                else:
                    # After warmup: use the regular scheduler
                    self.model_lr_scheduler.step()
                
                duration = time.time() - before_op_time

                # log less frequently after the first 2000 steps to save time & disk space
                log_step = self.step % self.opt.log_frequency == 0

                if log_step:
                    # Use accumulated losses for logging
                    avg_loss = accumulated_loss / max(accumulation_count, 1)
                    avg_mono_loss = accumulated_mono_loss / max(accumulation_count, 1)
                    avg_scale = accumulated_scale / max(accumulation_count, 1) if self.g2s else None
                    
                    if self.g2s:
                        self.log_time(batch_idx, duration, avg_loss, avg_mono_loss, avg_scale)
                    else:
                        self.log_time(batch_idx, duration, avg_loss, avg_mono_loss)

                    # Update losses with averaged values for logging
                    losses["loss"] = torch.tensor(avg_loss, device=losses["loss"].device)
                    mono_losses["loss"] = torch.tensor(avg_mono_loss, device=mono_losses["loss"].device)
                    if self.g2s and avg_scale is not None:
                        losses["scale"] = torch.tensor(avg_scale, device=losses["scale"].device)
                    
                    self.log("train", inputs, outputs, losses, mono_losses)
                    self.val()
                    
                # Reset accumulation variables only after logging
                if log_step:
                    accumulated_loss = 0.0
                    accumulated_mono_loss = 0.0
                    accumulated_scale = 0.0 if self.g2s else None
                    accumulation_count = 0

                if self.opt.save_intermediate_models:
                    self.save_model(save_step=True)

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        pose_pred = self.predict_poses(inputs)

        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)

        # Grab frames and stack for input to the multi-frame network.
        lookup_frames = [inputs[('color_aug_norm', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        # Apply static-frame and missing-fusion augmentation.
        batch_size = len(lookup_frames)
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        fusion_mask = torch.ones([batch_size, 1, 1]).to(self.device).float()
        if is_train:
            for batch_idx in range(batch_size):
                rand_num = random.random()
                # static camera augmentation -> overwrite lookup frames with current frame
                if rand_num < 0.1:
                    replace_frames = \
                        [inputs[('color_aug_norm', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                    replace_frames = torch.stack(replace_frames, 0)
                    lookup_frames[batch_idx] = replace_frames
                    augmentation_mask[batch_idx] += 1
                # Disable temporal residuals to simulate unavailable matching evidence.
                elif rand_num < 0.2:
                    fusion_mask[batch_idx] *= 0
                    augmentation_mask[batch_idx] += 1
        outputs['augmentation_mask'] = augmentation_mask

        with torch.no_grad():
            input_image = inputs["color_aug_norm", 0, 0]
            patch_h, patch_w = input_image.shape[-2] // 14, input_image.shape[-1] // 14
            feats = self.models["mono_encoder"].get_intermediate_layers(input_image, self.models["mono_encoder"].intermediate_layer_idx, return_class_token=True)
            monodepth, _ = self.models['mono_depth'](feats, patch_h, patch_w)
        monodepth = {("disp", 0): F.relu(monodepth)}
        mono_outputs.update(monodepth)

        self.generate_images_pred(inputs, mono_outputs)
        mono_losses = self.compute_losses(inputs, mono_outputs, is_multi=False)

        # update multi frame outputs dictionary with single frame outputs
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]

        # multi frame path
        encoder_lookup_frames = None if self.opt.no_temporal_fusion else lookup_frames
        need_encoder_grad = (
            self.opt.encoder_lr_coef != 0.0 or self.opt.use_cls_scale_shift
        )
        if not need_encoder_grad:
            with torch.no_grad():
                features, lookup_features = self.models["encoder"](
                    inputs["color_aug_norm", 0, 0], encoder_lookup_frames)
        else:
            features, lookup_features = self.models["encoder"](
                inputs["color_aug_norm", 0, 0], encoder_lookup_frames)

        depth, _ = self.models["depth"](
            features,
            lookup_features,
            fusion_mask=fusion_mask,
        )

        depth =  F.relu(depth)
        outputs.update({("disp", 0): depth})

        self.generate_images_pred(inputs, outputs, is_multi=True)
        losses = self.compute_losses(inputs, outputs, is_multi=True)

        return outputs, losses, mono_losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    
                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            # now we need poses for matching - compute without gradients
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.matching_ids}
            with torch.no_grad():
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.matching_ids[1:]:
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                    missing_key = ("missing_frame", fi)
                    if missing_key in inputs:
                        pose[inputs[missing_key].bool()] *= 0

                    inputs[('relative_pose', fi)] = pose
        else:
            raise NotImplementedError

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            try:
                inputs = next(self.val_iter)
            except StopIteration:
                self.set_train()
                return

        with torch.no_grad():
            outputs, losses, mono_losses = self.process_batch(inputs)

            self.log("val", inputs, outputs, losses, mono_losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.max_depth)

            # TODO
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]


                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)


        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    @staticmethod
    def compute_depth_edge_mask(disp, threshold=0.15, dilation=1):
        """Mask pixels near detached disparity discontinuities."""
        threshold = float(threshold)
        if threshold <= 0.0:
            return torch.zeros_like(disp[:, :1])

        disp = disp.detach()
        if disp.shape[1] != 1:
            disp = disp.mean(dim=1, keepdim=True)

        norm_disp = disp / (disp.mean(dim=(2, 3), keepdim=True) + 1e-7)
        edge_score = torch.zeros_like(norm_disp)

        grad_x = torch.abs(norm_disp[:, :, :, 1:] - norm_disp[:, :, :, :-1])
        grad_y = torch.abs(norm_disp[:, :, 1:, :] - norm_disp[:, :, :-1, :])

        edge_score[:, :, :, 1:] = torch.maximum(edge_score[:, :, :, 1:], grad_x)
        edge_score[:, :, :, :-1] = torch.maximum(edge_score[:, :, :, :-1], grad_x)
        edge_score[:, :, 1:, :] = torch.maximum(edge_score[:, :, 1:, :], grad_y)
        edge_score[:, :, :-1, :] = torch.maximum(edge_score[:, :, :-1, :], grad_y)

        edge_mask = (edge_score > threshold).float()
        dilation = int(dilation)
        if dilation > 0:
            kernel_size = 2 * dilation + 1
            edge_mask = F.max_pool2d(
                edge_mask, kernel_size=kernel_size, stride=1, padding=dilation
            )
        return edge_mask

    @staticmethod
    def pose_translation_norm(translation):
        """Return one translation magnitude per sample."""
        translation = translation[:, 0].reshape(translation.shape[0], -1)
        if translation.shape[1] != 3:
            raise RuntimeError(
                "Expected pose translation to contain 3 values per sample, got shape {}".format(
                    tuple(translation.shape)
                )
            )
        return torch.norm(translation, dim=1)

    @staticmethod
    def patch_ssi_loss(prediction, target, margin, patch_size=16, variance_threshold=1e-3):
        """Scale/shift-invariant patch loss for relative disparity supervision."""
        stride = patch_size // 2
        prediction_patches = F.unfold(prediction, kernel_size=patch_size, stride=stride)
        target_patches = F.unfold(target, kernel_size=patch_size, stride=stride)

        prediction_mean = prediction_patches.mean(dim=1, keepdim=True)
        target_mean = target_patches.mean(dim=1, keepdim=True)
        prediction_var = ((prediction_patches - prediction_mean) ** 2).mean(dim=1, keepdim=True)
        target_var = ((target_patches - target_mean) ** 2).mean(dim=1, keepdim=True)

        prediction_norm = (prediction_patches - prediction_mean) / torch.sqrt(prediction_var + 1e-5)
        target_norm = (target_patches - target_mean) / torch.sqrt(target_var + 1e-5)

        patch_loss = F.relu(torch.abs(prediction_norm - target_norm).mean(dim=1) - float(margin))
        prediction_norm_grid = prediction_norm.reshape(
            prediction.shape[0], prediction.shape[1], patch_size, patch_size, -1)
        target_norm_grid = target_norm.reshape(
            target.shape[0], target.shape[1], patch_size, patch_size, -1)
        grad_x_diff = torch.abs(
            (prediction_norm_grid[:, :, :, 1:, :] - prediction_norm_grid[:, :, :, :-1, :])
            - (target_norm_grid[:, :, :, 1:, :] - target_norm_grid[:, :, :, :-1, :])
        )
        grad_y_diff = torch.abs(
            (prediction_norm_grid[:, :, 1:, :, :] - prediction_norm_grid[:, :, :-1, :, :])
            - (target_norm_grid[:, :, 1:, :, :] - target_norm_grid[:, :, :-1, :, :])
        )
        grad_x_loss = F.relu(grad_x_diff).mean(dim=(1, 2, 3))
        grad_y_loss = F.relu(grad_y_diff).mean(dim=(1, 2, 3))
        patch_loss = 0 * patch_loss + 0.5 * (grad_x_loss + grad_y_loss)
        valid_mask = torch.min(prediction_var, target_var).squeeze(1) > variance_threshold
        loss = (patch_loss * valid_mask).sum(dim=-1) / (valid_mask.sum(dim=-1) + 1e-7)

        return loss.mean(), patch_loss

    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                      keepdim=True)

            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
            outputs[("reprojection_loss_map", scale)] = reprojection_loss.detach()

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(
                reprojection_loss,
                identity_reprojection_loss,
            )

            if self.opt.ignore_depth_edge_pixels:
                edge_source = outputs.get(("mono_disp", scale), disp)
                ignored_depth_edge_mask = self.compute_depth_edge_mask(
                    edge_source,
                    threshold=self.opt.depth_edge_mask_threshold,
                    dilation=self.opt.depth_edge_mask_dilation,
                )
                if ignored_depth_edge_mask.shape[-2:] != reprojection_loss_mask.shape[-2:]:
                    ignored_depth_edge_mask = F.interpolate(
                        ignored_depth_edge_mask,
                        size=reprojection_loss_mask.shape[-2:],
                        mode="nearest",
                    )
                ignored_depth_edge_mask = ignored_depth_edge_mask.to(
                    device=reprojection_loss_mask.device,
                    dtype=reprojection_loss_mask.dtype,
                )
                ignored_depth_edge_mask = ignored_depth_edge_mask * (
                    reprojection_loss_mask > 0
                ).to(dtype=reprojection_loss_mask.dtype)
                reprojection_loss_mask = reprojection_loss_mask * (
                    1.0 - ignored_depth_edge_mask
                )
            else:
                ignored_depth_edge_mask = torch.zeros_like(
                    reprojection_loss_mask, device=reprojection_loss.device
                )
            outputs[("depth_edge_mask", scale)] = ignored_depth_edge_mask.detach()

            # ------------------------------------------------------------------
            # Optionally ignore the highest-loss pixels among the already valid
            # pixels, and visualize which pixels were ignored.
            # ------------------------------------------------------------------
            if self.opt.ignore_high_low_loss_pixels:
                top_percent = 0.1  # ignore top 20% highest-loss pixels over the full batch

                valid = reprojection_loss_mask > 0
                ignored_high_loss_mask = torch.zeros_like(
                    reprojection_loss_mask, device=reprojection_loss.device
                )

                # Collect ALL valid pixels from the whole batch
                valid_vals = reprojection_loss[valid].detach().view(-1)

                if valid_vals.numel() > 0:
                    n = valid_vals.numel()

                    # Mask high-loss pixels
                    if top_percent > 0.0:
                        high_percentile = 1.0 - top_percent
                        # kthvalue is 1-indexed; pick the (percentile*n)-th smallest as threshold
                        k_high = max(1, min(int(high_percentile * n) + 1, n))
                        high_threshold, _ = torch.kthvalue(valid_vals, k_high)

                        # Mark high-loss pixels across the whole batch using the same threshold
                        high = (reprojection_loss >= high_threshold) & valid
                        ignored_high_loss_mask[high] = 1.0

                    # Remove high-loss pixels from the effective mask
                    reprojection_loss_mask = (
                        reprojection_loss_mask * (1.0 - ignored_high_loss_mask)
                    )

                # Store visualization of ignored pixels (per-scale)
                outputs[("high_loss_mask", scale)] = ignored_high_loss_mask.detach()
            else:
                outputs[("high_loss_mask", scale)] = torch.zeros_like(
                    reprojection_loss_mask, device=reprojection_loss.device
                ).detach()


            # Apply (possibly refined) mask to reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (
                reprojection_loss_mask.sum() + 1e-7
            )



            # consistency loss:
            if is_multi and not self.opt.no_consistency_loss:

                patch_size = 16
                multi_disp = outputs[("disp", scale)]
                mono_disp = outputs[("mono_disp", scale)].detach()

                ssi_loss, patch_ssi_loss = self.patch_ssi_loss(
                    multi_disp,
                    mono_disp,
                    self.opt.consistency_disp_margin,
                    patch_size=patch_size,
                )
                
                # Store patch-wise SSI loss for visualization
                b, _, h, w = multi_disp.shape
                stride = patch_size // 2
                patch_grid_h = (h - patch_size) // stride + 1
                patch_grid_w = (w - patch_size) // stride + 1
                patch_ssi_loss_spatial = patch_ssi_loss.view(b, patch_grid_h, patch_grid_w)
                outputs[("ssi_loss", scale)] = patch_ssi_loss_spatial
                
                # Combine losses
                if not self.opt.no_loss_dynamic_weight:
                    ssi_weight = (1-self.g2s_weight())/10
                else:
                    ssi_weight = 0.01
                
                consistency_loss = (ssi_weight * ssi_loss)
                
                losses['consistency_loss/{}'.format(scale)] = consistency_loss
            else:
                consistency_loss = 0

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            loss += reprojection_loss + consistency_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            if not is_multi:
                loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        

        total_loss /= self.num_scales



        if self.g2s and is_multi:
            #TRANSLATIONS
            t12 = self.pose_translation_norm(outputs[("translation", 0, -1)])
            t23 = self.pose_translation_norm(outputs[("translation", 0, 1)])

            eps = 1e-4
            min_gps_motion = 0.05

            pred_t = torch.stack([t12, t23], dim=1).clamp_min(eps)
            gps12 = inputs["gps12"].float().view(-1)
            gps23 = inputs["gps23"].float().view(-1)
            gps_t = torch.stack([gps12, gps23], dim=1).to(
                device=pred_t.device, dtype=pred_t.dtype
            )

            valid_g2s = gps_t > min_gps_motion
            clamped_gps_t = gps_t.clamp_min(eps)
            log_scale_error = (torch.log(pred_t) - torch.log(clamped_gps_t)).clamp(-5.0, 5.0)

            if valid_g2s.any().item():
                g2s_loss = log_scale_error[valid_g2s].pow(2).mean()
                scale_values = (clamped_gps_t / pred_t.detach())[valid_g2s]
                losses["scale"] = scale_values.mean()
            else:
                g2s_loss = pred_t.new_tensor(0.0)
                losses["scale"] = pred_t.new_tensor(0.0)
            
            if not self.opt.no_loss_dynamic_weight:
                total_loss += self.g2s_weight() * g2s_loss 
            else:
                total_loss += g2s_loss
            losses["g2s_loss"] = g2s_loss.detach()
            losses["g2s_valid_ratio"] = valid_g2s.float().mean()
            
        losses["loss"] = total_loss

        return losses

    def log_time(self, batch_idx, duration, loss, mono_loss, scale=None):
        """Print a logging statement to the terminal."""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        )

        print_string = (
            "epoch {:>3} | batch {:>6} | examples/s: {:5.1f} | "
            "loss: {:.5f} | mono_loss: {:.5f} | time elapsed: {} | time left: {}"
        )
        print_data = [
            self.epoch, batch_idx, samples_per_sec, loss, mono_loss,
            sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)
        ]

        if scale is not None:
            print_string += " | scale: {}"
            print_data.append(scale)
        
        # Add g2s weight if g2s is enabled
        if self.g2s:
            g2s_w = self.g2s_weight()
            print_string += " | g2s_weight: {:.5f}"
            print_data.append(g2s_w)
            
        # Add gradient accumulation info
        if self.gradient_accumulation_steps > 1:
            print_string += " | eff_batch: {}"
            print_data.append(self.effective_batch_size)
        
        # Add learning rate info (especially useful during warmup)
        current_lr = self.model_optimizer.param_groups[0]['lr']
        print_string += " | lr: {:.2e}"
        print_data.append(current_lr)
        
        # Add warmup status
        if self.warmup_steps > 0 and self.step <= self.warmup_steps:
            print_string += " | warmup: {}/{}"
            print_data.extend([self.step, self.warmup_steps])

        channel_gate_stats = self.channel_gate_stats()
        if channel_gate_stats is not None:
            print_string += " | channel_gate mean/min/max: {:.4f}/{:.4f}/{:.4f}"
            print_data.extend(channel_gate_stats)
        else:
            print_string += " | channel_gate: none"

        patch_gate_stats = self.patch_gate_stats()
        if patch_gate_stats is not None:
            print_string += " | patch_gate w mean/min/max: {:.4f}/{:.4f}/{:.4f}"
            print_string += " | patch_gate logit b mean/min/max: {:.4f}/{:.4f}/{:.4f}"
            print_data.extend(patch_gate_stats)
        else:
            print_string += " | patch_gate: none"

        patch_gate_activation_stats = self.patch_gate_activation_stats()
        if patch_gate_activation_stats is not None:
            print_string += " | patch_gate act mean/min/p10/p50/p90/max: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}"
            print_data.extend(patch_gate_activation_stats)

        print(print_string.format(*print_data))


    def log(self, mode, inputs, outputs, losses, mono_losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        for l, v in mono_losses.items():
            writer.add_scalar("mono_{}".format(l), v, self.step)

        patch_gate_activation_stats = self.patch_gate_activation_stats()
        if patch_gate_activation_stats is not None:
            for name, value in zip(
                ("mean", "min", "p10", "p50", "p90", "max"),
                patch_gate_activation_stats,
            ):
                writer.add_scalar("fusion/patch_gate_activation_{}".format(name), value, self.step)
            
        # Log gradient accumulation info
        if mode == "train":
            writer.add_scalar("gradient_accumulation_steps", self.gradient_accumulation_steps, self.step)
            writer.add_scalar("effective_batch_size", self.effective_batch_size, self.step)

        batch_size = inputs[("color", 0, 0)].shape[0]
        for j in range(min(4, batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)
                if s == 0 and frame_id != 0:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)

            disp = colormap(outputs[("disp", s)][j, 0])
            writer.add_image(
                "disp_multi_{}/{}".format(s, j),
                disp, self.step)

            disp = colormap(outputs[('mono_disp', s)][j, 0])
            writer.add_image(
                "disp_mono/{}".format(j),
                disp, self.step)

            # Log SSI loss if available (only for multi-frame)
            if ("ssi_loss", s) in outputs:
                ssi_loss_img = colormap(outputs[("ssi_loss", s)][j])
                writer.add_image(
                    "ssi_loss_{}/{}".format(s, j),
                    ssi_loss_img, self.step)

            # Log high loss mask visualization
            if ("high_loss_mask", s) in outputs:
                high_loss_mask_img = colormap(outputs[("high_loss_mask", s)][j, 0])
                writer.add_image(
                    "high_loss_mask_{}/{}".format(s, j),
                    high_loss_mask_img, self.step)

            if ("depth_edge_mask", s) in outputs:
                depth_edge_mask_img = colormap(outputs[("depth_edge_mask", s)][j, 0])
                writer.add_image(
                    "depth_edge_mask_{}/{}".format(s, j),
                    depth_edge_mask_img, self.step)

        self.save_depth_prediction_images(mode, outputs)

    def save_depth_prediction_images(self, mode, outputs):
        """Save predicted disparity visualizations matching TensorBoard logs."""
        s = 0
        disp_key = ("disp", s)
        if disp_key not in outputs:
            return

        disp = outputs[disp_key]
        num_images = min(4, disp.shape[0])
        os.makedirs(self.prediction_output_path, exist_ok=True)

        for j in range(num_images):
            disp_vis = colormap(disp[j, 0], torch_transpose=False)
            disp_vis = np.clip(disp_vis * 255.0, 0, 255).astype(np.uint8)
            save_path = os.path.join(
                self.prediction_output_path,
                "{}_step_{:08d}_disp_multi_{:02d}.png".format(mode, self.step, j),
            )
            Image.fromarray(disp_vis).save(save_path)

    def save_opts(self):
        """Save config to run output dir and to models/ as YAML (config.yaml).
        """
        import yaml
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.to_dict() if hasattr(self.opt, 'to_dict') else self.opt.__dict__.copy()
        for dest_dir in (models_dir, self.log_path):
            with open(os.path.join(dest_dir, 'config.yaml'), 'w') as f:
                yaml.dump(to_save, f, default_flow_style=False, sort_keys=False)

    def save_model(self, save_step=False):
        """Save model weights to disk
        """
        if save_step:
            save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,
                                                                                       self.step))
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes and ablation parameters - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['num_passes'] = self.opt.num_passes
                to_save['no_temporal_fusion'] = self.opt.no_temporal_fusion
                to_save['fusion_independent_blocks'] = self.opt.fusion_independent_blocks
                to_save['fusion_mode'] = self.opt.fusion_mode
                to_save['fusion_separate_norms'] = self.opt.fusion_separate_norms

            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("scheduler"))
        torch.save(self.model_lr_scheduler.state_dict(), save_path)

        training_state = {
            "epoch": self.epoch,
            "step": self.step,
            "base_lrs": self.base_lrs,
        }
        save_path = os.path.join(save_folder, "{}.pth".format("training_state"))
        torch.save(training_state, save_path)

    def _run_eval_on_test_set(self, save_folder):
        """Run test-set evaluation (evaluate_depth_mda.py) on the given weights folder."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(repo_root, "manydepth", "evaluate_depth_mda.py")
        cmd = [
            sys.executable, "-u", script,
            "--load_weights_folder", save_folder,
            "--data_path", self.opt.data_path,
            "--eval_split", self.opt.eval_split,
        ]
        print("Running test-set evaluation: {}".format(" ".join(cmd)))
        subprocess.run(cmd, cwd=repo_root, check=True)

    def load_mono_model(self):

        model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

        scheduler_load_path = os.path.join(self.opt.load_weights_folder, "scheduler.pth")
        if os.path.isfile(scheduler_load_path):
            scheduler_dict = torch.load(scheduler_load_path, map_location=self.device)
            if len(scheduler_dict.get("base_lrs", self.base_lrs)) == len(self.base_lrs):
                print("Loading scheduler state")
                self.model_lr_scheduler.load_state_dict(scheduler_dict)
            else:
                print("Can't load scheduler state - optimizer parameter groups changed")

        training_state_path = os.path.join(self.opt.load_weights_folder, "training_state.pth")
        if os.path.isfile(training_state_path):
            print("Loading training state")
            training_state = torch.load(training_state_path, map_location=self.device)
            self.step = int(training_state.get("step", self.step))
            self.epoch = int(training_state.get("epoch", -1)) + 1
            saved_base_lrs = list(training_state.get("base_lrs", self.base_lrs))
            if len(saved_base_lrs) == len(self.base_lrs):
                self.base_lrs = saved_base_lrs
            else:
                print("Can't load base learning rates - optimizer parameter groups changed")
            print("Resuming from epoch {}, global step {}".format(self.epoch, self.step))


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
