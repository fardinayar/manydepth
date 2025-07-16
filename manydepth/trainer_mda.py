# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
import time
import random
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import readlines, sec_to_hm_str
from layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors
import loralib as lora

import datasets, networks
import matplotlib.pyplot as plt
from networks.replace_with_lora import replace_qkv_with_mergedlinear, replace_conv_with_loraconv
_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 14
        assert self.opt.height % 14 == 0, "'height' must be a multiple of 14"
        assert self.opt.width % 14 == 0, "'width' must be a multiple of 14"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        self.models["encoder"] = networks.ManyDepthAnythingEncoder(encoder_name=self.opt.depth_anything_encoder)
        self.models['encoder'] = replace_qkv_with_mergedlinear(self.models["encoder"])
        
        lora.mark_only_lora_as_trainable(self.models['encoder'], bias='all')
        
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.ManyDepthAnythingDecoder(
            matching_height=self.opt.height // 14, matching_width=self.opt.width //14)

        depthanything_weights = torch.load(f'checkpoints/depth_anything_v2_{self.opt.depth_anything_encoder}.pth', map_location='cpu')
        depthanything_weights_decoder = {}
        for key, value in depthanything_weights.items():
            if "depth_head" in key:
                depthanything_weights_decoder.update({
                    key.replace('depth_head.', ''): value
                })

        self.models["depth"].load_state_dict(depthanything_weights_decoder, strict=False)
        self.models['depth'] = replace_conv_with_loraconv(self.models["depth"])
        lora.mark_only_lora_as_trainable(self.models['depth'], bias='all')
        for name, p in self.models['depth'].named_parameters():
            if 'multi_frame_feature_fusion' in name:
                p.requires_grad = True
        
        self.models["depth"].to(self.device)

        if self.opt.encoder_lr_coef != 0.0:
            self.parameters_to_train.append({'params': self.models["encoder"].parameters(), 'lr': self.opt.encoder_lr_coef * self.opt.learning_rate})
        self.parameters_to_train.append({'params': self.models["depth"].parameters(), 'lr': self.opt.learning_rate})

        encoder, decoder = networks.get_da_encoder_decoder(encoder_name=self.opt.depth_anything_encoder)
        self.models["mono_encoder"] = encoder
        self.models["mono_encoder"].to(self.device)

        self.models["mono_depth"] = decoder
        self.models["mono_depth"].to(self.device)
        

        self.models["pose_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                    num_input_images=self.num_pose_frames)
        self.models["pose"] = \
            networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                    num_input_features=1,
                                    num_frames_to_predict_for=2)
        
        '''
        pose_encoder_pretrained_weights = torch.load(f'KITTI_MR/pose_encoder.pth', map_location='cpu')
        self.models["pose_encoder"].load_state_dict(pose_encoder_pretrained_weights, strict=False)
        
        pose_decoder_pretrained_weights = torch.load(f'KITTI_MR/pose.pth', map_location='cpu')
        self.models["pose"].load_state_dict(pose_decoder_pretrained_weights, strict=False)
        '''
        self.models["pose_encoder"].to(self.device)
        self.models["pose"].to(self.device)
        

        self.parameters_to_train.append({'params': self.models["pose_encoder"].parameters(), 'lr': self.opt.learning_rate})
        self.parameters_to_train.append({'params': self.models["pose"].parameters(), 'lr': self.opt.learning_rate})

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, 2, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "gopro": datasets.GoProDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        # TODO
        # Use only 10 percent of the training data
        train_filenames = train_filenames[:int(len(train_filenames) * 1)]
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs


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
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
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

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        
        
        self.save_opts()

    def g2s_weight(self):
            return math.exp(0.01*(self.step - 1*5000)) * 0.1 if self.step <= 1*5000 else 0.1
        
    

    def set_train(self):
        """Convert all models to training mode
        """

        for k, m in self.models.items():
            if k in ['depth', 'encoder']:
                m.train()
            elif k == 'gps_variance' and self.g2s:
                m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            
            
    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses, mono_losses = self.process_batch(inputs, is_train=True)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 1000000
            late_phase = self.step % 10000 == 0

            if early_phase or late_phase:
                if self.g2s:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data, mono_losses["loss"].cpu().data, losses["scale"].cpu().data)
                else:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data, mono_losses["loss"].cpu())

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses, mono_losses)
                self.val()

            if self.opt.save_intermediate_models and late_phase:
                self.save_model(save_step=True)


            self.step += 1

        self.model_lr_scheduler.step()

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

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        # apply static frame and zero cost volume augmentation
        batch_size = len(lookup_frames)
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        if is_train and not self.opt.no_matching_augmentation:
            for batch_idx in range(batch_size):
                rand_num = random.random()
                # static camera augmentation -> overwrite lookup frames with current frame
                if rand_num < 0.25:
                    replace_frames = \
                        [inputs[('color', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                    replace_frames = torch.stack(replace_frames, 0)
                    lookup_frames[batch_idx] = replace_frames
                    augmentation_mask[batch_idx] += 1
                # missing cost volume augmentation -> set all poses to 0, the cost volume will
                # skip these frames
                elif rand_num < 0.5:
                    relative_poses[batch_idx] *= 0
                    augmentation_mask[batch_idx] += 1
        outputs['augmentation_mask'] = augmentation_mask

        with torch.no_grad():
            input_image = inputs["color_aug", 0, 0]
            patch_h, patch_w = input_image.shape[-2] // 14, input_image.shape[-1] // 14
            feats = self.models["mono_encoder"].get_intermediate_layers(input_image, [2, 5, 8, 11], return_class_token=True)
            monodepth, depth_feats = self.models['mono_depth'](feats, patch_h, patch_w)
        monodepth = {("disp", 0): monodepth.sigmoid()}
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
        if self.opt.encoder_lr_coef == 0.0:
            with torch.no_grad():
                features, lookup_features = self.models["encoder"](inputs["color_aug", 0, 0], lookup_frames)
        else:
            features, lookup_features = self.models["encoder"](inputs["color_aug", 0, 0], lookup_frames)
        

        depth, _ = self.models["depth"](features,
                                            lookup_features,
                                            patch_h,
                                            patch_w)
        
        depth =  (depth ).sigmoid()
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

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_feats[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0

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
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses, mono_losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses, mono_losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            # TODO
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                if is_multi:
                    # don't update posenet based on multi frame prediction
                    # TODO
                    pass
                    #T = T.detach()

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
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

    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                identity_reprojection_loss = None

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                if identity_reprojection_loss is not None:
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                             identity_reprojection_loss)
            
            reprojection_loss = reprojection_loss * reprojection_loss_mask 
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)


            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:

                # Get the depth outputs
                multi_depth = outputs[("depth", 0, scale)]
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, scale)].detach()

                # Scale-shift invariant loss between mono_depth and multi_depth
                # Patch-based implementation without using log

                # Define patch size
                patch_size = 8  # Can be adjusted based on input size

                # Unfold into patches
                b, c, h, w = multi_depth.shape
                patches_multi = F.unfold(multi_depth, kernel_size=patch_size, stride=patch_size//2, padding=0)
                patches_mono = F.unfold(mono_depth, kernel_size=patch_size, stride=patch_size//2, padding=0)

                # Reshape to [B, C*patch_size*patch_size, n_patches]
                n_patches = patches_multi.shape[2]
                patches_multi = patches_multi.reshape(b, c*patch_size*patch_size, n_patches)
                patches_mono = patches_mono.reshape(b, c*patch_size*patch_size, n_patches)

                # Calculate mean and variance for each patch
                mean_multi = patches_multi.mean(dim=1, keepdim=True)
                mean_mono = patches_mono.mean(dim=1, keepdim=True)

                var_multi = ((patches_multi - mean_multi)**2).mean(dim=1, keepdim=True)
                var_mono = ((patches_mono - mean_mono)**2).mean(dim=1, keepdim=True)

                # Normalize patches using alpha (scale) and beta (shift)
                alpha_multi = torch.sqrt(var_multi + 1e-7)
                alpha_mono = torch.sqrt(var_mono + 1e-7)

                beta_multi = mean_multi
                beta_mono = mean_mono

                patches_multi_norm = (patches_multi - beta_multi) / alpha_multi
                patches_mono_norm = (patches_mono - beta_mono) / alpha_mono

                # Calculate patch-wise loss
                patch_ssi_loss = torch.abs(patches_multi_norm - patches_mono_norm).mean(dim=1)
                # Mask outlier loss values using statistical threshold
                mean_loss = patch_ssi_loss.mean(dim=-1, keepdim=True)
                std_loss = patch_ssi_loss.std(dim=-1, keepdim=True)
                threshold = mean_loss + 2.0 * std_loss  # 2-sigma threshold
                outlier_mask = patch_ssi_loss <= threshold

                masked_patch_ssi_loss = patch_ssi_loss * outlier_mask
                ssi_loss = masked_patch_ssi_loss.sum(dim=-1) / (outlier_mask.sum(dim=-1) + 1e-7)
                ssi_loss = ssi_loss.mean()
                
                # Store patch-wise SSI loss for visualization
                # Calculate spatial dimensions of patch grid
                patch_grid_h = (h - patch_size) // (patch_size // 2) + 1
                patch_grid_w = (w - patch_size) // (patch_size // 2) + 1
                
                # Reshape patch_ssi_loss to spatial dimensions for visualization
                patch_ssi_loss_spatial = patch_ssi_loss.view(b, patch_grid_h, patch_grid_w)
                outputs[("ssi_loss", scale)] = patch_ssi_loss_spatial
                
                # Combine losses
                ssi_weight = 0.1
                

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
            t12 = torch.norm(outputs[("translation", 0, -1)][:, 0].squeeze(), dim=1)
            t23 = torch.norm(outputs[("translation", 0, 1)][:, 0].squeeze(), dim=1)
            
            
            s1 = inputs["gps12"].float() / t12 
            s2 = inputs["gps23"].float() / t23 
            
            g2s_loss = torch.mean((s1 - 1) ** 2 + (s2 - 1) ** 2)

            total_loss += self.g2s_weight() * g2s_loss
            losses["scale"] = 0.5 * torch.mean(s1 + s2)
            
        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

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

        print(print_string.format(*print_data))


    def log(self, mode, inputs, outputs, losses, mono_losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        for l, v in mono_losses.items():
            writer.add_scalar("mono_{}".format(l), v, self.step)

        # Unnormalization function for ImageNet normalization
        def unnormalize_image(img):
            """Unnormalize image from ImageNet normalization"""
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            return img * std + mean

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in self.opt.frame_ids:
                # Unnormalize color images before writing
                color_img = unnormalize_image(inputs[("color", frame_id, s)][j])
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    color_img.data, self.step)
                if s == 0 and frame_id != 0:
                    # Unnormalize predicted color images before writing
                    color_pred_img = unnormalize_image(outputs[("color", frame_id, s)][j])
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        color_pred_img.data, self.step)

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

        

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

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
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width

            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_mono_model(self):

        model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

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
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


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
