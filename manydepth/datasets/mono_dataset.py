# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import random
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

cv2.setNumThreads(0)


def txt_reader_eigen(path, frame_index):
    with open(path, 'r') as f:
        poses = f.readlines()
        translations = []
        for pose in poses[frame_index - 1: frame_index + 2]:
            pose = pose.rstrip()
            translation = [
                float(pose.split(" ")[3]),
                float(pose.split(" ")[7]),
                float(pose.split(" ")[11])
            ]
            translations.append(translation)
        return translations


def norm(t1, t2):
    diff = 0
    for c1, c2 in zip(t1, t2):
        diff += (c1 - c2) ** 2
    return diff ** 0.5

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def adjust_hue_pil(img, hue_factor):
    """PIL hue adjustment compatible with NumPy 2 negative uint casts."""
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError("hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor))

    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img

    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    hue_offset = int(hue_factor * 255)
    np_h = ((np_h.astype(np.int16) + hue_offset) % 256).astype(np.uint8)

    h = Image.fromarray(np_h, "L")
    return Image.merge("HSV", (h, s, v)).convert(input_mode)


def build_color_jitter(brightness, contrast, saturation, hue):
    """Sample one ColorJitter transform and reuse it for every frame/scale."""
    params = transforms.ColorJitter.get_params(brightness, contrast, saturation, hue)
    if callable(params):
        return params

    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params

    def color_aug(img):
        for fn_id in fn_idx:
            fn_id = int(fn_id)
            if fn_id == 0 and brightness_factor is not None:
                img = transforms.functional.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = transforms.functional.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = transforms.functional.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = adjust_hue_pil(img, hue_factor)
        return img

    return color_aug


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                img_ext='.png',
                load_gps=False
                ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales

        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_gps = load_gps

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                color = self.to_tensor(f)
                inputs[(n, im, i)] = color

                if inputs.get(("missing_frame", im), False):
                    color_aug_tensor = color
                    color_aug_norm = torch.zeros_like(color)
                else:
                    color_aug_tensor = self.to_tensor(color_aug(f))
                    color_aug_norm = self.normalize(color_aug_tensor)

                inputs[(n + "_aug", im, i)] = color_aug_tensor
                inputs[(n + "_aug_norm", im, i)] = color_aug_norm


    def __len__(self):
        return len(self.filenames)

    def load_intrinsics(self, folder, frame_index):
        return self.K.copy()

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("color_aug_norm", <frame_id>, <scale>) for normalized network inputs,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            ("missing_frame", <frame_id>)           for dummy-filled missing frames

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
        if type(self).__name__ in ["CityscapesPreprocessedDataset", "CityscapesEvalDataset"]:
            inputs.update(self.get_colors(folder, frame_index, side, do_flip))
        else:
            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(
                        folder, frame_index, other_side, do_flip)
                    inputs[("missing_frame", i)] = False
                else:
                    try:
                        inputs[("color", i, -1)] = self.get_color(
                            folder, frame_index + i, side, do_flip)
                        inputs[("missing_frame", i)] = False
                    except FileNotFoundError as e:
                        if i != 0:
                            # fill with dummy values
                            inputs[("color", i, -1)] = \
                                Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                            inputs[("missing_frame", i)] = True
                        else:
                            raise FileNotFoundError(f'Cannot find frame - make sure your '
                                                    f'--data_path is set correctly, or try adding'
                                                    f' the --png flag. {e}')


        if self.load_gps:
            gps_path = os.path.join(self.data_path, folder, folder.split("/")[-1] + ".txt")
            translations = txt_reader_eigen(gps_path, frame_index)
            inputs["gps12"] = norm(translations[1], translations[0])
            inputs["gps23"] = norm(translations[1], translations[2])
            
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.load_intrinsics(folder, frame_index)

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = build_color_jitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            del inputs[("color_aug_norm", i, -1)]

        # Add sample index for per-sample variance tracking
        inputs["sample_index"] = index

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
