# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

import networks
from layers import transformation_from_parameters
from networks.replace_with_lora import replace_qkv_with_mergedlinear, replace_conv_with_loraconv

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for ManyDepth models.')

    parser.add_argument('--target_image_path', type=str,
                        help='path to a test image to predict for', required=True)
    parser.add_argument('--source_image_path', type=str,
                        help='path to a previous image in the video sequence', required=True)
    parser.add_argument('--model_path', type=str,
                        help='path to a folder of weights to load', required=True)
    parser.add_argument('--mode', type=str, default='multi', choices=('multi', 'mono'),
                        help='"multi" or "mono". If set to "mono" then the network is run without '
                             'the source image, e.g. as described in Table 5 of the paper.',
                        required=False)
    return parser.parse_args()


def load_and_preprocess_image(image_path, resize_width, resize_height):
    image = pil.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image = image.resize((resize_width, resize_height), pil.LANCZOS)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if torch.cuda.is_available():
        return image.cuda(), (original_height, original_width)
    return image, (original_height, original_width)




def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_path is not None, \
        "You must specify the --model_path parameter"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", args.model_path)

    # Loading pretrained model
    print("   Loading pretrained encoder")
    encoder_dict = torch.load(os.path.join(args.model_path, "encoder.pth"), map_location=device)
    encoder = networks.ManyDepthAnythingEncoder()

    filtered_dict_enc = {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    print("   Loading pretrained decoder")
    HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
    depth_decoder = networks.ManyDepthAnythingDecoder(
                matching_height=HEIGHT // 14, matching_width=WIDTH //14)

    encoder = replace_qkv_with_mergedlinear(encoder)
    depth_decoder = replace_conv_with_loraconv(depth_decoder)
    
    loaded_dict = torch.load(os.path.join(args.model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    print("   Loading pose network")
    pose_enc_dict = torch.load(os.path.join(args.model_path, "pose_encoder.pth"),
                               map_location=device)
    pose_dec_dict = torch.load(os.path.join(args.model_path, "pose.pth"), map_location=device)

    pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
    pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                    num_frames_to_predict_for=2)

    pose_enc.load_state_dict(pose_enc_dict, strict=True)
    pose_dec.load_state_dict(pose_dec_dict, strict=True)

    # Setting states of networks
    encoder.eval()
    depth_decoder.eval()
    pose_enc.eval()
    pose_dec.eval()
    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()
        pose_enc.cuda()
        pose_dec.cuda()

    # Load input data
    input_image, original_size = load_and_preprocess_image(args.target_image_path,
                                                           resize_width=encoder_dict['width'],
                                                           resize_height=encoder_dict['height'])

    source_image, _ = load_and_preprocess_image(args.source_image_path,
                                                resize_width=encoder_dict['width'],
                                                resize_height=encoder_dict['height'])


    with torch.no_grad():

        # Estimate poses
        pose_inputs = [source_image, input_image]
        pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
        axisangle, translation = pose_dec(pose_inputs)
        pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

        if args.mode == 'mono':
            pose *= 0  # zero poses are a signal to the encoder not to construct a cost volume
            source_image *= 0

        # Estimate depth
        features, lookup_features = encoder(input_image,
                                         source_image.unsqueeze(1))

        output, _ = depth_decoder(features, lookup_features, HEIGHT // 14, WIDTH // 14)
        
        sigmoid_output = output.sigmoid()
        
        sigmoid_output_resized = torch.nn.functional.interpolate(
            sigmoid_output, original_size, mode="bilinear", align_corners=False)
        sigmoid_output_resized = sigmoid_output_resized.cpu().numpy()[:, 0]

        # Saving numpy file
        directory, filename = os.path.split(args.target_image_path)
        output_name = os.path.splitext(filename)[0]
        name_dest_npy = os.path.join(directory, "{}_disp_{}.npy".format(output_name, args.mode))
        np.save(name_dest_npy, sigmoid_output.cpu().numpy())

        # Saving colormapped depth image and cost volume argmin
        toplot = sigmoid_output_resized.squeeze()
        normalizer = mpl.colors.Normalize(vmin=toplot.min(), vmax=np.percentile(toplot, 95))
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(toplot)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        name_dest_im = os.path.join(directory,
                                    "{}_{}_{}.jpeg".format(output_name, 'disp', args.mode))
        im.save(name_dest_im)

        print("-> Saved output image to {}".format(name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
