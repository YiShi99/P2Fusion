import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchinfo import summary

from models.P2Fusion import p2fusion as net
from utils import utils_image as util
from train import Dataset as D

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    parser = argparse.ArgumentParser(description='Inference script for P2Fusion')
    parser.add_argument('--scale', type=int, default=1, help='Scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--model_path', type=str,
                        default='./Model/Infrared_Visible_Fusion/P2Fusion/models/best_E.pth',
                        help='Path to the pretrained model weights')
    parser.add_argument('--root_path', type=str, default='./Dataset/valsets/',
                        help='Input test image root folder')
    parser.add_argument('--dataset', type=str, default='MSRS',
                        help='Dataset name (e.g., MSRS, FMB, tno, roadscene, M3FD_Fusion)')
    parser.add_argument('--A_dir', type=str, default='ir',
                        help='Directory name for modality A (e.g., ir)')
    parser.add_argument('--B_dir', type=str, default='vi',
                        help='Directory name for modality B (e.g., vi)')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--in_channel', type=int, default=3, help='3 means color image and 1 means gray image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DOWNSAMPLE_FACTOR = 4  # The network has a 4x downsampling factor

    # Set up model
    print(f'Target model path: {args.model_path}')
    if os.path.exists(args.model_path):
        print(f'Loading model from {args.model_path}')
    else:
        print(f'Target model path: {args.model_path} does not exist!!!')
        sys.exit()

    model = define_model(args)
    model = model.to(device)

    input_shape = (1, args.in_channel, 512, 512)  # Includes batch_size
    summary(model, input_data=[torch.randn(input_shape).to(device), torch.randn(input_shape).to(device)])
    model.eval()

    # --- Parameter count statistics ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "=" * 50)
    print(f"[*] Total Parameters: {total_params / 1e6:.3f} M")
    print(f"[*] Trainable Parameters: {trainable_params / 1e6:.3f} M")
    print("=" * 50 + "\n")

    # Setup folder and path
    save_dir, window_size = setup(args)
    REQUIRED_MULTIPLE = window_size * DOWNSAMPLE_FACTOR  # e.g., 8 * 4 = 32
    a_dir = os.path.join(args.root_path, args.dataset, args.A_dir).replace('\\', '/')
    b_dir = os.path.join(args.root_path, args.dataset, args.B_dir).replace('\\', '/')
    os.makedirs(save_dir, exist_ok=True)

    test_opt = {
        "dataroot_A": a_dir,
        "dataroot_B": b_dir,
        "dataset_type": "vif",
        "name": "test_dataset",
        "n_channels": args.in_channel,
        "phase": "test"
    }

    test_set = D(test_opt)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    # --- Inference timing statistics ---
    total_inference_time = 0.0
    valid_counts = 0
    warmup_steps = 5  # Exclude the first 5 images from timing for warmup

    for i, test_data in enumerate(test_loader):
        imgname = test_data['A_path'][0]
        vis_path = test_data['B_path'][0]
        img_a = test_data['A'].to(device)
        img_b = test_data['B'].to(device)
        start = time.time()

        # Inference
        with torch.no_grad():
            _, _, h_old, w_old = img_a.size()

            # Calculate the new padded dimensions (multiples of REQUIRED_MULTIPLE)
            h_new = (h_old + REQUIRED_MULTIPLE - 1) // REQUIRED_MULTIPLE * REQUIRED_MULTIPLE
            w_new = (w_old + REQUIRED_MULTIPLE - 1) // REQUIRED_MULTIPLE * REQUIRED_MULTIPLE

            # Calculate the padding amounts (right and bottom)
            pad_h = h_new - h_old
            pad_w = w_new - w_old

            # Pad using F.pad (reflect mode is recommended)
            # Order: (left, right, top, bottom)
            img_a_padded = F.pad(img_a, (0, pad_w, 0, pad_h), mode='reflect')
            img_b_padded = F.pad(img_b, (0, pad_w, 0, pad_h), mode='reflect')

            # --- Model inference (using padded images) ---
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_infer = time.time()

            output_raw_padded = infer(img_a_padded, img_b_padded, model, args, window_size)[0]

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_infer = time.time()

            # Accumulate valid time (excluding warmup)
            if i >= warmup_steps:
                total_inference_time += (end_infer - start_infer)
                valid_counts += 1

            # --- Crop back to the original dimensions ---
            output = output_raw_padded[..., :h_old * args.scale, :w_old * args.scale]

            # --- Post-processing ---
            output = output.detach()[0].float().cpu()

        end = time.time()

        # =========================================================
        # Model output post-processing (Fuse Y channel + Restore Visible CbCr)
        # =========================================================
        fused_Y = util.tensor2uint(output)  # (H, W)
        vis_img = cv2.imread(vis_path, cv2.IMREAD_COLOR)  # BGR
        vis_ycbcr = cv2.cvtColor(vis_img, cv2.COLOR_BGR2YCrCb)
        Y_vis, Cr_vis, Cb_vis = cv2.split(vis_ycbcr)

        # Ensure size matching
        if fused_Y.shape[:2] != Y_vis.shape[:2]:
            fused_Y = cv2.resize(fused_Y, (Y_vis.shape[1], Y_vis.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Merge into new YCrCb and convert back to BGR (Note the order: Y, Cr, Cb)
        fused_ycbcr = cv2.merge([fused_Y, Cr_vis, Cb_vis])

        # Convert back to BGR, then to RGB for imsave
        fused_bgr = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)
        fused_rgb = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB)

        # Save fused image
        save_name = os.path.join(save_dir, os.path.basename(imgname))
        util.imsave(fused_rgb, save_name)

        print("[{}/{}]  Saving fused image to : {}, Processing time is {:.4f} s".format(
            i + 1, len(test_loader), save_name, end - start))

    # --- Output final statistics ---
    if valid_counts > 0:
        avg_time = total_inference_time / valid_counts
        fps = 1.0 / avg_time
        print("\n" + "=" * 50)
        print(f"[*] Final Performance on Dataset: {args.dataset}")
        print(f"[*] Average Inference Time: {avg_time:.4f} s")
        print(f"[*] Inference FPS: {fps:.2f}")
        print("=" * 50)


def define_model(args):
    model = net(in_chans=3,
                prompt_channels=8,
                window_size=8,
                img_range=1,
                embed_dim=60,
                mlp_ratio=2)

    param_key_g = 'params'
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)
    return model


def setup(args):
    save_dir = f'results/P2Fusion_{args.dataset}'
    window_size = 8
    return save_dir, window_size


def infer(img_a, img_b, model, args, window_size):
    if args.tile is None:
        # Direct whole-image inference
        outputs = model(img_a, img_b)
        if isinstance(outputs, dict):
            output = outputs['result']
        else:
            output = outputs
    else:
        # Tile-based inference (if tile size is specified)
        b, c, h, w = img_a.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "Tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_a)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch_a = img_a[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                in_patch_b = img_b[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch_dict = model(in_patch_a, in_patch_b)
                out_patch = out_patch_dict['result'] if isinstance(out_patch_dict, dict) else out_patch_dict
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


if __name__ == '__main__':
    main()