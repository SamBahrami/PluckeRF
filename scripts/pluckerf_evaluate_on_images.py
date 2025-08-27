# Copyright (c) 2025 Sam Bahrami
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import imageio.v2 as imageio
import pandas as pd
import torch
import lpips as lpips_lib
import tqdm
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F

# Make the evaluation csv files
gt_dir = "/path/to/evaluation_outputs_gt" # Update to your own GT path
eval_output_dir = "/path/to/evaluation_outputs" # PluckerF outputs path from inference

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim_fn(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class Metricator():
    # Metricator class from https://github.com/szymanowiczs/splatter-image/blob/78a6ad098e0cdc40c59c8ec98ca4fa439870fabd/eval.py#L22
    # So that we are evaluating on the same metrics as this comparison method
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    def compute_metrics(self, image, target):
        lpips = self.lpips_net( image.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((image - target) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, target).item()
        return psnr, lpips, ssim

def evaluate_image_directory(pred_image_dir: str, gt_image_dir: str, eval_name: str, device: torch.device):
    Pred_folder = Path(pred_image_dir)
    GT_folder = Path(gt_image_dir)
    pred_folders = sorted([f for f in Pred_folder.glob("*") if f.is_dir()])
    uids, psnrs, ssims, lpipss, indexes = [], [], [], [], []
    metricator = Metricator(device)

    for uid in tqdm.tqdm(pred_folders):
        uid_name = uid.name

        GT_rgb_folder = GT_folder / uid_name
        pred_rgb_folder = uid
        # Get and sort filenames (by the numeric part of the name)
        filenames = sorted([int(f.stem) for f in pred_rgb_folder.glob("*.png")])
        
        for view_idx, view_number in enumerate(filenames):
            view_idx_str_pred = str(view_number).zfill(6) + ".png"
            view_idx_str_gt = str(view_number).zfill(6) + ".png"

            pred_path = pred_rgb_folder / view_idx_str_pred
            gt_path = GT_rgb_folder / view_idx_str_gt

            if not pred_path.exists():
                continue

            # Load generated image
            rgb_gen = torch.Tensor(imageio.imread(pred_path))
            # Load GT image 
            rgb_gt = torch.Tensor(imageio.imread(gt_path))
            if rgb_gt.shape[2] == 4:
                rgb_gt = rgb_gt[:, :, :3]
            
            # Normalize to [0, 1]
            rgb_gen = torch.clamp(rgb_gen / 255.0, 0.0, 1.0)
            rgb_gt = torch.clamp(rgb_gt / 255.0, 0.0, 1.0)

            # Convert from HWC to CHW format
            rgb_gen = rgb_gen.permute(2, 0, 1)
            rgb_gt = rgb_gt.permute(2, 0, 1)

            # Compute metrics using the Metricator class
            psnr, lpips_value, ssim = metricator.compute_metrics(rgb_gen.to(device), rgb_gt.to(device))
            
            uids.append(uid_name)
            indexes.append(view_number)
            psnrs.append(psnr)
            lpipss.append(lpips_value)
            ssims.append(ssim)

    df = pd.DataFrame({
        "uid": uids,
        "index": indexes,
        "psnr": psnrs,
        "ssim": ssims,
        "lpips": lpipss,
    })
    df.to_csv(f"{eval_name}.csv", index=False)
    return df

extrapolated_views = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
                26, 27, 28, 29, 30, 31, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
                98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 161, 162, 163, 164, 165, 
                166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 
                182, 183, 184, 185, 186, 187, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 
                247, 248, 249] # views which are mutually over 90 degrees from both input views

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = evaluate_image_directory(eval_output_dir, gt_dir, "evaluation_results", device)
print(df[["psnr", "ssim", "lpips"]].mean())

# isolate the over 90 degree views i.e. where index is in the list extrapolated_views
# and then print the mean of the psnr, ssim, and lpips
extrapolated_views_df = df[df["index"].isin(extrapolated_views)]
print(extrapolated_views_df[["psnr", "ssim", "lpips"]].mean())

