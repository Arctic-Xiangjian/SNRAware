from einops import rearrange
import torch
from pathlib import Path
import sys

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri import fft2c, ifft2c, complex_abs
from fastmri_data.fastmri_data import SliceDataset
from fastmri.data.transforms import complex_center_crop

import hashlib

def _seed_from_name(name: str) -> int:
    # 用 sha256 的前 8 字节作为 64-bit，再裁成 RandomState 兼容的 31-bit 正整数
    h = hashlib.sha256(name.encode('utf-8')).digest()
    return int.from_bytes(h[:8], 'big') & 0x7fffffff

def get_mask(img, size, batch_size, type='uniform1d', acc_factor=4, center_fraction=0.08, fix=False, name=None):
    mux_in = size ** 2
    if type.endswith('2d'):
        Nsamp = mux_in // acc_factor
    elif type.endswith('1d'):
        Nsamp = size // acc_factor
    else:
        raise NotImplementedError(f'Mask type {type} is currently not supported.')

    if type != 'uniform1d':
        raise NotImplementedError(f'Mask type {type} is currently not supported.')

    mask = torch.zeros_like(img)

    Nsamp_tgt = int(round(size / acc_factor))         # 目标总数（按比例取整）
    Nsamp_center = int(round(size * center_fraction)) # 中心 ACS 列数
    if Nsamp_center > Nsamp_tgt:
        raise ValueError("center_fraction 超过 1/acc_factor，无法达到目标采样率。")

    # outside 区域的伯努利概率，使期望总数 = Nsamp_tgt
    denom = size - Nsamp_center
    prob = (Nsamp_tgt - Nsamp_center) / denom if denom > 0 else 0.0

    c_from = size // 2 - Nsamp_center // 2
    c_to = c_from + Nsamp_center

    # 决定使用哪个随机源：
    # - name 为 None：保持原始行为，使用全局 np.random（不可复现）
    # - name 为 str：用 name 派生的种子构造独立 RandomState（可复现）
    rng = None
    if isinstance(name, str):
        rng = np.random.RandomState(_seed_from_name(name))

    if fix or isinstance(name, str):
        # 生成一张固定的 sel（若给了 name，则忽略 fix 的影响）
        sel = (rng.rand(size) if rng is not None else np.random.rand(size)) < prob
        if Nsamp_center > 0:
            sel[c_from:c_to] = True
        idx = np.nonzero(sel)[0]
        # 复用到整个 batch
        mask[:, :, :, idx] = 1
    else:
        # 原行为：batch 内每个样本独立随机
        for i in range(batch_size):
            sel = (np.random.rand(size) < prob)
            if Nsamp_center > 0:
                sel[c_from:c_to] = True
            idx = np.nonzero(sel)[0]
            mask[i, :, :, idx] = 1

    return mask


from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset



class MRIDataset(Dataset):
    def __init__(self,
                 given_dataset: Dataset = None,
                 norm_mean: Optional[Union[float, Sequence[float]]] = None,
                 norm_std: Optional[Union[float, Sequence[float]]] = None,
                 acc_factor: int = 8):
        super().__init__()
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.acc_factor = acc_factor

        self.given_dataset = given_dataset


    def __len__(self):
        return len(self.given_dataset)

    def __getitem__(self, idx: int):
        sample_from_given = self.given_dataset[idx]

        orig_kspace_slice = np.zeros((1,sample_from_given[0].shape[0],sample_from_given[0].shape[1],2))
        # print(orig_kspace_slice.shape, sample_from_given[0].shape)
        orig_kspace_slice[0,:,:,0] = sample_from_given[0].real
        orig_kspace_slice[0,:,:,1] = sample_from_given[0].imag
        orig_kspace_slice = torch.tensor(orig_kspace_slice, dtype=torch.float32)

        case_name = sample_from_given[4]

        attrs = sample_from_given[3] if len(sample_from_given) > 3 else {}
        latent_feature = attrs.get("latent_feature") if isinstance(attrs, dict) else None
        if latent_feature is None:
            raise ValueError(
                "Missing latent_feature in SliceDataset sample. "
                "Please construct SliceDataset with latent_feature_root set."
            )
        latent_feature = torch.as_tensor(latent_feature).float()

        target = sample_from_given[2]
        target = torch.from_numpy(target).float()

        del sample_from_given

        # this three line to crop the original image to 320x320 and the fft back to kspace, this can keep all kspace information
        orig_recon_slice = ifft2c(orig_kspace_slice)
        orig_recon_slice = complex_center_crop(orig_recon_slice, (320, 320))
        orig_kspace_slice = fft2c(orig_recon_slice)
        center_fraction = 0.08 if self.acc_factor == 4 else 0.04

        mask_use_for = get_mask(
            img=orig_kspace_slice.permute(0,3,1,2),
            size=orig_kspace_slice.shape[2],
            batch_size=1,
            type='uniform1d',
            acc_factor=self.acc_factor,
            center_fraction=center_fraction,
            fix=False
        )

        # currently is (1,2,320,320) we need to change to (1,320,320,2)

        mask_use_for = mask_use_for.permute(0, 2, 3, 1)

        masked_kspace_slice = orig_kspace_slice * mask_use_for

        # permute mask_use_for_back to (1,2,320,320)
        mask_use_for = mask_use_for.squeeze()

        under_recon_slice = ifft2c(masked_kspace_slice)

        # print(under_recon_slice.shape, orig_recon_slice.shape)
        under_recon_slice = complex_center_crop(under_recon_slice, (320, 320))
        # mask_cropped = complex_center_crop(mask_use_for, (320, 320))
        
        masked_kspace_slice = masked_kspace_slice.squeeze(0)
        under_recon_slice_abs = complex_abs(under_recon_slice).squeeze().unsqueeze(0).numpy()
        target = target.squeeze().unsqueeze(0).numpy()

        lq_running_mean = 0
        lq_running_std = np.mean(under_recon_slice_abs) # this is not a bug, we use the mean of the under_recon_slice_abs as the std for normalization, this can keep the same scale for both under_recon_slice and target


        under_recon_slice = under_recon_slice / torch.tensor(lq_running_std).float()
        target = target / torch.tensor(lq_running_std).float()
        masked_kspace_slice = masked_kspace_slice / torch.tensor(lq_running_std).float()

        under_recon_slice = under_recon_slice.squeeze()
        under_recon_slice = rearrange(under_recon_slice, 'h w c -> c h w')

        self.norm_mean = 0
        self.norm_std = lq_running_std

        sample = {
            "name": case_name,
            "masked_kspace": masked_kspace_slice,
            "mask": mask_use_for,
            "mean": self.norm_mean,
            "std": self.norm_std,
            "latent_feature": latent_feature,
        }
        
        return under_recon_slice, target, sample
    
