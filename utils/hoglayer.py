# Copied from MaskFeat.

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, base_size=16):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) 
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.patch_size = patch_size if type(patch_size) in [tuple, list] else (patch_size, patch_size)
        self.img_size = img_size if type(img_size) in [tuple, list] else (img_size, img_size)

        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    

# New version
def get_gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    def _gaussian_fn(kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n**2)
        return w
    gkern1d = _gaussian_fn(kernlen, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()

class HOGGenerator(nn.Module):
    """Generate HOG feature for images.

    This module is used in MaskFeat to generate HOG feature. The code is
    modified from file `slowfast/models/operators.py
    <https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/operators.py>`_.
    Here is the link of `HOG wikipedia
    <https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients>`_.

    Args:
        nbins (int): Number of bin. Defaults to 9.
        pool (float): Number of cell. Defaults to 8.
        gaussian_window (int): Size of gaussian kernel. Defaults to 16.
    """

    def __init__(self,
                 nbins: int = 9,
                 pool: int = 8,
                 gaussian_window: int = 16,
                 unfold_size=14,
                 channels : int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.nbins = nbins
        self.unfold = unfold_size
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(1, self.channels, 1, 1).contiguous()
        weight_y = weight_x.transpose(2, 3).contiguous()
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gaussian_kernel = self.get_gaussian_kernel(gaussian_window,
                                                       gaussian_window // 2)
            self.register_buffer('gaussian_kernel', gaussian_kernel)

    def get_gaussian_kernel(self, kernlen: int, std: int) -> torch.Tensor:
        """Returns a 2D Gaussian kernel array."""

        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n**2)
            return w

        kernel_1d = _gaussian_fn(kernlen, std)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d / kernel_2d.sum()

    def _reshape(self, hog_feat: torch.Tensor) -> torch.Tensor:
        """Reshape HOG Features for output."""
        hog_feat = hog_feat.flatten(1, 2)
        self.unfold_size = hog_feat.shape[-1] // self.unfold
        hog_feat = hog_feat.permute(0, 2, 3, 1)
        hog_feat = hog_feat.unfold(1, self.unfold_size,
                                   self.unfold_size).unfold(
                                       2, self.unfold_size, self.unfold_size)
        hog_feat = hog_feat.flatten(1, 2).flatten(2)
        return hog_feat

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate hog feature for each batch images.

        Args:
            x (torch.Tensor): Input images of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Hog features.
        """
        # input is RGB image with shape [B 3 H W]
        self.h, self.w = x.size(-2), x.size(-1)
        g = 1
        x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=g)
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=g)

        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)

        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w),
                          dtype=torch.float,
                          device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, 'h {} gw {}'.format(
                    h, self.gaussian_window)
                repeat_rate = h // self.gaussian_window
                temp_gaussian_kernel = self.gaussian_kernel.repeat(
                    [repeat_rate, repeat_rate])
            else:
                temp_gaussian_kernel = self.gaussian_kernel
            norm_rgb *= temp_gaussian_kernel

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        self.out = F.normalize(out, p=2, dim=2)
        return self._reshape(self.out)
    
    def generate_hog_image(self, hog_out: torch.Tensor = None) -> np.ndarray:
        """Generate HOG image according to HOG features."""
        hog_out = self.out if hog_out is None else hog_out

        zoom = 1
        assert hog_out.size(0) == 1 
        'Check the input batch size and the channcel number, only support'\
            '"batch_size = 1".'
        hog_image = np.zeros([zoom * self.h, zoom * self.w])
        cell_gradient = np.array(hog_out.mean(dim=1).squeeze().detach().cpu())
        cell_width = self.pool / 2
        max_mag = np.array(cell_gradient).max()
        angle_gap = 0 # 360 / self.nbins
        angles = 180 * (np.arange(self.nbins) - 4.5 + 0.5) / self.nbins 

        for x in range(cell_gradient.shape[1]):
            for y in range(cell_gradient.shape[2]):
                cell_grad = cell_gradient[:, x, y]
                cell_grad /= max_mag
                angle = 0 + 360/self.nbins/2
                for i, magnitude in enumerate(cell_grad):
                    angle_radian = math.radians(angles[i])
                    x1 = int(x * self.pool +
                             magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.pool +
                             magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.pool -
                             magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.pool -
                             magnitude * cell_width * math.sin(angle_radian))
                    magnitude = 0 if magnitude < 0 else magnitude
                    cv2.line(hog_image, (zoom*y1, zoom*x1), (zoom*y2, zoom*x2),
                             int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return hog_image

# class HOGLayerC(nn.Module):
#     def __init__(self, nbins=9, pool=8, gaussian_window=0, norm_pix_loss=True):
#         super(HOGLayerC, self).__init__()
#         self.nbins = nbins
#         self.pool = pool
#         self.pi = math.pi
#         weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
#         weight_x = weight_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
#         weight_y = weight_x.transpose(2, 3)
#         self.register_buffer("weight_x", weight_x)
#         self.register_buffer("weight_y", weight_y)
#         self.norm_pix_loss = norm_pix_loss

#         self.gaussian_window = gaussian_window
#         if gaussian_window:
#             gkern = get_gkern(gaussian_window, gaussian_window//2)
#             self.register_buffer("gkern", gkern)

#     @torch.no_grad()
#     def forward(self, x):
#         # input is RGB image with shape [B 3 H W]
#         x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")

#         g = x.shape[1]
#         print(g)
#         # gx_rgb = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=0, groups=g)
#         # gy_rgb = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=0, groups=g)

#         g_row = torch.zeros(x.shape)
#         g_row[0, :] = 0
#         g_row[-1, :] = 0
#         g_row[1:-1, :] = x[2:, :] - x[:-2, :]
#         g_col = torch.zeros(x.shape, dtype=x.dtype)
#         g_col[:, 0] = 0
#         g_col[:, -1] = 0
#         g_col[:, 1:-1] = x[:, 2:] - x[:, :-2]

#         gx_rgb = g_row
#         gy_rgb = g_col

#         norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
#         phase = torch.atan2(gy_rgb, gx_rgb)
#         phase = phase / self.pi * self.nbins  # [-9, 9]

#         b, c, h, w = norm_rgb.shape
#         out = torch.zeros((b, c, self.nbins, h, w), dtype=torch.float, device=x.device)
#         phase = phase.reshape(b, c, 1, h, w)
#         norm_rgb = norm_rgb.view(b, c, 1, h, w)
#         if self.gaussian_window:
#             if h != self.gaussian_window:
#                 assert h % self.gaussian_window == 0, "h {} gw {}".format(h, self.gaussian_window)
#                 repeat_rate = h // self.gaussian_window
#                 temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
#             else:
#                 temp_gkern = self.gkern
#             norm_rgb *= temp_gkern

#         out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)
#         out = out.unfold(3, self.pool, self.pool)
#         out = out.unfold(4, self.pool, self.pool)
#         out = out.sum(dim=[-1, -2])
#         if self.norm_pix_loss:
#             out = torch.nn.functional.normalize(out, p=2, dim=2)

#         out = out.flatten(1, 2)  # return n c h w
#         return out