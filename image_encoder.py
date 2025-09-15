
# ./models/mmseg/models/sam
# --- START OF FILE image_encoder.py ---

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# (Dependencies: torch, nn, math, warnings, itertools, collections.abc)
# (Assuming common.py with LayerNorm2d, MLPBlock exists)

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from .common import LayerNorm2d, MLPBlock # Assuming common.py exists
import math
import warnings
from itertools import repeat
import numpy as np # Keep for potential future use

# --- Helper functions from standard libraries ---
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # (Truncated normal initialization function remains the same)
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std): warnings.warn("...", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std); u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1); tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.)); tensor.add_(mean); tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# --- Original ViT Components (PatchEmbed, Block, Attention, etc.) ---
# (Include definitions for PatchEmbed, Block, Attention, windowing, rel_pos helpers)
# ... (definitions from previous answers, ensure Attention calls add_decomposed_rel_pos correctly if use_rel_pos=True) ...
class PatchEmbed(nn.Module): # Main PatchEmbed
    def __init__(self, kernel_size: Tuple[int, int] = (16, 16), stride: Tuple[int, int] = (16, 16), padding: Tuple[int, int] = (0, 0), in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x); x = x.permute(0, 2, 3, 1); return x # B H W C

class PatchEmbed2(nn.Module): # For PromptGenerator paths
    """ Image to Patch Embedding specific for PromptGenerator """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size); patch_size = to_2tuple(patch_size);
        self.img_size = img_size; self.patch_size = patch_size;
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        B, C, H, W = x.shape
        # Allow different input size but project using patch_size kernel/stride
        # if H != self.img_size[0] or W != self.img_size[1]: pass # Warning removed for flexibility
        x = self.proj(x); return x # Output: (B, embed_dim, H_patch, W_patch)

class Block(nn.Module): # ViT Block
    def __init__( self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm, act_layer: Type[nn.Module] = nn.GELU, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, window_size: int = 0, input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, input_size=input_size if window_size == 0 else (window_size, window_size))
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x; x_norm = self.norm1(x) # Apply norm before attention
        # Window partition
        if self.window_size > 0: H, W = x_norm.shape[1], x_norm.shape[2]; x_attn_in, pad_hw = window_partition(x_norm, self.window_size)
        else: x_attn_in = x_norm # Global attention uses normalized input directly
        # Attention
        x_attn_out = self.attn(x_attn_in)
        # Reverse window partition
        if self.window_size > 0: x_attn_out = window_unpartition(x_attn_out, self.window_size, pad_hw, (H, W))
        # Residual + MLP
        x = shortcut + x_attn_out; x = x + self.mlp(self.norm2(x)); return x

class Attention(nn.Module): # Multi-head Attention
    def __init__( self, dim: int, num_heads: int = 8, qkv_bias: bool = True, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(); self.num_heads = num_heads; head_dim = dim // num_heads; self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias); self.proj = nn.Linear(dim, dim); self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided for rel pos."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            if rel_pos_zero_init: nn.init.trunc_normal_(self.rel_pos_h, std=.02); nn.init.trunc_normal_(self.rel_pos_w, std=.02)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_dim, H, W, C_dim = x.shape # Input shape B H W C or (B*num_win) H W C
        qkv = self.qkv(x).reshape(B_dim, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B_dim * self.num_heads, H * W, -1).unbind(0); attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos: attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W), self.num_heads) # Pass num_heads
        attn = attn.softmax(dim=-1); x_out = (attn @ v).view(B_dim, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B_dim, H, W, -1); x_out = self.proj(x_out); return x_out

# --- Windowing Functions ---
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape; pad_h = (window_size - H % window_size) % window_size; pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0: x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h)) # Pad C, W, H
    Hp, Wp = H + pad_h, W + pad_w; x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C); return windows, (Hp, Wp)

def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
    Hp, Wp = pad_hw; H, W = hw; B = windows.shape[0] // (Hp * Wp // window_size // window_size); x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1);
    if Hp > H or Wp > W: x = x[:, :H, :W, :].contiguous(); return x

# --- Relative Position Functions (using latest corrected versions) ---
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """ Gets relative positional embeddings, interpolates if necessary. """
    max_rel_dist = int(2 * max(q_size, k_size) - 1); stored_max_dist = rel_pos.shape[0]; head_dim = rel_pos.shape[1]; device = rel_pos.device
    if stored_max_dist != max_rel_dist:
        rel_pos_transposed = rel_pos.transpose(0, 1).unsqueeze(0); rel_pos_resized = F.interpolate(rel_pos_transposed, size=max_rel_dist, mode="linear", align_corners=False)
        rel_pos_resized = rel_pos_resized.squeeze(0).transpose(0, 1)
    else: rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=device); k_coords = torch.arange(k_size, device=device); relative_coords = q_coords[:, None] - k_coords[None, :]; relative_indices = relative_coords + (k_size - 1)
    relative_indices = torch.clamp(relative_indices, 0, max_rel_dist - 1); final_rel_pos = rel_pos_resized[relative_indices.long()]

    expected_shape = (q_size, k_size, head_dim);
    # Re-enable strict check
    if final_rel_pos.shape != expected_shape: raise RuntimeError(f"Shape mismatch in get_rel_pos: Expected {expected_shape}, got {final_rel_pos.shape}. Stored={rel_pos.shape}, Resized={rel_pos_resized.shape}, Indices={relative_indices.shape}")
    return final_rel_pos


def add_decomposed_rel_pos( attn: torch.Tensor, q: torch.Tensor, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor, q_size: Tuple[int, int], k_size: Tuple[int, int], num_heads: int) -> torch.Tensor:
    """ Adds decomposed relative positional embeddings to attention map. (Corrected einsum subscripts) """
    q_h, q_w = q_size; k_h, k_w = k_size; q_HW = q_h * q_w; k_HW = k_h * k_w
    B_times_nH, HW, C_head = q.shape
    if HW != q_HW: raise ValueError(f"q HW dimension {HW} does not match q_size {q_HW}")

    N = B_times_nH // num_heads
    if B_times_nH % num_heads != 0: raise ValueError(f"Batch*num_heads dimension {B_times_nH} is not divisible by num_heads {num_heads}")

    # Reshape q to expose the head dimension: (N, nH, q_h, q_w, C_head)
    q_reshaped = q.view(N, num_heads, q_h, q_w, C_head)

    # Get relative positional embeddings biases using refined get_rel_pos
    Rh = get_rel_pos(q_h, k_h, rel_pos_h) # Shape (q_h, k_h, C_head)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w) # Shape (q_w, k_w, C_head)

    # --- Corrected Einsum Equations ---
    # Calculate relative height bias: (N, nH, q_h, q_w, k_h)
    rel_h_bias = torch.einsum('nihwd,hkd->nihwk', q_reshaped, Rh) # n=B, i=Head, h=qH, w=qW, d=Dim, k=kH

    # Calculate relative width bias: (N, nH, q_h, q_w, k_w)
    rel_w_bias = torch.einsum('nihwd,wkd->nihwk', q_reshaped, Rw) # n=B, i=Head, h=qH, w=qW, d=Dim, k=kW
    # --- End Correction ---


    # Reshape attention map to add biases
    attn = attn.view(N, num_heads, q_h, q_w, k_h, k_w)

    # Add biases
    attn = attn + rel_h_bias.unsqueeze(-1)
    attn = attn + rel_w_bias.unsqueeze(-2)

    # Reshape attention map back
    attn = attn.view(B_times_nH, q_HW, k_HW)

    return attn

# --- Modified PromptGenerator ---
class PromptGenerator(nn.Module):
    def __init__(self, scale_factor, prompt_type, embed_dim, tuning_stage, depth, input_type,
                 freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size, patch_size):
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor if scale_factor > 0 else 1
        self.embed_dim = embed_dim # Main ViT embed_dim
        self.depth = depth
        self.freq_nums = freq_nums # For FFT path

        # Calculate the feature dimension for internal MLPs and prompt_generator output
        # This is the dim used *before* shared_mlp projection back to embed_dim
        internal_embed_dim = max(1, self.embed_dim // self.scale_factor)

        # MLP for projecting back to main ViT dimension
        self.shared_mlp = nn.Linear(internal_embed_dim, self.embed_dim)
        # MLP for projecting main ViT embedding down for combination
        self.embedding_generator = nn.Linear(self.embed_dim, internal_embed_dim)

        # Lightweight MLPs operating at internal_embed_dim
        for i in range(self.depth):
            lightweight_mlp = nn.Sequential(
                nn.Linear(internal_embed_dim, internal_embed_dim),
                nn.GELU()
            )
            setattr(self, f'lightweight_mlp_{i}', lightweight_mlp)

        # --- PatchEmbed layers for different handcrafted paths ---
        # Input: original image x (3 channels) + optional enhanced edge (1 channel) = 4 channels
        self.prompt_gen_struct_4ch = PatchEmbed2(img_size=img_size, patch_size=patch_size, in_chans=4, embed_dim=internal_embed_dim)
        # Input: original image x (3 channels) - fallback if filtered_image not provided
        self.prompt_gen_struct_3ch = PatchEmbed2(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=internal_embed_dim)
        # Input: FFT result (assuming 3 channels after abs or real part)
        self.prompt_gen_fft = PatchEmbed2(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=internal_embed_dim)

        # --- Kernels for structure-aware path ---
        # Laplacian Kernel
        lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.lap_kernel = lap_k.unsqueeze(0).unsqueeze(0) # Shape (1, 1, 3, 3) - register as buffer
        self.register_buffer('laplacian_kernel', self.lap_kernel)

        # Gaussian Kernel (example)
        gaussian_k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
        self.gauss_kernel = gaussian_k.unsqueeze(0).unsqueeze(0) # Shape (1, 1, 3, 3)
        self.register_buffer('gaussian_blur_kernel', self.gauss_kernel)

        # Learnable weight for combining structure and FFT prompts
        self.fusion_alpha = nn.Parameter(torch.tensor(0.3)) # Initialize

        self.apply(self._init_weights) # Apply weight initialization

    def _init_weights(self, m):
        # (Weight initialization function remains the same)
        if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=.02);
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels; fan_out //= m.groups if m.groups > 0 else 1; m.weight.data.normal_(0, math.sqrt(2.0 / max(1, fan_out)));
            if m.bias is not None: m.bias.data.zero_()

    def compute_sharp_blur_map(self, filtered_image: torch.Tensor):
        """ Computes map indicating potentially blurred regions (value 1 = blur). """
        if filtered_image.shape[1] > 1: gray_filtered = filtered_image.mean(dim=1, keepdim=True)
        else: gray_filtered = filtered_image
        # Use registered buffer for kernel
        globally_blurred = F.conv2d(gray_filtered, weight=self.gaussian_blur_kernel.repeat(1, gray_filtered.shape[1], 1, 1), padding=1, groups=gray_filtered.shape[1])
        difference = (gray_filtered - globally_blurred).abs()
        max_diff = torch.max(difference.view(difference.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1) + 1e-6
        sharpness_map = difference / max_diff
        blur_map = 1.0 - sharpness_map # High value (near 1) means blurred region in filtered_image
        return blur_map

    def init_embeddings(self, x):
        """ x is the output of main PatchEmbed: (N, H, W, C=embed_dim) """
        N, H, W, C = x.shape; L = H * W; x_reshaped = x.view(N, L, C)
        # embedding_generator outputs (N, L, internal_embed_dim)
        return self.embedding_generator(x_reshaped)

    def fft(self, x, rate):
        """ Applies high-pass FFT filtering. """
        # Original implementation: cuts out a low-freq square/rect. Let's keep that.
        mask = torch.zeros_like(x) # Mask=0 means REMOVE frequency
        _, _, h, w = x.shape
        line_h = int(h * rate * 0.5) # Half-width/height of low-freq area to REMOVE
        line_w = int(w * rate * 0.5)
        # Create mask where low frequencies are 1 (to be multiplied by 0 later)
        low_freq_mask_area = torch.zeros_like(mask)
        low_freq_mask_area[:, :, h//2-line_h:h//2+line_h, w//2-line_w:w//2+line_w] = 1
        # High-pass mask: Keep everything EXCEPT the low-freq area
        high_pass_mask = 1.0 - low_freq_mask_area

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        fft_filtered = fft * high_pass_mask # Apply high-pass mask
        ifft_result = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        inv = torch.fft.ifft2(ifft_result, norm="ortho").real

        # Original code took abs value. Keep it for consistency with original PromptGenerator.
        inv = torch.abs(inv)
        return inv # Shape (N, C_in, H, W)


    def init_handcrafted(self, x: torch.Tensor, filtered_image: Optional[torch.Tensor] = None):
        """ Generates combined handcrafted features using structure and FFT paths. """
        B, C_in, H, W = x.shape
        device = x.device
        internal_embed_dim = self.embedding_generator.out_features # Get dim from layer

        # === [1] Structure-Aware Path ===
        if filtered_image is not None:
            filtered_image = filtered_image.to(device)
            if filtered_image.shape[-2:] != (H, W):
                filtered_image = F.interpolate(filtered_image, size=(H, W), mode='bilinear', align_corners=False)
            blur_map = self.compute_sharp_blur_map(filtered_image) # (N, 1, H, W)

            if C_in > 1: gray_x = x.mean(dim=1, keepdim=True)
            else: gray_x = x
            # Use registered buffer for kernel
            edge_map = F.conv2d(gray_x, weight=self.laplacian_kernel.repeat(1, gray_x.shape[1], 1, 1), padding=1, groups=gray_x.shape[1]).abs()
            edge_map = edge_map / (torch.max(edge_map.view(B,-1), dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1) + 1e-6) # Normalize
            sharp_region_mask = 1.0 - blur_map
            enhanced_edge = edge_map * sharp_region_mask
            x_structure_input = torch.cat([x, enhanced_edge], dim=1) # (N, 4, H, W)
            prompt_structure_raw = self.prompt_gen_struct_4ch(x_structure_input)
        else:
            # Fallback: Use original image with 3ch prompt generator
            x_structure_input = x
            prompt_structure_raw = self.prompt_gen_struct_3ch(x_structure_input)

        # Reshape structure prompt -> (N, L, C_internal)
        N_p, C_p, H_p, W_p = prompt_structure_raw.shape; L_p = H_p * W_p
        prompt_structure = prompt_structure_raw.permute(0, 2, 3, 1).reshape(N_p, L_p, C_p)

        # === [2] FFT Path ===
        x_fft = self.fft(x, self.freq_nums) # Use original x; returns abs value (N, C_in, H, W)
        prompt_fft_raw = self.prompt_gen_fft(x_fft) # Input has C_in channels (e.g., 3)
        prompt_fft = prompt_fft_raw.permute(0, 2, 3, 1).reshape(N_p, L_p, C_p) # Reshape -> (N, L, C_internal)

        # === [3] Fusion ===
        # Combine using learnable alpha (ensure shapes match)
        if prompt_structure.shape != prompt_fft.shape:
            # This shouldn't happen if PatchEmbed2 instances are correct
            raise RuntimeError(f"Shape mismatch between structure {prompt_structure.shape} and FFT prompts {prompt_fft.shape}")
        combined_prompt = prompt_structure + self.fusion_alpha * prompt_fft

        return combined_prompt # Shape: (N, L, C_internal)


    def get_prompt(self, combined_handcrafted_feature, embedding_feature):
        """ Generates final prompts projected to main embed_dim. """
        N, L, C_internal = combined_handcrafted_feature.shape
        if embedding_feature.shape != (N, L, C_internal):
             raise ValueError(f"Shape mismatch: combined_handcrafted {combined_handcrafted_feature.shape}, embedding {embedding_feature.shape}")

        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(self, f'lightweight_mlp_{i}')
            # Combine features at internal dim
            combined_feature = combined_handcrafted_feature + embedding_feature
            prompt_internal = lightweight_mlp(combined_feature)
            # Project back to main embed_dim
            prompt_final = self.shared_mlp(prompt_internal)
            prompts.append(prompt_final) # List of (N, L, embed_dim)
        return prompts

    # Remove the old forward method from PromptGenerator if it exists
    # def forward(self, x): # This method seems out of place now
    #     ...


# --- Main ImageEncoderViT Class (using modified PromptGenerator) ---
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 14, # Example
        global_attn_indexes: Tuple[int, ...] = (),
        # --- PromptGenerator parameters ---
        pg_scale_factor: int = 32, # Example
        pg_freq_nums: float = 0.25, # Example for high-pass FFT cutoff (fraction to cut from center)
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # Main Patch Embedding
        self.patch_embed = PatchEmbed(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), in_chans=in_chans, embed_dim=embed_dim)
        num_patches_h = img_size // patch_size
        num_patches_w = img_size // patch_size

        # Absolute Positional Embedding
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches_h, num_patches_w, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        # ViT Blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, window_size=window_size if i not in global_attn_indexes else 0, input_size=(num_patches_h, num_patches_w))
            self.blocks.append(block)

        # Neck
        self.neck = nn.Sequential(nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False), LayerNorm2d(out_chans), nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False), LayerNorm2d(out_chans))

        # Prompt Generator Initialization
        self.prompt_generator = PromptGenerator(scale_factor=pg_scale_factor, prompt_type='highpass', embed_dim=embed_dim, tuning_stage=1234, depth=depth, input_type='fft', freq_nums=pg_freq_nums, handcrafted_tune=True, embedding_tune=True, adaptor='adaptor', img_size=img_size, patch_size=patch_size)

        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages)) # Keep if used elsewhere

    # --- forward method uses filtered_image ---
    def forward(self, x: torch.Tensor, filtered_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Input x: (N, C_in, H_img, W_img) - Original image
        Input filtered_image: Optional (N, C_in, H_img, W_img) - FD-based filtered image
        """
        inp = x # Keep original input reference

        # --- 1. Main Patch Embedding ---
        x_patch_embedded = self.patch_embed(inp) # (N, H_patch, W_patch, C=embed_dim)
        B, H_patch, W_patch, C_embed = x_patch_embedded.shape
        L_patch = H_patch * W_patch

        # --- 2. Initialize Features for PromptGenerator ---
        embedding_feature = self.prompt_generator.init_embeddings(x_patch_embedded) # (N, L, C_internal)
        combined_handcrafted_feature = self.prompt_generator.init_handcrafted(inp, filtered_image) # (N, L, C_internal)

        # --- 3. Generate Prompts ---
        prompts = self.prompt_generator.get_prompt(combined_handcrafted_feature, embedding_feature) # List[ (N, L, C_embed) ]

        # --- 4. Apply Positional Embedding ---
        x_processed = x_patch_embedded # Start with patch features
        if self.pos_embed is not None:
             x_processed = x_processed + self.pos_embed

        # --- 5. Process through ViT Blocks with Prompts ---
        for i, blk in enumerate(self.blocks):
            prompt_reshaped = prompts[i].view(B, H_patch, W_patch, C_embed)
            x_processed = prompt_reshaped + x_processed # Add prompt before block
            x_processed = blk(x_processed)

        # --- 6. Neck ---
        neck_output = self.neck(x_processed.permute(0, 3, 1, 2))

        return neck_output # Final output

# --- END OF FILE ---
