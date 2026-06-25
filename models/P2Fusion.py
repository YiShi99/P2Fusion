import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ==============================================================================
# Feature Extraction Modules
# ==============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=60, embed_dim=60, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.proj(x)  # Downsample by patch_size (4x)
        x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, HW, C)
        x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=96, prompt_channels=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.prompt_channels = prompt_channels
        self.prompt_replicator = lambda x: x.repeat(1, self.prompt_channels, 1, 1)

        # Projection layer: from embed_dim to output channels
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.prompt_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            SELayer(embed_dim // 4),
            nn.Conv2d(embed_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, x_size):
        H, W = x_size
        B, N, C = x.shape
        assert N == H * W, f"N={N}, but expected H_feat*W_feat={H * W}"

        # Convert back to spatial feature map
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        feat_out = self.proj(x)
        prompt_single = self.prompt_head(x)
        prompt_out = self.prompt_replicator(prompt_single)

        return feat_out, prompt_out


class PatchUnEmbed_vis(nn.Module):
    def __init__(self, embed_dim=96, prompt_channels=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.prompt_channels = prompt_channels

        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.prompt_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            SELayer(embed_dim // 4),
            nn.Conv2d(embed_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Local quality map
        self.score_head = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, x_size):
        H, W = x_size
        B, N, C = x.shape
        assert N == H * W, f"N={N}, but expected H_feat*W_feat={H * W}"

        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        feat_out = self.proj(x)
        quality_map = self.prompt_head(x)

        # Learnable weighting followed by pooling
        weighted_map = torch.sigmoid(self.score_head(quality_map))
        quality_score = self.global_pool(weighted_map).flatten(1)

        return feat_out, quality_map, quality_score


# ==============================================================================
# Transformer & Attention Modules
# ==============================================================================

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(RSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN & Residual Connection
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ==============================================================================
# Helper Functions
# ==============================================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, f"H({H}), W({W}) must be divisible by window_size({window_size})"
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ==============================================================================
# Fusion Blocks
# ==============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Cross_WindowAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim_q % num_heads == 0, f"dim_q({dim_q}) must be divisible by num_heads({num_heads})"
        head_dim = dim_q // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_kv * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        B_, N, C_q = x.shape
        C_kv = y.shape[-1]
        q = self.q(x).reshape(B_, N, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C_kv // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C_q)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Cross_SwinTransformerBlock(nn.Module):
    def __init__(self, dim_ir, dim_vis, input_resolution, num_heads, window_size=8, shift_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_ir = dim_ir
        self.dim_vis = dim_vis
        self.input_resolution = input_resolution
        H, W = input_resolution
        assert H % window_size == 0 and W % window_size == 0, f"H({H}), W({W}) must be divisible by window_size({window_size})"

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_A = norm_layer(dim_ir)
        self.norm1_B = norm_layer(dim_vis)

        self.attn_A = Cross_WindowAttention(
            dim_q=dim_ir, dim_kv=dim_vis, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_B = Cross_WindowAttention(
            dim_q=dim_vis, dim_kv=dim_ir, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path_A = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_B = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_A = norm_layer(dim_ir)
        self.norm2_B = norm_layer(dim_vis)

        mlp_hidden_dim_ir = int(dim_ir * mlp_ratio)
        mlp_hidden_dim_vis = int(dim_vis * mlp_ratio)
        self.mlp_A = Mlp(in_features=dim_ir, hidden_features=mlp_hidden_dim_ir, act_layer=act_layer, drop=drop)
        self.mlp_B = Mlp(in_features=dim_vis, hidden_features=mlp_hidden_dim_vis, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, y, x_size):
        H, W = x_size
        B, L, C_x = x.shape
        C_y = y.shape[-1]

        assert L == H * W, f"x.shape[1]={L} != H*W={H * W}"

        shortcut_A = x
        shortcut_B = y

        x = self.norm1_A(x)
        y = self.norm1_B(y)
        x = x.view(B, H, W, C_x)
        y = y.view(B, H, W, C_y)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        x_windows = window_partition(shifted_x, self.window_size)
        y_windows = window_partition(shifted_y, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C_x)
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C_y)

        # Cross Attention
        if self.input_resolution == x_size:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.attn_mask)
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.attn_mask)
        else:
            attn_mask = self.calculate_mask(x_size).to(x.device)
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=attn_mask)
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=attn_mask)

        # Window Reverse
        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C_x)
        attn_windows_B = attn_windows_B.view(-1, self.window_size, self.window_size, C_y)

        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)
        shifted_y = window_reverse(attn_windows_B, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            y = shifted_y

        x = x.view(B, H * W, C_x)
        y = y.view(B, H * W, C_y)

        # MLP & Residual connections
        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))

        y = shortcut_B + self.drop_path_B(y)
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))

        return x, y


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        assert in_channels >= reduction, f"in_channels({in_channels}) must be >= reduction({reduction})"
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        channel_weight = self.channel_gate(x)
        x = x * channel_weight
        spatial_weight = self.spatial_gate(torch.cat([avg_pool, max_pool], dim=1))
        x = x * spatial_weight
        return x


class MoE(nn.Module):
    def __init__(self, in_channels_fuse, out_channels, prompt_channels=8):
        super().__init__()
        self.num_experts = 2
        self.out_channels = out_channels
        self.expert_enhance = nn.Sequential(
            nn.Conv2d(in_channels_fuse + 20, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, out_channels, kernel_size=1)
        )
        self.expert_modality = nn.Sequential(
            CBAM(in_channels_fuse),
            nn.Conv2d(in_channels_fuse, out_channels, kernel_size=1)
        )
        self.gate_fuse = nn.Conv2d(in_channels_fuse + 20, 48, kernel_size=1)
        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(48, self.num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Prompt convolutional layers to enhance saliency mask
        self.prompt_enhance = nn.Sequential(
            nn.Conv2d(prompt_channels, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            nn.GELU(),
        )
        self.vis_enhance = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            nn.GELU()
        )

    def forward(self, F_FUSE, F_modality_fused, P_vis, P_ir):
        P_vis_enhance = self.vis_enhance(P_vis)
        P_ir_enhance = self.prompt_enhance(P_ir)
        F_enhance = torch.cat([F_FUSE, P_ir_enhance, P_vis_enhance], dim=1)

        # Expert 1: Enhancement
        out_enhance = torch.sigmoid(self.expert_enhance(F_enhance))

        # Expert 2: Modality optimization
        out_modality = torch.sigmoid(self.expert_modality(F_FUSE))

        # Gating
        gate_features = self.gate_fuse(F_enhance)
        gate_weights = self.gate(gate_features).unsqueeze(2)
        gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Differentiated fusion
        F_updated = F_modality_fused

        # 1. Enhance salient regions (multiplicative)
        enhance_weight = gate_weights[:, 0] * out_enhance
        F_updated = F_updated * (1 + enhance_weight)

        # 2. Modality optimization (multiplicative)
        modality_weight = gate_weights[:, 1] * out_modality
        F_updated = F_updated * (1 + modality_weight)

        return F_updated, enhance_weight, modality_weight, gate_weights


class PromptFusionBlock(nn.Module):
    def __init__(self, in_channels_ir=68, in_channels_vis=60, out_channels=60, prompt_channels=8,
                 input_resolution=(160, 120)):
        super(PromptFusionBlock, self).__init__()
        self.out_channels = out_channels
        self.prompt_channels = prompt_channels
        self.in_channels_ir = in_channels_ir
        self.in_channels_vis = in_channels_vis

        self.cross_swin = Cross_SwinTransformerBlock(
            dim_ir=in_channels_vis,
            dim_vis=in_channels_vis,
            input_resolution=input_resolution,
            num_heads=4,  # Reduced heads to save computation
            window_size=8,
            shift_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            drop=0.,
            attn_drop=0.,
            drop_path=0.
        )
        self.special = Specific()
        self.conv = nn.Sequential(
            nn.Conv2d(240, 120, 3, 1, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
            nn.Conv2d(120, 60, 3, 1, 1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        # MoE expert models for IR and VIS
        self.moe_ir = MoE(
            in_channels_fuse=in_channels_vis,
            out_channels=out_channels,
            prompt_channels=prompt_channels
        )
        self.moe_vis = MoE(
            in_channels_fuse=in_channels_vis,
            out_channels=out_channels,
            prompt_channels=prompt_channels
        )

        # Linear projection to 60 dims
        self.F_ir_proj = nn.Linear(self.in_channels_ir, self.in_channels_vis)
        self.F_vis_proj = nn.Linear(self.in_channels_vis, self.in_channels_vis)

        # IR prior residual projection
        self.P_ir_res_proj = nn.Conv2d(
            in_channels=self.prompt_channels,
            out_channels=self.in_channels_ir,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # VIS prior residual projection
        self.P_vis_res_proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.in_channels_vis,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, F_ir, F_vis, P_ir, P_vis):
        B, C_ir, H, W = F_ir.shape
        C_vis = F_vis.shape[1]
        assert P_ir.shape[1] >= 1, f"P_ir must have at least 1 channel, got {P_ir.shape[1]}"
        assert P_vis.shape[1] == 1, f"P_vis is expected to have 1 channel, got {P_vis.shape[1]}"

        # Residual injection for IR prior
        P_ir_res = self.P_ir_res_proj(P_ir)
        F_ir = F_ir + P_ir_res

        # Residual injection for VIS prior
        P_vis_res = self.P_vis_res_proj(P_vis)
        F_vis = F_vis + P_vis_res

        # Flatten IR and VIS features
        F_ir_flat = F_ir.permute(0, 2, 3, 1).contiguous().view(B, H * W, C_ir)
        F_vis_flat = F_vis.permute(0, 2, 3, 1).contiguous().view(B, H * W, C_vis)

        F_ir_flat = self.F_ir_proj(F_ir_flat)
        F_vis_flat = self.F_vis_proj(F_vis_flat)

        # Cross Swin Transformer Block
        F_ir_fused_flat, F_vis_fused_flat = self.cross_swin(F_ir_flat, F_vis_flat, x_size=(H, W))

        F_ir_fused = F_ir_fused_flat.view(B, H, W, C_vis).permute(0, 3, 1, 2)
        F_vis_fused = F_vis_fused_flat.view(B, H, W, C_vis).permute(0, 3, 1, 2)

        F_comm = torch.cat([F_ir_fused, F_vis_fused], dim=1)

        P_ir_mask = P_ir[:, 0:1]

        # Dynamic weighted fusion for salient regions
        F_fused_weighted = P_ir_mask * (P_vis * F_vis_fused + (1 - P_vis) * F_ir_fused)
        # Average fusion for non-salient regions
        F_fused_unweighted = (1 - P_ir_mask) * (F_vis_fused + F_ir_fused) / 2
        F_fused = F_fused_weighted + F_fused_unweighted

        F_spec = self.special(F_ir_fused, F_vis_fused)

        # Concatenate and compress to 60 channels
        F_FUSE = self.conv(torch.cat([F_comm, F_spec, F_fused], dim=1))

        # MoE update
        F_fused_ir_final, enhance_weight_ir, modality_weight_ir, gate_weights_ir = \
            self.moe_ir(F_FUSE, F_ir_fused, P_vis, P_ir)
        F_fused_vis_final, enhance_weight_vis, modality_weight_vis, gate_weights_vis = \
            self.moe_vis(F_FUSE, F_vis_fused, P_vis, P_ir)

        return (F_fused_ir_final, F_fused_vis_final,
                F_ir_fused, F_vis_fused,
                enhance_weight_ir, modality_weight_ir, gate_weights_ir,
                enhance_weight_vis, modality_weight_vis, gate_weights_vis)


# ==============================================================================
# Main Network Architecture
# ==============================================================================

class p2fusion(nn.Module):
    def __init__(self, patch_size=4, num_blocks=4, prompt_channels=8, mlp_ratio=2., input_resolution=(160, 120),
                 in_chans=3, embed_dim=60, Ex_depths=[4], Ex_num_heads=[6], window_size=8,
                 qkv_bias=True, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, img_range=1.0, **kwargs):
        super(p2fusion, self).__init__()
        self.num_blocks = num_blocks
        self.prompt_channels = prompt_channels
        self.input_resolution = input_resolution
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.img_range = img_range
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if self.patch_norm else None)
        self.patch_size = patch_size

        self.patch_unembed = PatchUnEmbed(
            embed_dim=embed_dim, prompt_channels=8)
        self.patch_unembed_vis = PatchUnEmbed_vis(
            embed_dim=embed_dim)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_first = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 1, 1),
            self.lrelu,
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1),
            self.lrelu
        )

        dpr_Ex = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Ex_depths))]
        self.rstb = RSTB(dim=embed_dim,
                         input_resolution=self.input_resolution,
                         depth=Ex_depths[0],
                         num_heads=Ex_num_heads[0],
                         window_size=self.window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Ex,
                         norm_layer=norm_layer,
                         )
        self.norm_Ex_ir = norm_layer(self.embed_dim)

        self.fusion_blocks = nn.ModuleList()
        # Block 1: Input original features
        self.fusion_blocks.append(PromptFusionBlock(
            in_channels_ir=60,
            in_channels_vis=60,
            out_channels=60,
            prompt_channels=prompt_channels,
            input_resolution=input_resolution
        ))

        # Subsequent blocks: Input fused features from previous block
        for _ in range(num_blocks - 1):
            self.fusion_blocks.append(PromptFusionBlock(
                in_channels_ir=60,
                in_channels_vis=60,
                out_channels=60,
                prompt_channels=prompt_channels,
                input_resolution=input_resolution
            ))

        # Feature Reconstruction
        self.reconstruct_head = nn.Sequential(
            nn.Conv2d(120, 64, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            SELayer(16),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward_features_ir(self, ir_img):
        x = self.conv_first(ir_img)
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        H_p, W_p = x_size[0] // self.patch_size, x_size[1] // self.patch_size

        x = self.rstb(x, (H_p, W_p))
        x = self.norm_Ex_ir(x)

        ir_feat_out, ir_prompt_out = self.patch_unembed(x, (H_p, W_p))
        return ir_feat_out, ir_prompt_out

    def forward_features_vis(self, vis_img):
        x = self.conv_first(vis_img)
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        H_p, W_p = x_size[0] // self.patch_size, x_size[1] // self.patch_size

        x = self.rstb(x, (H_p, W_p))
        x = self.norm_Ex_ir(x)

        vis_feat_out, vis_prompt_map, vis_prompt_score = self.patch_unembed_vis(x, (H_p, W_p))
        return vis_feat_out, vis_prompt_map, vis_prompt_score

    def forward_features_Fusion(self, F_ir, F_vis, P_ir, P_vis_map):
        # IR and P_ir are no longer concatenated; residual injection is done in PromptFusionBlock
        F_IR = F_ir
        F_VIS = F_vis

        F_fused_ir_final = F_IR
        F_fused_vis_final = F_VIS

        for block in self.fusion_blocks:
            # Save current block input for residual connection
            F_IR_in = F_IR
            F_VIS_in = F_VIS

            (F_fused_ir_final, F_fused_vis_final,
             F_ir_fused, F_vis_fused,
             enhance_weight_ir, modality_weight_ir, gate_weights_ir,
             enhance_weight_vis, modality_weight_vis, gate_weights_vis) = \
                block(F_IR, F_VIS, P_ir, P_vis_map)

            # Next block input: Residual connections for IR/VIS
            F_IR = F_IR_in + F_fused_ir_final
            F_VIS = F_VIS_in + F_fused_vis_final

        return F_fused_ir_final, F_fused_vis_final, \
               F_ir_fused, F_vis_fused, \
               enhance_weight_ir, modality_weight_ir, gate_weights_ir, \
               enhance_weight_vis, modality_weight_vis, gate_weights_vis

    def forward_features_Re(self, F_fused_ir_final, F_fused_vis_final):
        F_fuse_total = torch.cat([F_fused_ir_final, F_fused_vis_final], dim=1)
        fused_img = self.reconstruct_head(F_fuse_total)
        return fused_img

    def forward(self, ir_img, vis_img, ablate_prompt=None):
        x = ir_img
        y = vis_img
        H, W = x.shape[2:]
        x = self.check_image_size(x, downsample_ratio=4)
        y = self.check_image_size(y, downsample_ratio=4)

        F_ir, P_ir = self.forward_features_ir(x)
        F_vis, P_vis_map, P_vis_score = self.forward_features_vis(y)

        # Required logic for ablation study
        if ablate_prompt == 'ir':
            P_ir = torch.zeros_like(P_ir)
        elif ablate_prompt == 'vis':
            P_vis_map = torch.zeros_like(P_vis_map)
        elif ablate_prompt == 'both':
            P_ir = torch.zeros_like(P_ir)
            P_vis_map = torch.zeros_like(P_vis_map)

        # P_ir and P_vis_map flow into MoE to calculate gates
        F_fused_ir_final, F_fused_vis_final, F_ir_fused, \
        F_vis_fused, en_w_ir, mo_w_ir, gate_ir, \
        en_w_vis, mo_w_vis, gate_vis = self.forward_features_Fusion(F_ir, F_vis, P_ir, P_vis_map)

        result = self.forward_features_Re(F_fused_ir_final, F_fused_vis_final)
        final_result = result[:, :, :H, :W]

        # Strictly return 12 elements
        return (final_result, P_ir, P_vis_map, F_fused_ir_final, F_fused_vis_final,
                F_ir_fused, en_w_ir, mo_w_ir, gate_ir,
                en_w_vis, mo_w_vis, gate_vis)

    def check_image_size(self, x, downsample_ratio=4):
        """
        Ensure spatial dimensions are divisible by window_size after downsampling
        """
        _, _, h, w = x.size()
        target_multiple = self.window_size * downsample_ratio
        mod_pad_h = (target_multiple - h % target_multiple) % target_multiple
        mod_pad_w = (target_multiple - w % target_multiple) % target_multiple
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Specific(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, 60, kernel_size=7, padding=3),
            nn.BatchNorm2d(60),
            nn.GELU(),
            nn.Conv2d(60, 60, kernel_size=3, padding=1),
            nn.BatchNorm2d(60),
            nn.GELU(),
        )

    def forward(self, F_vis, F_ir):
        avg_vis = torch.mean(F_vis, dim=1, keepdim=True)
        max_vis, _ = torch.max(F_vis, dim=1, keepdim=True)
        avg_ir = torch.mean(F_ir, dim=1, keepdim=True)
        max_ir, _ = torch.max(F_ir, dim=1, keepdim=True)
        spatial = self.spatial_gate(torch.cat([avg_vis, max_vis, avg_ir, max_ir], dim=1))

        return spatial