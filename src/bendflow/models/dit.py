import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bendflow.models.common import TimeMLP, zero_init


def rope_tables(grid_h: int, grid_w: int, head_dim: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    quarter = head_dim // 4
    freqs = base ** (-torch.arange(quarter, dtype=torch.get_default_dtype()) / quarter)
    rows, cols = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    angles = torch.cat([rows.reshape(-1, 1) * freqs, cols.reshape(-1, 1) * freqs], dim=1)
    return angles.cos(), angles.sin()


def rope_rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    a, b = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[:, None], sin[:, None]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).reshape_as(x)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attn_out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp_in = nn.Linear(dim, hidden)
        self.mlp_out = nn.Linear(hidden, dim)
        self.ada = zero_init(nn.Linear(dim, 6 * dim))

    def forward(self, x: Tensor, c: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(c)).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1) + b1
        q, k, v = (layer(h).unflatten(-1, (self.num_heads, -1)) for layer in (self.query, self.key, self.value))
        q, k = rope_rotate(q, *rope), rope_rotate(k, *rope)
        attention = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + g1 * self.attn_out(attention.transpose(1, 2).reshape_as(h))
        h = self.norm2(x) * (1 + s2) + b2
        return x + g2 * self.mlp_out(F.gelu(self.mlp_in(h), approximate="tanh"))


class DiT(nn.Module):
    """Time-conditioned pixel-space DiT with adaLN-Zero and 2D RoPE."""

    def __init__(self, image_shape: tuple[int, int, int], *, patch_size: int, hidden_dim: int, depth: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        c, h, w = image_shape
        if h % patch_size or w % patch_size:
            raise ValueError(f"patch_size {patch_size} must divide the image size {h}x{w}")
        if hidden_dim % num_heads or (hidden_dim // num_heads) % 4:
            raise ValueError(f"hidden_dim / num_heads must be a multiple of 4 for 2D RoPE, got {hidden_dim} / {num_heads}")
        self.image_shape = tuple(image_shape)
        self.patch_size = patch_size
        self.head_dim = hidden_dim // num_heads
        patch_dim = patch_size * patch_size * c
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        self.time_mlp = TimeMLP(hidden_dim)
        self.blocks = nn.ModuleList(DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth))
        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ada_out = zero_init(nn.Linear(hidden_dim, 2 * hidden_dim))
        self.linear_out = zero_init(nn.Linear(hidden_dim, patch_dim))
        cos, sin = rope_tables(h // patch_size, w // patch_size, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def patchify(self, x: Tensor) -> Tensor:
        (c, h, w), p = self.image_shape, self.patch_size
        return x.reshape(*x.shape[:-3], c, h // p, p, w // p, p).movedim(-5, -1).transpose(-4, -3).flatten(-3).flatten(-3, -2)

    def unpatchify(self, tokens: Tensor) -> Tensor:
        (c, h, w), p = self.image_shape, self.patch_size
        batch_dims = tokens.shape[:-2]
        b = len(batch_dims)
        x = tokens.reshape(*batch_dims, h // p, w // p, p, p, c)
        x = x.permute(*range(b), b + 4, b, b + 2, b + 1, b + 3)
        return x.reshape(*batch_dims, c, h, w)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        unbatched = x.ndim == 3
        if unbatched:
            x = x.unsqueeze(0)
        cond = self.time_mlp(t.expand(x.shape[0]))
        h = self.patch_embed(self.patchify(x))
        for block in self.blocks:
            h = block(h, cond, (self.rope_cos, self.rope_sin))
        shift, scale = self.ada_out(F.silu(cond)).unsqueeze(1).chunk(2, dim=-1)
        out = self.unpatchify(self.linear_out(self.norm_out(h) * (1 + scale) + shift))
        return out[0] if unbatched else out
