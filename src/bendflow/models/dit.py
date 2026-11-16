import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from bendflow.models.common import TimeMLP, zero_init


def _layer_norm(dim: int) -> eqx.nn.LayerNorm:
    return eqx.nn.LayerNorm(dim, use_weight=False, use_bias=False)


# Cosines and sines of the 2D RoPE angles, each (grid_h * grid_w, head_dim / 2), tokens in row-major order.
# The first head_dim / 4 columns use the row index of a token, the rest its column index.
def rope_tables(grid_h: int, grid_w: int, head_dim: int, base: float = 10000.0) -> tuple[Array, Array]:
    quarter = head_dim // 4
    freqs = base ** (-jnp.arange(quarter) / quarter)
    rows, cols = jnp.meshgrid(jnp.arange(grid_h), jnp.arange(grid_w), indexing="ij")
    angles = jnp.concatenate([rows.reshape(-1, 1) * freqs, cols.reshape(-1, 1) * freqs], axis=1)
    return jnp.cos(angles), jnp.sin(angles)


# Rotates the entry pairs (2k, 2k + 1) of x (tokens, heads, head_dim) by the angle k of each token.
def rope_rotate(x: Array, cos: Array, sin: Array) -> Array:
    a, b, cos, sin = x[..., 0::2], x[..., 1::2], cos[:, None, :], sin[:, None, :]
    return jnp.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(x.shape)


class DiTBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp_in: eqx.nn.Linear
    mlp_out: eqx.nn.Linear
    ada: eqx.nn.Linear

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, *, key: Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        hidden = int(dim * mlp_ratio)
        self.norm1 = _layer_norm(dim)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads,
            dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=k1,
        )
        self.norm2 = _layer_norm(dim)
        self.mlp_in = eqx.nn.Linear(dim, hidden, key=k2)
        self.mlp_out = eqx.nn.Linear(hidden, dim, key=k3)
        self.ada = zero_init(eqx.nn.Linear(dim, 6 * dim, key=k4))

    # x: tokens (L, dim), c: conditioning vector (dim,), rope: output of rope_tables
    def __call__(self, x: Array, c: Array, rope: tuple[Array, Array]) -> Array:
        s1, b1, g1, s2, b2, g2 = jnp.split(self.ada(jax.nn.silu(c)), 6)
        h = jax.vmap(self.norm1)(x) * (1.0 + s1) + b1
        x = x + g1 * self.attn(h, h, h, process_heads=lambda q, k, v: (rope_rotate(q, *rope), rope_rotate(k, *rope), v))
        h = jax.vmap(self.norm2)(x) * (1.0 + s2) + b2
        return x + g2 * jax.vmap(lambda z: self.mlp_out(jax.nn.gelu(self.mlp_in(z))))(h)


# Time-conditioned DiT with adaLN-Zero and 2D RoPE, called as model(x, t) on one image (channels, height, width).
# The keyword arguments match the [dit] table of configs/networks.toml.
class DiT(eqx.Module):
    image_shape: tuple[int, int, int]
    patch_size: int
    head_dim: int
    patch_embed: eqx.nn.Linear
    time_mlp: TimeMLP
    blocks: list[DiTBlock]
    norm_out: eqx.nn.LayerNorm
    ada_out: eqx.nn.Linear
    linear_out: eqx.nn.Linear

    def __init__(self, image_shape: tuple[int, int, int], *, patch_size: int, hidden_dim: int, depth: int, num_heads: int, mlp_ratio: float, key: Array):
        c, h, w = image_shape
        if h % patch_size or w % patch_size:
            raise ValueError(f"patch_size {patch_size} must divide the image size {h}x{w}")
        if hidden_dim % num_heads or (hidden_dim // num_heads) % 4:
            raise ValueError(f"hidden_dim / num_heads must be a multiple of 4 for 2D RoPE, got {hidden_dim} / {num_heads}")
        k_embed, k_time, k_ada, k_out, *k_blocks = jax.random.split(key, 4 + depth)
        patch_dim = patch_size * patch_size * c

        self.image_shape = tuple(image_shape)
        self.patch_size = patch_size
        self.head_dim = hidden_dim // num_heads
        self.patch_embed = eqx.nn.Linear(patch_dim, hidden_dim, key=k_embed)
        self.time_mlp = TimeMLP(hidden_dim, key=k_time)
        self.blocks = [DiTBlock(hidden_dim, num_heads, mlp_ratio, key=k) for k in k_blocks]
        self.norm_out = _layer_norm(hidden_dim)
        self.ada_out = zero_init(eqx.nn.Linear(hidden_dim, 2 * hidden_dim, key=k_ada))
        self.linear_out = zero_init(eqx.nn.Linear(hidden_dim, patch_dim, key=k_out))

    # (C, H, W) -> (num_patches, p * p * C)
    def patchify(self, x: Array) -> Array:
        (c, h, w), p = self.image_shape, self.patch_size
        x = x.reshape(c, h // p, p, w // p, p).transpose(1, 3, 2, 4, 0)
        return x.reshape((h // p) * (w // p), p * p * c)

    # (num_patches, p * p * C) -> (C, H, W)
    def unpatchify(self, tokens: Array) -> Array:
        (c, h, w), p = self.image_shape, self.patch_size
        x = tokens.reshape(h // p, w // p, p, p, c).transpose(4, 0, 2, 1, 3)
        return x.reshape(c, h, w)

    def __call__(self, x: Array, t: Array) -> Array:
        (_, height, width), p = self.image_shape, self.patch_size
        cond = self.time_mlp(t)
        rope = rope_tables(height // p, width // p, self.head_dim)
        h = jax.vmap(self.patch_embed)(self.patchify(x))
        for blk in self.blocks:
            h = blk(h, cond, rope)
        shift, scale = jnp.split(self.ada_out(jax.nn.silu(cond)), 2)
        h = jax.vmap(self.norm_out)(h) * (1.0 + scale) + shift
        return self.unpatchify(jax.vmap(self.linear_out)(h))
