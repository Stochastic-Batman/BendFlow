import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from bendflow.models.common import TimeMLP, zero_init


class ResBlock(eqx.Module):
    norm1: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    time_proj: eqx.nn.Linear
    norm2: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    skip: eqx.nn.Conv2d | None

    def __init__(self, c_in: int, c_out: int, time_dim: int, groups: int, *, key: Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.norm1 = eqx.nn.GroupNorm(groups, c_in)
        self.conv1 = eqx.nn.Conv2d(c_in, c_out, 3, padding=1, key=k1)
        self.time_proj = eqx.nn.Linear(time_dim, c_out, key=k2)
        self.norm2 = eqx.nn.GroupNorm(groups, c_out)
        self.conv2 = eqx.nn.Conv2d(c_out, c_out, 3, padding=1, key=k3)
        self.skip = None if c_in == c_out else eqx.nn.Conv2d(c_in, c_out, 1, key=k4)

    def __call__(self, x: Array, temb: Array) -> Array:
        h = self.conv1(jax.nn.silu(self.norm1(x)))
        h = h + self.time_proj(temb)[:, None, None]
        h = self.conv2(jax.nn.silu(self.norm2(h)))
        return h + (x if self.skip is None else self.skip(x))


class AttnBlock(eqx.Module):
    norm: eqx.nn.GroupNorm
    attn: eqx.nn.MultiheadAttention

    def __init__(self, channels: int, groups: int, *, key: Array):
        self.norm = eqx.nn.GroupNorm(groups, channels)
        self.attn = eqx.nn.MultiheadAttention(1, channels, key=key)

    def __call__(self, x: Array) -> Array:
        c, h, w = x.shape
        y = self.norm(x).reshape(c, h * w).T
        return x + self.attn(y, y, y).T.reshape(c, h, w)


# Called as model(x, t) on one image (channels, height, width); batch with jax.vmap.
# The keyword arguments match the [unet] table of configs/networks.toml.
class UNet(eqx.Module):
    time_mlp: TimeMLP
    conv_in: eqx.nn.Conv2d
    down: list[list[ResBlock]]
    downsample: list[eqx.nn.Conv2d]
    mid1: ResBlock
    mid_attn: AttnBlock | None
    mid2: ResBlock
    up: list[list[ResBlock]]
    upsample: list[eqx.nn.Conv2d]
    norm_out: eqx.nn.GroupNorm
    conv_out: eqx.nn.Conv2d

    def __init__(self, image_shape: tuple[int, int, int], *, base_channels: int, channel_mults: tuple[int, ...], num_res_blocks: int, time_dim: int, groups: int, attention: bool, key: Array):
        keys = iter(jax.random.split(key, 4 * len(channel_mults) * (num_res_blocks + 1) + 16))
        in_channels, ch, levels = image_shape[0], base_channels, len(channel_mults)

        def block(c_in, c_out):
            return ResBlock(c_in, c_out, time_dim, groups, key=next(keys))

        self.time_mlp = TimeMLP(time_dim, key=next(keys))
        self.conv_in = eqx.nn.Conv2d(in_channels, ch, 3, padding=1, key=next(keys))

        cur, skips = ch, [ch]
        self.down, self.downsample = [], []
        for i, mult in enumerate(channel_mults):
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(block(cur, ch * mult))
                cur = ch * mult
                skips.append(cur)
            self.down.append(blocks)
            if i < levels - 1:
                self.downsample.append(eqx.nn.Conv2d(cur, cur, 3, stride=2, padding=1, key=next(keys)))
                skips.append(cur)

        self.mid1 = block(cur, cur)
        self.mid_attn = AttnBlock(cur, groups, key=next(keys)) if attention else None
        self.mid2 = block(cur, cur)

        self.up, self.upsample = [], []
        for i in reversed(range(levels)):
            blocks = []
            for _ in range(num_res_blocks + 1):
                blocks.append(block(cur + skips.pop(), ch * channel_mults[i]))
                cur = ch * channel_mults[i]
            self.up.append(blocks)
            if i > 0:
                self.upsample.append(eqx.nn.Conv2d(cur, cur, 3, padding=1, key=next(keys)))

        self.norm_out = eqx.nn.GroupNorm(groups, cur)
        self.conv_out = zero_init(eqx.nn.Conv2d(cur, in_channels, 3, padding=1, key=next(keys)))

    def __call__(self, x: Array, t: Array) -> Array:
        temb = self.time_mlp(t)
        h = self.conv_in(x)
        skips = [h]
        for i, blocks in enumerate(self.down):
            for blk in blocks:
                h = blk(h, temb)
                skips.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
                skips.append(h)

        h = self.mid1(h, temb)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid2(h, temb)

        for i, blocks in enumerate(self.up):
            for blk in blocks:
                h = blk(jnp.concatenate([h, skips.pop()]), temb)
            if i < len(self.upsample):
                h = jax.image.resize(h, (h.shape[0], *skips[-1].shape[1:]), "nearest")
                h = self.upsample[i](h)

        return self.conv_out(jax.nn.silu(self.norm_out(h)))
