import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bendflow.models.common import TimeMLP, zero_init


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, time_dim: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, c_out)
        self.norm2 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(temb)[..., None, None]
        return self.conv2(F.silu(self.norm2(h))) + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x).flatten(2).transpose(1, 2)
        q, k, v = (layer(y).unsqueeze(1) for layer in (self.query, self.key, self.value))
        y = F.scaled_dot_product_attention(q, k, v).squeeze(1)
        return x + self.output(y).transpose(1, 2).reshape_as(x)


class UNet(nn.Module):
    """Time-conditioned U-Net with the same resolution and skip layout as the reference."""

    def __init__(self, image_shape: tuple[int, int, int], *, base_channels: int, channel_mults: tuple[int, ...], num_res_blocks: int, time_dim: int, groups: int, attention: bool):
        super().__init__()
        in_channels, ch, levels = image_shape[0], base_channels, len(channel_mults)

        def block(c_in, c_out):
            return ResBlock(c_in, c_out, time_dim, groups)

        self.time_mlp = TimeMLP(time_dim)
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        cur, skips = ch, [ch]
        for i, mult in enumerate(channel_mults):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(block(cur, ch * mult))
                cur = ch * mult
                skips.append(cur)
            self.down.append(blocks)
            if i < levels - 1:
                self.downsample.append(nn.Conv2d(cur, cur, 3, stride=2, padding=1))
                skips.append(cur)

        self.mid1 = block(cur, cur)
        self.mid_attn = AttnBlock(cur, groups) if attention else nn.Identity()
        self.mid2 = block(cur, cur)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(levels)):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(block(cur + skips.pop(), ch * channel_mults[i]))
                cur = ch * channel_mults[i]
            self.up.append(blocks)
            if i > 0:
                self.upsample.append(nn.Conv2d(cur, cur, 3, padding=1))

        self.norm_out = nn.GroupNorm(groups, cur)
        self.conv_out = zero_init(nn.Conv2d(cur, in_channels, 3, padding=1))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        unbatched = x.ndim == 3
        if unbatched:
            x = x.unsqueeze(0)
        temb = self.time_mlp(t.expand(x.shape[0]))
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
        h = self.mid_attn(h)
        h = self.mid2(h, temb)
        for i, blocks in enumerate(self.up):
            for blk in blocks:
                h = blk(torch.cat([h, skips.pop()], dim=1), temb)
            if i < len(self.upsample):
                h = F.interpolate(h, size=skips[-1].shape[-2:], mode="nearest-exact")
                h = self.upsample[i](h)

        out = self.conv_out(F.silu(self.norm_out(h)))
        return out[0] if unbatched else out
