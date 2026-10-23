from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    dim: int = 96
    layers: int = 3
    heads: int = 4
    wavelet: bool = True
    stft: bool = True
    polarization: bool = False
    gabor: bool = False
    scales: int = 12
    sample_rate: int = 100
    mask_input: bool = False


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv1d(cin, cout, 7, stride=stride, padding=3),
                         nn.GroupNorm(4, cout), nn.GELU())


class SpectralBranch(nn.Module):
    def __init__(self, mode, scales, rate):
        super().__init__()
        self.mode, self.scales, self.rate = mode, scales, rate
        self.cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(8, 16, 3, padding=1), nn.GELU())
        self.register_buffer("window", torch.hann_window(128), persistent=False)

    def forward(self, x, tokens):
        b, c, t = x.shape
        # FFT/eigendecomposition remain FP32 under AMP. All transforms use the
        # supplied (possibly masked) waveform: no clean-target path during MWM.
        with torch.autocast(device_type=x.device.type, enabled=False):
            flat = x.float().reshape(b * c, t)
            if self.mode == "stft":
                z = torch.stft(flat, n_fft=128, hop_length=16,
                               window=self.window, return_complex=True).abs()
            else:
                # Analytic Morlet frequency-domain filter bank, padded to suppress
                # circular wrap. Only magnitudes are used, not a learned CWT.
                nfft = 2 ** math.ceil(math.log2(t + 2048))
                f = torch.fft.fftfreq(nfft, d=1 / self.rate, device=x.device)
                center = torch.logspace(math.log10(.5), math.log10(25), self.scales, device=x.device)
                bank = torch.exp(-.5 * ((f[None] - center[:, None]) / (center[:, None] / 6)) ** 2)
                bank = bank * (f[None] > 0)
                bank = bank / bank.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8) * math.sqrt(nfft)
                spectrum = torch.fft.fft(flat, n=nfft)
                z = torch.fft.ifft(spectrum[:, None] * bank[None], n=nfft)[..., :t].abs()
            z = F.adaptive_avg_pool2d(torch.log1p(z), (z.shape[-2], tokens))
            z = z.reshape(b, c, z.shape[-2], tokens)
        return self.cnn(z).mean(dim=2)


class Physics(nn.Module):
    """Sign-invariant covariance features; optional with amplitude-preserving inputs.

    Polarization is only defined when all three components are present. Records
    with a missing component are zero-filled by the standardizer, so the
    covariance would be degenerate; those samples get a zero feature and a zero
    attention bias instead of an arbitrary direction.
    """
    def __init__(self, dim):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, dim))
        self.alpha = nn.Parameter(torch.tensor(.1))

    def forward(self, x, tokens):
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            windows = F.pad(x.float(), (50, 50)).unfold(2, 101, 16).permute(0, 2, 1, 3)
            windows = windows - windows.mean(-1, keepdim=True)
            cov = windows @ windows.transpose(-1, -2) / 100
            eig, vec = torch.linalg.eigh(cov)
            linear = 1 - eig[..., 1] / eig[..., 2].clamp_min(1e-8)
            v = vec[..., -1]
            projector = v[..., :, None] * v[..., None, :]
            attrs = torch.stack([linear, projector[..., 0, 0], projector[..., 1, 1],
                                 projector[..., 2, 2], projector[..., 0, 1],
                                 projector[..., 0, 2], projector[..., 1, 2]], -1)
            attrs = F.interpolate(attrs.transpose(1, 2), size=tokens, mode="linear", align_corners=False).transpose(1, 2)
            # A component that is identically zero is missing, not a real zero.
            present = (x.abs().amax(dim=-1) > 0).all(dim=1).float()  # (B,)
        e = self.embed(attrs) * present[:, None, None]
        unit = F.normalize(e.float(), dim=-1)
        bias = self.alpha * (unit @ unit.transpose(1, 2)) * present[:, None, None]
        return e, bias[:, None]


def rope(x):
    n, d = x.shape[-2:]
    pos = torch.arange(n, device=x.device, dtype=torch.float32)
    inv = 10000 ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    phase = pos[:, None] * inv[None]
    co, si = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * co - b * si, a * si + b * co), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.up, self.down = nn.Linear(dim, dim * 4), nn.Linear(dim * 2, dim)

    def forward(self, x, bias=None):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=None if bias is None else bias.to(q.dtype))
        x = x + self.proj(y.transpose(1, 2).reshape(b, n, d))
        a, gate = self.up(self.norm2(x)).chunk(2, -1)
        return x + self.down(a * F.silu(gate))


class SeisFormer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.dim % c.heads or (c.dim // c.heads) % 2 or c.dim % 4:
            raise ValueError("dim must be divisible by 4 and head dimension must be even")
        in_channels = 3 + 4 if c.mask_input else 3
        self.stem = nn.ModuleList([conv(in_channels, 24, 2), conv(24, 32, 2), conv(32, 48, 2), conv(48, c.dim, 2)])
        self.wavelet = SpectralBranch("wavelet", c.scales, c.sample_rate) if c.wavelet else None
        self.stft = SpectralBranch("stft", c.scales, c.sample_rate) if c.stft else None
        if c.gabor:
            from seisformer.gabor import GaborBank
            self.gabor = GaborBank(n_filters=16, sample_rate=c.sample_rate, out_channels=16)
            gabor_ch = 16
        else:
            self.gabor = None
            gabor_ch = 0
        self.fusion = nn.Conv1d(c.dim + 16 * (int(c.wavelet) + int(c.stft)) + gabor_ch, c.dim, 1)
        self.physics = Physics(c.dim) if c.polarization else None
        self.blocks = nn.ModuleList([Block(c.dim, c.heads) for _ in range(c.layers)])
        self.decode = nn.ModuleList([conv(c.dim + 48, 48), conv(48 + 32, 32), conv(32 + 24, 24), conv(24 + 3, 24)])
        self.phase_head = nn.Conv1d(24, 2, 1)
        self.det_head = nn.Sequential(nn.Linear(c.dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.reconstruction = nn.Conv1d(24, 3, 1)

    def _encode(self, x, mask=None):
        # The decoder skips always use the 3-channel waveform; only the stem
        # input is widened so the network can separate missing samples from
        # genuine zeros.
        if self.config.mask_input:
            if mask is None:
                raise ValueError("mask_input=True requires a (B, 4, T) validity mask")
            if mask.shape[0] != x.shape[0] or mask.shape[-1] != x.shape[-1]:
                raise ValueError("Mask shape does not match waveform")
            h = torch.cat((x, mask.to(x.dtype)), 1)
        else:
            h = x
        skips = [x]
        for stem in self.stem:
            h = stem(h)
            skips.append(h)
        z = [h]
        for branch in (self.wavelet, self.stft):
            if branch is not None:
                z.append(branch(x, h.shape[-1]))
        if self.gabor is not None:
            g = self.gabor(x, mask if self.config.mask_input else None)
            if g.shape[-1] != h.shape[-1]:
                g = F.interpolate(g, size=h.shape[-1], mode="linear", align_corners=False)
            z.append(g)
        h = self.fusion(torch.cat(z, 1)).transpose(1, 2)
        bias = None
        if self.physics is not None:
            e, bias = self.physics(x, h.shape[1])
            h = h + e
        for block in self.blocks:
            h = block(h, bias)
        return h, skips

    def _decode(self, h, skips):
        h = h.transpose(1, 2)
        for decode, skip in zip(self.decode, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = decode(torch.cat((h, skip), 1))
        return h

    def forward(self, x, mask=None):
        h, skips = self._encode(x, mask)
        det = self.det_head(h.mean(1)).squeeze(-1)
        feats = self._decode(h, skips)
        return {"phase_logits": self.phase_head(feats), "event_logit": det,
                "reconstruction": self.reconstruction(feats), "features": feats,
                "tokens": h}

    def configuration(self):
        return asdict(self.config)
