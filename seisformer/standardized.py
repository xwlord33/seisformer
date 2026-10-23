"""Standardized full-context waveforms: explicit coordinates and missingness."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import resample_poly

RATE = 100
VERSION = "standardized-v1"
DIRECT_PHASES = {"P", "Pg", "Pn", "P1", "S", "Sg", "Sn", "S1"}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_source_trace(h, name):
    if "$" not in name:
        return np.asarray(h["data"][name], dtype=np.float64)
    key, expression = name.split("$", 1)
    parts = expression.split(",")
    selection = [int(parts[0])]
    for part in parts[1:]:
        if ":" not in part:
            raise ValueError("Unsupported bucket slice")
        selection.append(slice(*(int(v) if v else None for v in part.split(":"))))
    return np.asarray(h["data"][key][tuple(selection)], dtype=np.float64)


def standardize(raw, rate, order, dimension="CW"):
    """Return ENZ at 100 Hz, shared RMS scaling, and conservative FIR masks.

    Unknown sensor directions are not guessed. A constant component is missing.
    Missing intervals are zero-filled only numerically and remain invalid in mask.
    Center and scale permit approximate recovery of the original signal units.
    """
    if dimension == "WC":
        raw = raw.T
    if raw.ndim != 2 or raw.shape[0] != len(order) or not 1 <= raw.shape[0] <= 3:
        raise ValueError("Shape/component declaration mismatch")
    if not np.isfinite(rate) or not 0 < rate <= 10000:
        raise ValueError("Invalid source sampling rate")
    if raw.shape[-1] < 2:
        raise ValueError("Trace too short")
    ratio = Fraction(RATE / rate).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    actual_rate = rate * up / down
    # Sub-sample drift bounded across the entire trace, including near-integer Hz.
    drift = abs(raw.shape[-1] * (up / down - RATE / rate))
    if drift > 0.01:
        raise ValueError("Resampling rational approximation exceeds 0.01 sample drift")
    n = math.ceil(raw.shape[-1] * up / down)
    result = np.zeros((3, n), dtype=np.float32)
    mask = np.zeros((3, n), dtype=bool)
    centers = np.zeros(3, dtype=np.float64)
    for output_c, component in enumerate("ENZ"):
        if order.count(component) != 1:
            continue
        a = raw[order.index(component)]
        valid = np.isfinite(a)
        if valid.sum() < 2 or np.ptp(a[valid]) == 0:
            continue
        centers[output_c] = np.mean(a[valid], dtype=np.float64)
        a = np.where(valid, a - centers[output_c], 0)
        if up == down:
            b, good = a, valid
        else:
            b = resample_poly(a, up, down)
            # Default scipy FIR has half-width 10*max(up,down) on the upsampled grid.
            radius = math.ceil(10 * max(up, down) / up) + 1
            bad = maximum_filter1d((~valid).astype(np.uint8), 2 * radius + 1, mode="constant", cval=1)
            nearest = np.minimum((np.arange(n) * down / up).astype(np.int64), len(a) - 1)
            good = bad[nearest] == 0
        result[output_c] = np.where(good, b, 0)
        mask[output_c] = good
    if not mask.any():
        raise ValueError("No finite nonconstant component with known ENZ direction")
    scale = float(np.sqrt(np.mean(np.square(result[mask], dtype=np.float64))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid shared normalization scale")
    result /= scale
    encoded = result.astype(np.float16)
    if not np.isfinite(encoded).all():
        raise ValueError("float16 normalized storage overflow")
    return encoded, mask, centers, scale, {"up": up, "down": down, "effective_rate_hz": actual_rate, "max_label_drift_samples": drift}


def partition(group):
    bucket = int(digest("standardized-v1-split-42:" + group)[:16], 16) % 10000
    return "train" if bucket < 8000 else "val" if bucket < 9000 else "cal" if bucket < 9500 else "test"


class StandardizedWaveforms:
    """Read completed shards; returns masks required by downstream training.

    Crop starts are independent of picks. Unknown/out-of-window event phases and
    event labels without an in-window pick are masked, never relabelled as noise.
    This reader deliberately requires an explicit shard list (or full manifest).
    """
    def __init__(self, root, split="train", samples=6000, seed=42, random_crop=False, limit=0, snapshot=None, only_rows=None):
        self.root = Path(root)
        manifest = json.loads(Path(snapshot if snapshot else self.root / "MANIFEST.json").read_text(encoding="utf-8"))
        if snapshot:
            if manifest.get("status") != "frozen_partial" or Path(manifest["data_root"]).resolve() != self.root.resolve():
                raise ValueError("Invalid partial-data snapshot")
            for shard in manifest["shards"]:
                done = self.root / shard / "COMPLETE.json"
                if hashlib.sha256(done.read_bytes()).hexdigest() != manifest["completion_sha256"][shard]:
                    raise ValueError("Frozen shard changed")
        elif manifest["status"] != "complete":
            raise ValueError("Dataset is not finalized; use explicit completed-shard preview for inspection")
        self.entries = []
        self.samples, self.seed, self.random_crop = samples, seed, random_crop
        selection_rng = np.random.default_rng(seed)
        seen = 0
        for shard in manifest["shards"]:
            path = self.root / shard
            allowed = None
            if only_rows is not None:
                allowed = only_rows.get(str(path))
                if not allowed:
                    continue
            with np.load(path / "index.npz", allow_pickle=False) as z:
                wanted = np.flatnonzero(z["split"] == split)
                if allowed is not None:
                    wanted = np.asarray([i for i in wanted if int(i) in allowed], dtype=np.int64)
                for i in wanted:
                    entry = (str(path), int(i))
                    seen += 1
                    if not limit or len(self.entries) < limit:
                        self.entries.append(entry)
                    else:
                        j = int(selection_rng.integers(seen))
                        if j < limit:
                            self.entries[j] = entry
        self._cache = {}
        self._indexes = {}

    def __len__(self):
        return len(self.entries)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        state["_indexes"] = {}
        return state

    def __getitem__(self, i):
        import torch
        path, row = self.entries[i]
        if not hasattr(self, "_cache"):
            self._cache = {}
        if path not in self._cache:
            if len(self._cache) >= 64:
                self._cache.clear()
            p = Path(path)
            if path not in self._indexes:
                with np.load(p / "index.npz", allow_pickle=False) as z:
                    self._indexes[path] = {k: z[k] for k in z.files}
            idx = self._indexes[path]
            self._cache[path] = (idx, np.memmap(p / "waveforms.f16", mode="r", dtype=np.float16), np.memmap(p / "valid.bits", mode="r", dtype=np.uint8))
        idx, waves, masks = self._cache[path]
        n = int(idx["length"][row])
        start_max = max(0, n - self.samples)
        stable_path = Path(path).relative_to(self.root).as_posix()
        rng = np.random if self.random_crop else np.random.default_rng(int(digest(f"{self.seed}:{stable_path}:{row}")[:16], 16))
        start = int(rng.randint(start_max + 1)) if self.random_crop else int(rng.integers(start_max + 1))
        pos, bit = int(idx["offset"][row]), int(idx["mask_offset"][row])
        raw = waves[pos:pos + 3 * n].reshape(3, n)
        valid = np.unpackbits(masks[bit:bit + math.ceil(3 * n / 8)], count=3 * n).reshape(3, n).astype(bool)
        x = np.zeros((3, self.samples), dtype=np.float32)
        m = np.zeros((3, self.samples), dtype=bool)
        size = min(n, self.samples)
        x[:, :size], m[:, :size] = raw[:, start:start + size], valid[:, start:start + size]
        picks = idx["picks"][row].copy() - start
        known = (picks >= 0) & (picks < size)
        for c in range(2):
            if known[c] and not m[:, int(picks[c])].any():
                known[c] = False
        picks[~known] = -1
        noise = bool(idx["noise"][row])
        return {"x": torch.from_numpy(x), "valid_mask": torch.from_numpy(m),
                "component_mask": torch.from_numpy(m.any(axis=1)), "time_mask": torch.from_numpy(m.any(axis=0)),
                "picks": torch.from_numpy(picks), "phase_mask": torch.from_numpy(known | noise),
                "event": torch.tensor(float(not noise)), "event_mask": torch.tensor(bool(noise or known.any())),
                "source": str(idx["source"][row]), "event_group": str(idx["group"][row]), "row": i}


def standardized_supervised_loss(out, batch):
    """Loss for this reader: invalid time bins and unknown event crops have no gradient."""
    import torch
    from torch.nn import functional as F
    logits = out["phase_logits"].float()
    device = logits.device
    picks = batch["picks"].to(device)
    mask = batch["time_mask"].to(device)[:, None, :] & batch["phase_mask"].to(device)[:, :, None]
    axis = torch.arange(logits.shape[-1], device=device)
    target = torch.exp(-.5 * ((axis[None, None] - picks[..., None]) / 10.) ** 2) * (picks >= 0)[..., None]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    phase = (bce * (1 + 60 * target) * mask).sum() / mask.sum().clamp_min(1)
    event_mask = batch["event_mask"].to(device)
    detection = F.binary_cross_entropy_with_logits(out["event_logit"].float(), batch["event"].to(device), reduction="none")
    detection = (detection * event_mask).sum() / event_mask.sum().clamp_min(1)
    return 2 * phase + detection
