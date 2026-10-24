# XFNet: Compact Time–Frequency Fusion for P- and S-Wave Arrival Picking

Official open-source implementation (`seisformer` Python package), preserved primary model checkpoint, and reproducibility materials for:

> **XFNet: compact time–frequency fusion for P- and S-wave arrival picking across multi-source seismic archives**  
> **Authors:** Wei Xiang$^{1,2,*}$, Jianying Mou$^{1}$  
> $^{1}$ Guangxi Zhuang Autonomous Region Earthquake Agency, Nanning 530022, China  
> $^{2}$ Beihai Weizhou Island Marine Geophysics Observation and Research Station of Guangxi, Weizhou Island Earthquake Monitoring Center Station, Beihai 536000, China  
> $^{*}$ Corresponding author: `xiangwei@gxearthquake.cn` (ORCID: `0009-0001-4572-3507`)

---

## Highlights

- **Compact Spectral Fusion:** Aligns fixed 12-band Morlet filterbank and 65-bin STFT views to a 375-token convolutional representation of a 60-s, 100-Hz 3-component waveform, adding only **5,856 trainable parameters (1.38%)** for a total of **428,774 parameters**.
- **Controlled Ablation:** Across three paired fine-tuning seeds on 262,144 training records, the spectral paths reduce mean P/S arrival MAE from **0.551 / 0.346 s** (temporal control) to **0.473 / 0.313 s**.
- **Matched Recent-Model Benchmark:** Trained from scratch under an identical 262k, 20-epoch, 3-seed budget, XFNet achieves **0.425 / 0.295 s** P/S MAE versus **0.751 / 0.376 s** for TwoStream-EQT, with **31.1% fewer parameters** and lower single-window latency (**3.66 ms vs. 9.91 ms** on an RTX 5070 Ti).
- **Primary 1M Model:** Trained on 1,048,576 standardized records, the preserved primary checkpoint (`weights/best_detection.pt`, SHA256: `08afa0d899bbc03e5680dff2b49768aacb1ce5fdfe07525d037b1c87e9ff3342`) reaches **0.266 / 0.216 s** pooled P/S MAE (**81.9% / 72.9%** of picks within $\pm 0.1$ s; unweighted non-NEIC macro-mean **0.422 / 0.515 s**), picking F1 of **0.908 / 0.880** within 0.3 s, and window-level detection F1 of **0.9988** across the 127,940-record multi-source test cohort.
- **Source-Specific Conformal Calibration:** Finite-sample residual calibration reduces mean absolute deviation from nominal 90% coverage from **0.132** to **0.017** across 16 source–phase groups.

---

## Repository Structure

- `seisformer/` — Core PyTorch implementation of XFNet (temporal stem, Morlet filterbank, STFT branch, RoPE Transformer, skip-connected decoder, physical resampling/standardization, and conformal calibration).
- `weights/best_detection.pt` — Preserved primary 1M FP32 checkpoint (428,774 parameters; 5.2 MB).
- `config/` — Model configuration (`model_config.json`), global and Mondrian source-specific 90% conformal calibration radii (`conformal_90.json`, `mondrian_90.json`), and phase definitions.
- `infer.py` — Standalone command-line inference script supporting raw/standardized NumPy arrays (`.npy`/`.npz`) and continuous MiniSEED (`.mseed`) records.
- `reproducibility/` — Frozen train/validation/calibration/test row manifests (`row_manifests/*.json.gz`), evidence summaries (`submission_evidence.json`), held-out waveform data (`heldout_waveform_data.npz`), runtime logs (`primary_runtime.json`), and paper figure/table reproduction scripts (`source_scripts/`).
- `figures/` — Architecture diagrams and manuscript figures.

---

## Quick Start

### 1. Environment Setup

Requires **Python 3.10+** and **PyTorch >= 2.0**:

```bash
pip install -r requirements.txt
```

### 2. Run Standalone Inference

Run inference on a raw 3-component waveform array (`[B, 3, N]` or `[3, N]`) with automatic resampling to 100 Hz and ENZ standardization:

```bash
python infer.py --input demo/input_raw.npy --sample-rate 100 --order ENZ --output demo/predictions.json
```

For pre-standardized `B x 3 x 6000` (60 s at 100 Hz) windows:

```bash
python infer.py --input demo/input_standardized.npz --standardized --output demo/predictions.json
```

For MiniSEED (`.mseed`) waveforms (requires `obspy`):

```bash
python infer.py --input path/to/record.mseed --min-support 5 --output picks.json
```

---

## Citation

```bibtex
@article{xiang2026xfnet,
  title   = {XFNet: compact time--frequency fusion for P- and S-wave arrival picking across multi-source seismic archives},
  author  = {Xiang, Wei and Mou, Jianying},
  journal = {Computers \& Geosciences},
  year    = {2026}
}
```

## Funding

Supported by the “Leitai” Action Plan (Guangxi Science and Technology Innovation Platform Program), under the capacity-building initiative for autonomous-region-level science and technology innovation platforms (project agreement No. Guike LT2600640028 / 桂科LT2600640028).

## License

Released under the [MIT License](LICENSE). Source waveform collections (STEAD, INSTANCE, PNW, TXED, ETHZ, IQUIQUE, GEOFON, CREW, NEIC, CEED) remain subject to their respective data provider licenses.
