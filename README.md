<p align="center">
  <h1 align="center">ComfyUI-MotifVideo2B</h1>
</p>

<p align="center">
  <b>Official ComfyUI custom nodes for the Motif-Video 2B text-to-video diffusion transformer</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.16503">Technical Report</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/Motif-Technologies/Motif-Video-2B">Hugging Face</a> &nbsp;|&nbsp;
  <a href="https://motiftech.io/videoshowcase">Project Page</a>
</p>


---

## Introduction

`ComfyUI-MotifVideo2B` exposes Motif Technologies' Motif-Video 2B text-to-video and image-to-video diffusion transformer as a set of ComfyUI custom nodes, so the model plugs directly into the standard `Load Diffusion Model → KSampler → VAE Decode` graph.

Motif-Video 2B is a flow-matching diffusion transformer organized around a three-stage DDT-style backbone (dual-stream + single-stream + DDT decoder) with **Shared Cross-Attention** for long-context text alignment. The architectural derivation and full training recipe are in the [Motif-Video 2B technical report](https://arxiv.org/abs/2604.16503); this repository ships the inference-time ComfyUI integration.


---

## Installation

### 1. Install the custom nodes (recommended)

Clone the repository and install dependencies:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/MotifTechnologies/ComfyUI-MotifVideo2B.git
pip install -r ComfyUI-MotifVideo2B/requirements.txt
```

`motif_core` and `motif-pipelines` do not need to be installed separately — `MotifVideoTransformer3DModel` is bundled under `models/transformer/`, so the repository is self-contained.

### Install via ComfyUI-Manager

> **Heads-up:** an upstream `custom-node-list.json` registration PR is planned and will be opened shortly. This section will be revised after that PR merges.

ComfyUI-Manager exposes two install paths for this repository, both subject to its `security_level` setting:

- **Custom Nodes Manager** (search and install from the registry): not available yet — a `custom-node-list.json` registration PR to upstream Manager is **planned but not yet submitted**. Until that PR is opened and merged, this repository will not appear in Manager's search results.
- **Install via Git URL** (paste a Git URL): under Manager's default `normal` security level, this path is rejected outright with `This action is not allowed with this security level configuration.` (the upstream policy requires `security_level = normal-` with `--listen` on a local IP, or `security_level = middle`/`weak`). It is not specific to externally-exposed setups.

If neither Manager path works for you, fall back to the [git clone method above](#1-install-the-custom-nodes-recommended). Whether the **Custom Nodes Manager** search-and-install path works under the default `normal` security level after the registration PR merges depends on upstream Manager and registry policy at that time and is not something this repository can guarantee. The **Install via Git URL** path remains gated by Manager's security policy regardless of registration.

When you use either of the two Manager paths above, Manager runs `install.py` automatically, which installs the entries in `requirements.txt`. (The manual `git clone` path documented earlier does not involve Manager — there you run `pip install -r requirements.txt` yourself.) After restarting and loading either example workflow, the automatic model-download dialog described in [Automatic model download](#automatic-model-download) will pull the three weight files from Hugging Face on first use, regardless of how the custom node was installed.

### 2. Download the model weights from Hugging Face

All weights live on the official Hugging Face repository:

- 🤗 <https://huggingface.co/Motif-Technologies/Motif-Video-2B>

Download the three files listed below and place them under ComfyUI's standard model directories. The filenames and target directories shown here are the ones the example workflows load by default — pick your own names if you prefer, but keep the target directory the same.

```
ComfyUI/
├── models/
│   ├── diffusion_models/
│   │   └── motifvideo_2b.safetensors       ← transformer/diffusion_pytorch_model.safetensors
│   ├── text_encoders/
│   │   └── motifvideo_t5gemma2.safetensors ← text_encoder/model.safetensors (rename when you save it)
│   └── vae/
│       └── motifvideo_vae.safetensors      ← vae/diffusion_pytorch_model.safetensors
```

Use `huggingface-cli` to fetch the three files you need:

```bash
# Option A: huggingface-cli (fetch only the files you need)
huggingface-cli download Motif-Technologies/Motif-Video-2B \
  transformer/diffusion_pytorch_model.safetensors \
  text_encoder/model.safetensors \
  vae/diffusion_pytorch_model.safetensors \
  --local-dir /tmp/motif-video-2b
```

After download, rename the files to match the local naming convention above: `transformer/diffusion_pytorch_model.safetensors` → `motifvideo_2b.safetensors`, `text_encoder/model.safetensors` → `motifvideo_t5gemma2.safetensors`, and `vae/diffusion_pytorch_model.safetensors` → `motifvideo_vae.safetensors`. Then place each in its target directory shown above.

The VAE is in diffusers layout; its `state_dict` keys are remapped to ComfyUI's WAN VAE at load time, so no manual conversion is needed.

### Automatic model download

Recent ComfyUI versions read the `models` manifest embedded in `Motif-2B_T2V_example.json` / `Motif-2B_I2V_example.json` and offer a one-click dialog to pull the three weight files straight from Hugging Face the first time you open the workflow. Older ComfyUI installs ignore the manifest and fall back to the manual `huggingface-cli` path above.

If the download dialog does not appear right away on the first drop, drop the workflow once more — ComfyUI's model directory scan can still be initializing.

Tested on ComfyUI v0.18.0 (frontend v1.25.3).

---

## Usage

### Recommended: launch ComfyUI with `--highvram`

On a host with enough VRAM (H200 or similar), use `--highvram`:

```bash
python main.py --highvram --listen 0.0.0.0 --port 8188
```

- Without `--highvram` (default `NORMAL_VRAM`): a bf16 workflow runs at roughly **222 s/step** — the transformer is placed on the "staged" path and weights are dispatched every forward.
- With `--highvram`: **30 s/step** — all weights stay resident on the GPU.

**Why.** On hosts where ComfyUI's `comfy_aimdo` (`DynamicVRAM`) is active, models whose leaves use `comfy.ops.*` are automatically routed to the staged path under `NORMAL_VRAM`, which means weight dispatch on every forward. This repository's transformer deliberately uses `comfy.ops.*` end-to-end so that `fp8`/`manual_cast` paths work, which means the staging cannot be disabled at the model level. The engine-side workaround is `--highvram`.


---

## Nodes

| Node | Inputs | Outputs | Description |
|------|--------|---------|-------------|
| Load MotifVideo Text Encoder | clip_name, dtype | CLIP | Loads the T5Gemma2 encoder and exposes it as a ComfyUI `CLIP` |
| MotifVideo Text Encode | CLIP, text, negative_prompt | CONDITIONING × 2 | Encodes positive and negative prompts in a single node |
| Empty MotifVideo Latent | width, height, num_frames, batch_size | LATENT | Empty video latent sized for the Wan-family VAE |
| Load MotifVideo VAE | vae_name | VAE | Loads the 3D VAE in diffusers layout with automatic key remapping |
| MotifVideo Image Encode | positive, negative, VAE, IMAGE | CONDITIONING × 2 | I2V: VAE-encodes the input image and injects it into the conditioning |

---

## Performance

Measured on a single H200 with the default 1280×736, 121-frame workflow:

| Setup | VRAM peak | s/step | Notes |
|-------|-----------|--------|-------|
| bf16 + `--highvram` | ~29.28 GB | 30 s | Baseline |
| **fp8_e4m3fn + `--highvram`** | **~28.68 GB** | **~31 s** | **Recommended** production path |
| fp8 + `NORMAL_VRAM` | — | — | Avoid — staged path and earlier fallback regression can re-emerge |

---

## Text-to-Video

<p align="center">
  <img src="assets/demo.gif" width="100%" alt="ComfyUI-MotifVideo2B demo"/>
</p>

T2V is the default sampling path: `MotifVideo Text Encode` feeds `KSampler` directly, with no image-conditioning branch. The full wiring is in [`workflows/Motif-2B_T2V_example.json`](workflows/Motif-2B_T2V_example.json), shipped as a reusable ComfyUI subgraph.

Recommended parameters, as shipped in `Motif-2B_T2V_example.json`:
- `ModelSamplingSD3` shift = 15
- `APG` eta = 0, norm_threshold = 12, momentum = 0.1 (Adaptive Projected Guidance, between ModelSamplingSD3 and KSampler)
- `KSampler` cfg = 8.0, steps = 50, sampler = `dpmpp_2m`, scheduler = `simple`
- `EmptyMotifLatent` 1280×736, 33, 65 or 121 frames

---

## Image-to-Video

<p align="center">
  <img src="assets/demo_i2v.gif" width="100%" alt="ComfyUI-MotifVideo2B I2V demo"/>
</p>

For I2V, `MotifVideo Image Encode` sits between `MotifVideo Text Encode` and `KSampler`: it VAE-encodes the input image and injects it into the conditioning as `concat_latent_image`, so downstream nodes continue to see a normal `CONDITIONING` pair. The full wiring is in [`workflows/Motif-2B_I2V_example.json`](workflows/Motif-2B_I2V_example.json).

The shipped I2V workflow's `LoadImage` node points at `i2v_sample.jpg`. Copy [`assets/i2v_sample.jpg`](assets/i2v_sample.jpg) into ComfyUI's `input/` folder before running it, or load any other image of your own.

Recommended parameters, as shipped in `Motif-2B_I2V_example.json`:
- `ModelSamplingSD3` shift = 8
- `APG` eta = 0, norm_threshold = 12, momentum = 0.1 (Adaptive Projected Guidance, between ModelSamplingSD3 and KSampler)
- `KSampler` cfg = 8.0, steps = 50, sampler = `dpmpp_2m`, scheduler = `simple`
- `EmptyMotifLatent` 1280×736, 33, 65 or 121 frames

Switch back to T2V by removing the `MotifVideo Image Encode` node and wiring `MotifVideo Text Encode` straight into `KSampler` — or just load the T2V workflow instead.

---

## Workflows

The standard MotifVideo sampling graph flows `UNETLoader → ModelSamplingSD3 → KSampler → VAE Decode → Create Video → Save Video`, with `MotifTextEncoderLoader + MotifTextEncode` feeding the KSampler conditioning and `EmptyMotifLatent` seeding the latent. Load the example JSON through ComfyUI's **Load** menu rather than rebuilding the graph by hand:

- [`workflows/Motif-2B_T2V_example.json`](workflows/Motif-2B_T2V_example.json) — Text-to-Video graph 
- [`workflows/Motif-2B_I2V_example.json`](workflows/Motif-2B_I2V_example.json) — Image-to-Video graph 

Make sure the model files described in the [Installation](#installation) section are in place first, and that the `UNETLoader` / text-encoder / VAE selections inside the loaded workflow match your local filenames.

---

## Citation

If you use Motif-Video 2B in your research, please cite the technical report:

```bibtex
@techreport{motifvideo2b2026,
  title       = {Motif-Video 2B: Technical Report},
  author      = {Motif Technologies},
  year        = {2026},
  institution = {Motif Technologies},
  url         = {https://arxiv.org/abs/2604.16503}
}
```

---

## License

This repository is released under the **Apache 2.0** License. See `LICENSE` for details.
