<p align="center">
  <img src="assets/banner.png" width="100%" alt="Motif-Video 2B teaser"/>
</p>

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

These nodes are first-class citizens of ComfyUI's memory manager: FP8 weight conversion, CPU offload, and attention-backend auto-selection (Flash / cuDNN / xFormers) are all delegated to the engine rather than reimplemented. What this repository adds on top is MotifVideo-specific glue: the T5Gemma2 text encoder, the Wan-family 3D VAE in diffusers layout, a TeaCache accelerator, and an Image-to-Video conditioning node.

<p align="center">
  <img src="assets/demo.gif" width="100%" alt="ComfyUI-MotifVideo2B demo"/>
</p>

---

## Installation

### 1. Install the custom nodes

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/MotifTechnologies/ComfyUI-MotifVideo2B.git
pip install -r ComfyUI-MotifVideo2B/requirements.txt
```

`motif_core` and `motif-pipelines` do not need to be installed separately — `MotifVideoTransformer3DModel` is bundled under `models/transformer/`, so the repository is self-contained.

### 2. Download the model weights from Hugging Face

All weights live on the official Hugging Face repository:

- 🤗 <https://huggingface.co/Motif-Technologies/Motif-Video-2B>

Download the files listed below and place them under ComfyUI's standard model directories. The filenames and target directories shown here are the ones the example workflows load by default — pick your own names if you prefer, but keep the target directory the same.

```
ComfyUI/
├── models/
│   ├── diffusion_models/
│   │   └── motifvideo_2b.safetensors         ← transformer/diffusion_pytorch_model.safetensors
│   ├── text_encoders/
│   │   ├── motifvideo_t5gemma2/                ← text_encoder/ (entire directory)
│   │   └── motifvideo_tokenizer/               ← tokenizer/ (entire directory)
│   └── vae/    
│       └── motifvideo_vae.safetensors          ← vae/diffusion_pytorch_model.safetensors
```

The easiest way to fetch all of them at once is `huggingface-cli`:

```bash
huggingface-cli download Motif-Technologies/Motif-Video-2B \
  --local-dir /tmp/motif-video-2b
# then copy or move the four pieces into the directories shown above
```

The VAE is in diffusers layout; its `state_dict` keys are remapped to ComfyUI's WAN VAE at load time, so no manual conversion is needed.

> **Filename note.** The exact on-disk filename of the transformer is arbitrary — it just needs to match the value selected in the `UNETLoader` node of the workflow you load. The example workflows under `workflows/` ship with their own filenames bound; either rename your download to match, or open the workflow and change the `UNETLoader` value to whatever you saved locally. The text encoder directory (`motifvideo_t5gemma2/`) and VAE (`motifvideo_vae.safetensors`) use the fixed names above in both shipped workflows.

---

## Usage

### Recommended: launch ComfyUI with `--highvram`

On a host with enough VRAM (H200 or similar), use `--highvram`:

```bash
python main.py --highvram --listen 0.0.0.0 --port 8188
```

- Without `--highvram` (default `NORMAL_VRAM`): a bf16 workflow runs at roughly **222 s/step** — the transformer is placed on the "staged" path and weights are dispatched every forward.
- With `--highvram`: **30 s/step** — all weights stay resident on the GPU.

**Why.** On hosts where ComfyUI's `comfy_aimdo` (`DynamicVRAM`) is active, models whose leaves use `comfy.ops.*` are automatically routed to the staged path under `NORMAL_VRAM`, which means weight dispatch on every forward. This repository's transformer deliberately uses `comfy.ops.*` end-to-end so that `fp8`/`manual_cast` paths work, which means the staging cannot be disabled at the model level. The engine-side workaround is `--highvram`. Tracked in #26.


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

## Workflow

The standard MotifVideo sampling graph flows `UNETLoader → ModelSamplingSD3 → KSampler → VAE Decode → Create Video → Save Video`, with `MotifTextEncoderLoader + MotifTextEncode` feeding the KSampler conditioning and `EmptyMotifLatent` seeding the latent. The complete wiring is in [`workflows/Motif-2B_T2V_example.json`](workflows/Motif-2B_T2V_example.json) — load it through ComfyUI's **Load** menu rather than rebuilding the graph by hand.

---

## Performance

Measured on a single H200 with the default 1280×736, 121-frame workflow:

| Setup | VRAM peak | s/step | Notes |
|-------|-----------|--------|-------|
| bf16 + `--highvram` | ~30 GB | 30 s | Baseline |
| **fp8_e4m3fn + `--highvram`** | **~28 GB** | **~31 s** | **Recommended** production path |
| fp8 + `NORMAL_VRAM` | — | — | Avoid — staged path and earlier fallback regression can re-emerge |

---

## Image-to-Video

For I2V, `MotifVideo Image Encode` sits between `MotifVideo Text Encode` and `KSampler`: it VAE-encodes the input image and injects it into the conditioning as `concat_latent_image`, so downstream nodes continue to see a normal `CONDITIONING` pair. The full wiring is in [`workflows/i2v_example.json`](workflows/i2v_example.json).

Recommended parameters, as shipped in `i2v_example.json`:
- `ModelSamplingSD3` shift = 2.5
- `KSampler` cfg = 8.0, steps = 50, sampler = `euler`, scheduler = `simple`
- `EmptyMotifLatent` 1280×736, 121 frames

Switch back to T2V by removing the `MotifVideo Image Encode` node and wiring `MotifVideo Text Encode` straight into `KSampler` — or just load the T2V workflow instead.

---

## Workflows

Reference workflows live under `workflows/`:

- [`workflows/Motif-2B_T2V_example.json`](workflows/Motif-2B_T2V_example.json) — default Text-to-Video graph (1280×736, KSampler sampler `dpmpp_2m_sde` / scheduler `simple` / 50 steps, wrapped as a reusable ComfyUI subgraph).
- [`workflows/i2v_example.json`](workflows/i2v_example.json) — Image-to-Video graph (1280×736 / 121 frames, KSampler `euler` / `simple` / 50 steps, `ModelSamplingSD3` shift 2.5).

Load either JSON from ComfyUI's **Load** menu. Make sure the model files described in the [Installation](#installation) section are in place first, and that the `UNETLoader`/text-encoder/VAE selections in the workflow match your local filenames.

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
