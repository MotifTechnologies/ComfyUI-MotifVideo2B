# ComfyUI-MotifVideo1.9B

ComfyUI custom nodes for MotifVideo 1.9B video generation model.

## Features

- **Load Diffusion Model** 호환 — 기존 ComfyUI 노드로 transformer 로드 (FP8 지원)
- **Load MotifVideo Text Encoder** — T5Gemma2 text encoder를 ComfyUI CLIP 시스템으로 로드
- **MotifVideo Text Encode** — positive/negative 프롬프트 편의 노드
- **Empty MotifVideo Latent** — 비디오 latent 생성 (1280x736, 121 frames 등)
- **Load MotifVideo VAE** — diffusers 포맷 3D 비디오 VAE 로드 (키 자동 변환)
- ComfyUI 내장 메모리 관리, offload, FP8 변환 자동 적용

## 설치

### 의존성

```bash
cd /path/to/ComfyUI
pip install -r custom_nodes/ComfyUI-MotifVideo1.9B/requirements.txt
pip install -e /path/to/motif-models/packages/motif-core
pip install -e /path/to/motif-models/packages/motif-pipelines
```

### 모델 심링크

```bash
# Transformer
ln -s /path/to/checkpoint/transformer/diffusion_pytorch_model.safetensors \
  models/diffusion_models/motifvideo_1.9b.safetensors

# Text Encoder (디렉토리)
ln -s /path/to/checkpoint/text_encoder \
  models/text_encoders/motifvideo_t5gemma2

# Tokenizer (디렉토리)
ln -s /path/to/checkpoint/tokenizer \
  models/text_encoders/motifvideo_tokenizer

# VAE (diffusers 포맷 → 자동 변환)
ln -s /path/to/checkpoint/vae/diffusion_pytorch_model.safetensors \
  models/vae/motifvideo_vae.safetensors
```

## 워크플로우

```
[Load Diffusion Model]          → motifvideo_1.9b (bf16/fp8)
         ↓ MODEL
[Load MotifVideo Text Encoder]  → motifvideo_t5gemma2/model.safetensors
         ↓ CLIP
[MotifVideo Text Encode]        → 프롬프트 입력
         ↓ positive, negative
[Empty MotifVideo Latent]       → 1280x736, 121 frames
         ↓ LATENT
[KSampler]                      ← MODEL + positive + negative + LATENT
         ↓ LATENT
[Load MotifVideo VAE]           → motifvideo_vae.safetensors
         ↓ VAE
[VAE Decode]                    → 비디오 출력
```

## 노드

| 노드 | 입력 | 출력 | 설명 |
|------|------|------|------|
| Load MotifVideo Text Encoder | clip_name, dtype | CLIP | T5Gemma2 텍스트 인코더 로드 |
| MotifVideo Text Encode | CLIP, text, negative_prompt | CONDITIONING x2 | 프롬프트 인코딩 |
| Empty MotifVideo Latent | width, height, num_frames, batch_size | LATENT | 빈 비디오 latent |
| Load MotifVideo VAE | vae_name | VAE | diffusers 포맷 3D VAE 로드 (키 자동 변환) |

## 체크포인트 교체

학습된 새 transformer 체크포인트로 교체:

```bash
# 새 체크포인트 심링크
ln -s /path/to/new_checkpoint/transformer/diffusion_pytorch_model.safetensors \
  models/diffusion_models/motifvideo_new.safetensors
```

ComfyUI에서 `Load Diffusion Model` → `motifvideo_new` 선택.

## 아키텍처

- Transformer: MotifVideoTransformer3DModel (motif_core — 직접 import, 코드 복사 X)
- Text Encoder: T5Gemma2Model (transformers 5.0.0+)
- VAE: AutoencoderKLWan (diffusers 키 자동 변환 → ComfyUI WAN VAE)
- Scheduler: FlowMatchEulerDiscreteScheduler (KSampler 연동)

## Jira

- [MM-959](https://motiftech-kr-team.atlassian.net/browse/MM-959)
