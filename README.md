# ComfyUI-MotifVideo1.9B

ComfyUI custom nodes for MotifVideo 1.9B video generation model.

## Features

- **Load Diffusion Model** 호환 — 기존 ComfyUI 노드로 transformer 로드 (FP8 지원)
- **Load MotifVideo Text Encoder** — T5Gemma2 text encoder를 ComfyUI CLIP 시스템으로 로드
- **MotifVideo Text Encode** — positive/negative 프롬프트 편의 노드
- **Empty MotifVideo Latent** — 비디오 latent 생성 (1280x736, 121 frames 등)
- **Load MotifVideo VAE** — diffusers 포맷 3D 비디오 VAE 로드 (키 자동 변환)
- **MotifVideo TeaCache** — TeaCache 가속으로 샘플링 속도 향상 (캐시 기반 블록 스킵)
- ComfyUI 내장 메모리 관리, offload, FP8 변환 자동 적용

## 설치

### 의존성

```bash
cd /path/to/ComfyUI
pip install -r custom_nodes/ComfyUI-MotifVideo1.9B/requirements.txt
```

motif_core, motif-pipelines 별도 설치 불필요. MotifVideoTransformer3DModel이 `models/transformer/`에 내장되어 있어 독립 배포가 가능합니다.

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
[MotifVideo TeaCache]           → MODEL (TeaCache 가속 적용)
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
| MotifVideo TeaCache | model, rel_l1_thresh, enable, start, end, calibrate | MODEL | TeaCache 가속 적용 (샘플링 속도 향상) |
| Load MotifVideo Text Encoder | clip_name, dtype | CLIP | T5Gemma2 텍스트 인코더 로드 |
| MotifVideo Text Encode | CLIP, text, negative_prompt | CONDITIONING x2 | 프롬프트 인코딩 |
| Empty MotifVideo Latent | width, height, num_frames, batch_size | LATENT | 빈 비디오 latent |
| Load MotifVideo VAE | vae_name | VAE | diffusers 포맷 3D VAE 로드 (키 자동 변환) |

### TeaCache 파라미터

**MotifVideo TeaCache** 노드 파라미터:
- `rel_l1_thresh` (float): 캐시 재사용 임계값. 낮을수록 더 공격적 캐싱으로 빠르지만 품질 저하 위험. 높을수록 안전하지만 속도 향상 제한. 권장값: **0.15–0.3**
- `enable` (boolean): True = TeaCache 적용, False = 비활성화 (A/B 비교 용도)
- `start` (float, 0.0–1.0): TeaCache가 활성화되는 샘플링 진행도 시작점. 0.0=샘플링 시작(고노이즈), 1.0=샘플링 끝(클린). 초기 스텝(코스 구조 결정 구간)은 캐싱하지 않는 것이 품질에 유리하므로 0.1–0.2 권장. 기본값: **0.0** (전체 구간 캐싱)
- `end` (float, 0.0–1.0): TeaCache가 비활성화되는 샘플링 진행도 종료점. 후반 스텝(세부 디테일 결정 구간)은 캐싱하지 않는 것이 품질에 유리하므로 0.8–0.9 권장. 기본값: **1.0** (전체 구간 캐싱)
- `calibrate` (boolean): True = 캘리브레이션 모드. 캐싱을 하지 않고 모든 스텝에서 full forward를 실행하며 (raw_diff, output_diff) 데이터를 수집합니다. 수집된 데이터로 polynomial coefficients를 재학습하여 정확도를 높일 수 있습니다. 기본값: **False**

## 체크포인트 교체

학습된 새 transformer 체크포인트로 교체:

```bash
# 새 체크포인트 심링크
ln -s /path/to/new_checkpoint/transformer/diffusion_pytorch_model.safetensors \
  models/diffusion_models/motifvideo_new.safetensors
```

ComfyUI에서 `Load Diffusion Model` → `motifvideo_new` 선택.

### 기본 체크포인트 변경 안내 (2026-04-04)

기본 체크포인트(`motifvideo_1.9b`)가 cross-attn fine-tune 체크포인트(720p-6_400)로 변경되었습니다.

- 새 기본 체크포인트: `motif-video-1.9b-720p-6_400` (cross-attn fine-tune)
  - `enable_text_cross_attention_single=True` 지원
  - state_dict 키 기반 자동 감지 (`single_transformer_blocks.0.cross_attn_query_proj.weight`)
- 이전 기본 체크포인트(`motifvideo_1.9b`)를 계속 사용하려면 별도 심링크 필요:

```bash
ln -s /path/to/original_checkpoint/transformer/diffusion_pytorch_model.safetensors \
  models/diffusion_models/motifvideo_1.9b_original.safetensors
```

## 아키텍처

- Transformer: MotifVideoTransformer3DModel (`models/transformer/transformer_motif_video.py` 내장)
  - cross-attn variant 자동 감지: state_dict 키(`single_transformer_blocks.0.cross_attn_query_proj.weight`, `transformer_blocks.0.cross_attn_query_proj.weight`) 기반으로 `enable_text_cross_attention_single/dual` 자동 설정
  - patch_size, patch_size_t: state_dict에서 동적 감지 (하드코딩 제거)
- Text Encoder: T5Gemma2Model (transformers 5.0.0+)
- VAE: AutoencoderKLWan (diffusers 키 자동 변환 → ComfyUI WAN VAE)
- Scheduler: FlowMatchEulerDiscreteScheduler (KSampler 연동)

### Upstream 동기화

motif_core 원본이 변경되면 수동으로 내장 파일을 업데이트해야 합니다.

| 내장 파일 | 원본 위치 | 비고 |
|-----------|-----------|------|
| `models/transformer/transformer_motif_video.py` | `motif-core/src/motif_core/models/transformers/` | import 경로 변환 필요 |
| `models/transformer/tread_mixin.py` | `motif-core/src/motif_core/models/mixin/` | loguru→logging 교체 필요 |
| `models/transformer/accelerate_patch.py` | — | no-op stub (동기화 불필요) |

복사 후 `from motif_core.` → 로컬 상대 import, `from loguru import logger` → `import logging` 교체 필요.

## Jira

- [MM-959](https://motiftech-kr-team.atlassian.net/browse/MM-959)
- [MM-1036](https://motiftech-kr-team.atlassian.net/browse/MM-1036)
