"""항목 2 검증: Transformer 모델 연결 테스트.

GPU 환경에서 실행:
    cd /lustrefs/team-multimodal/minsu/ComfyUI/custom_nodes/ComfyUI-MotifVideo1.9B
    python tests/test_transformer_connection.py

테스트 항목:
  1. motif_core import 가능 여부
  2. MotifVideoTransformer3DModel 로드 (config.json 기반)
  3. MotifVideoModelAdapter forward pass (더미 입력)
  4. MotifVideoLatent process_in/process_out 왕복
  5. in_channels=33 concat_cond 구성 확인
"""

import sys
import os
import traceback

# sys.path 세팅 (__init__.py와 동일)
sys.path.insert(0, "/lustrefs/team-multimodal/minsu/motif-models/packages/motif-core/src")
sys.path.insert(0, "/lustrefs/team-multimodal/minsu/motif-models/packages/motif-pipelines/src")
# ComfyUI를 import할 수 있게
sys.path.insert(0, "/lustrefs/team-multimodal/minsu/ComfyUI")

results = []


def test(name):
    """데코레이터: 테스트 함수를 실행하고 결과 기록."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((name, "PASS", ""))
                print(f"  [PASS] {name}")
            except Exception as e:
                results.append((name, "FAIL", str(e)))
                print(f"  [FAIL] {name}: {e}")
                traceback.print_exc()
        return wrapper
    return decorator


# ====================================================================
# 테스트 1: motif_core import
# ====================================================================
@test("motif_core import")
def test_motif_core_import():
    from motif_core.models.transformers.transformer_motif_video import (
        MotifVideoTransformer3DModel,
    )
    assert MotifVideoTransformer3DModel is not None


# ====================================================================
# 테스트 2: Transformer 로드 (config.json 기반)
# ====================================================================
@test("Transformer from_pretrained (base checkpoint)")
def test_transformer_load():
    import torch
    from motif_core.models.transformers.transformer_motif_video import (
        MotifVideoTransformer3DModel,
    )

    ckpt_dir = "/lustrefs/team-multimodal/checkpoints/base_checkpoint/model/transformer"
    assert os.path.exists(ckpt_dir), f"Checkpoint not found: {ckpt_dir}"

    model = MotifVideoTransformer3DModel.from_pretrained(
        ckpt_dir, torch_dtype=torch.bfloat16
    )
    print(f"    Loaded: {type(model).__name__}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    assert model is not None


# ====================================================================
# 테스트 3: Adapter forward pass (더미 입력, GPU)
# ====================================================================
@test("Adapter forward pass (dummy input)")
def test_adapter_forward():
    import torch
    from motif_core.models.transformers.transformer_motif_video import (
        MotifVideoTransformer3DModel,
    )

    # config.json에서 읽은 파라미터로 작은 모델 생성 (실제 체크포인트 아님)
    model = MotifVideoTransformer3DModel(
        in_channels=33,
        out_channels=16,
        num_attention_heads=4,   # 축소 (원본 12)
        attention_head_dim=32,   # 축소 (원본 128)
        num_layers=1,            # 축소 (원본 12)
        num_single_layers=1,     # 축소 (원본 24)
        num_decoder_layers=1,    # 축소 (원본 8)
        text_embed_dim=64,       # 축소 (원본 2560)
        patch_size=2,
        patch_size_t=1,
        rope_axes_dim=(4, 8, 8), # 축소 (원본 16, 56, 56)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=torch.float32)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.adapter import MotifVideoModelAdapter

    adapter = MotifVideoModelAdapter(model)
    adapter.eval()

    B, T, H, W = 1, 5, 16, 16
    x = torch.randn(B, 33, T, H, W, device=device)
    t = torch.tensor([500.0], device=device)
    context = torch.randn(B, 10, 64, device=device)
    mask = torch.ones(B, 10, device=device)

    with torch.no_grad():
        out = adapter(x, t, context=context, encoder_attention_mask=mask)

    assert out.shape == (B, 16, T, H, W), f"Expected (1,16,5,16,16), got {out.shape}"
    print(f"    Output shape: {out.shape}")


# ====================================================================
# 테스트 4: MotifVideoLatent process_in / process_out 왕복
# ====================================================================
@test("MotifVideoLatent round-trip")
def test_latent_format():
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.latent_format import MotifVideoLatent

    fmt = MotifVideoLatent()
    assert fmt.latent_channels == 16
    assert fmt.latent_dimensions == 3

    latent = torch.randn(1, 16, 4, 8, 8)
    processed = fmt.process_in(latent)
    recovered = fmt.process_out(processed)

    diff = (latent - recovered).abs().max().item()
    assert diff < 1e-4, f"Round-trip error too large: {diff}"
    print(f"    Round-trip max error: {diff:.2e}")


# ====================================================================
# 테스트 5: in_channels=33 확인 (config.json)
# ====================================================================
@test("in_channels=33 from config.json")
def test_in_channels():
    import json

    config_path = "/lustrefs/team-multimodal/checkpoints/base_checkpoint/model/transformer/config.json"
    assert os.path.exists(config_path), f"Config not found: {config_path}"

    with open(config_path) as f:
        config = json.load(f)

    in_ch = config.get("in_channels", None)
    out_ch = config.get("out_channels", None)
    assert in_ch == 33, f"Expected in_channels=33, got {in_ch}"
    assert out_ch == 16, f"Expected out_channels=16, got {out_ch}"
    print(f"    in_channels={in_ch}, out_channels={out_ch}")


# ====================================================================
# 실행
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MotifVideo Transformer Connection Tests")
    print("=" * 60)

    test_motif_core_import()
    test_transformer_load()
    test_adapter_forward()
    test_latent_format()
    test_in_channels()

    print("\n" + "=" * 60)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"Results: {passed} passed, {failed} failed / {len(results)} total")

    if failed > 0:
        print("\nFailed tests:")
        for name, status, err in results:
            if status == "FAIL":
                print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("All tests passed!")
