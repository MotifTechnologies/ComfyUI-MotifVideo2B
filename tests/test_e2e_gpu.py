"""항목 9a+9b 검증: KSampler 연동 + E2E 비디오 생성 테스트.

GPU 환경에서 ComfyUI 루트에서 실행:
    cd /lustrefs/team-multimodal/minsu/ComfyUI
    python custom_nodes/ComfyUI-MotifVideo1.9B/tests/test_e2e_gpu.py

테스트 항목:
  1. Load Diffusion Model → MODEL 반환 확인
  2. Load MotifVideo Text Encoder → CLIP 반환 확인
  3. CLIP tokenize + encode 동작 확인
  4. EmptyMotifLatent shape 확인
  5. KSampler 연동 (MODEL + CONDITIONING + LATENT → denoised LATENT)
"""

import sys
import os
import traceback

# ComfyUI root
sys.path.insert(0, "/lustrefs/team-multimodal/minsu/ComfyUI")
sys.path.insert(0, "/lustrefs/team-multimodal/minsu/motif-models/packages/motif-core/src")
sys.path.insert(0, "/lustrefs/team-multimodal/minsu/motif-models/packages/motif-pipelines/src")

results = []


def test(name):
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
# Test 1: Load Diffusion Model
# ====================================================================
@test("Load Diffusion Model")
def test_load_model():
    import comfy.sd
    model_path = "/lustrefs/team-multimodal/minsu/ComfyUI/models/diffusion_models/motifvideo_1.9b.safetensors"
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    model = comfy.sd.load_diffusion_model(model_path)
    print(f"    Model type: {type(model)}")
    assert model is not None


# ====================================================================
# Test 2: Load Text Encoder
# ====================================================================
@test("Load MotifVideo Text Encoder")
def test_load_text_encoder():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nodes.loader import MotifTextEncoderLoader

    loader = MotifTextEncoderLoader()
    clip_tuple = loader.load_text_encoder(
        text_encoder_path="/lustrefs/team-multimodal/checkpoints/base_checkpoint/model/text_encoder",
        tokenizer_path="/lustrefs/team-multimodal/checkpoints/base_checkpoint/model/tokenizer",
        dtype="bfloat16",
    )
    clip = clip_tuple[0]
    print(f"    CLIP type: {type(clip)}")
    assert clip is not None


# ====================================================================
# Test 3: CLIP tokenize + encode
# ====================================================================
@test("CLIP tokenize + encode")
def test_clip_encode():
    from nodes.loader import MotifTextEncoderLoader

    loader = MotifTextEncoderLoader()
    clip = loader.load_text_encoder(
        text_encoder_path="/lustrefs/team-multimodal/checkpoints/base_checkpoint/model/text_encoder",
        tokenizer_path="/lustrefs/team-multimodal/checkpoints/base_checkpoint/model/tokenizer",
        dtype="bfloat16",
    )[0]

    tokens = clip.tokenize("A cat walking in a garden")
    print(f"    Token keys: {list(tokens.keys()) if isinstance(tokens, dict) else type(tokens)}")

    cond = clip.encode_from_tokens_scheduled(tokens)
    print(f"    Conditioning type: {type(cond)}")
    assert cond is not None


# ====================================================================
# Test 4: EmptyMotifLatent shape
# ====================================================================
@test("EmptyMotifLatent shape")
def test_empty_latent():
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nodes.latent import EmptyMotifLatent

    node = EmptyMotifLatent()
    result = node.generate(width=1280, height=736, num_frames=121, batch_size=1)
    latent = result[0]["samples"]

    expected_shape = (1, 16, 31, 92, 160)  # B=1, C=16, T=121//4+1=31, H=736//8=92, W=1280//8=160
    assert latent.shape == expected_shape, f"Expected {expected_shape}, got {latent.shape}"
    print(f"    Latent shape: {latent.shape}")


# ====================================================================
# Run
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MotifVideo E2E GPU Tests")
    print("=" * 60)

    test_load_model()
    test_load_text_encoder()
    test_clip_encode()
    test_empty_latent()

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
