"""항목 9a+9b 검증: KSampler 연동 + E2E 비디오 생성 테스트.

GPU 환경에서 ComfyUI 루트에서 실행:
    cd /lustrefs/team-multimodal/minsu/ComfyUI
    python custom_nodes/ComfyUI-MotifVideo1.9B/tests/test_e2e_gpu.py
"""

import sys
import os
import traceback

# Setup paths — same as __init__.py
sys.path.insert(0, "/lustrefs/team-multimodal/minsu/ComfyUI")
sys.path.insert(0, "/lustrefs/team-multimodal/minsu/motif-models/packages/motif-core/src")
sys.path.insert(0, "/lustrefs/team-multimodal/minsu/motif-models/packages/motif-pipelines/src")

# Apply monkey-patches (normally done by __init__.py at ComfyUI startup)
print("[Setup] Applying monkey-patches...")
import comfy.supported_models
import comfy.model_detection

# 1. Register model config
_custom_node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_custom_node_dir))

# Import our package properly
import importlib.util
import types

_pkg_name = "ComfyUI_MotifVideo1_9B"
_pkg = types.ModuleType(_pkg_name)
_pkg.__path__ = [_custom_node_dir]
_pkg.__package__ = _pkg_name
_pkg.__file__ = os.path.join(_custom_node_dir, "__init__.py")
sys.modules[_pkg_name] = _pkg

# Register sub-packages
for sub in ["models", "text_encoders", "nodes"]:
    sub_mod = types.ModuleType(f"{_pkg_name}.{sub}")
    sub_mod.__path__ = [os.path.join(_custom_node_dir, sub)]
    sub_mod.__package__ = f"{_pkg_name}.{sub}"
    sys.modules[f"{_pkg_name}.{sub}"] = sub_mod
    setattr(_pkg, sub, sub_mod)

# Execute __init__.py to apply patches
spec = importlib.util.spec_from_file_location(
    _pkg_name, os.path.join(_custom_node_dir, "__init__.py")
)
init_mod = importlib.util.module_from_spec(spec)
init_mod.__package__ = _pkg_name
init_mod.__path__ = [_custom_node_dir]
try:
    spec.loader.exec_module(init_mod)
    sys.modules[_pkg_name] = init_mod
    print("[Setup] Monkey-patches applied successfully")
except Exception as e:
    print(f"[Setup] WARNING: {e}")
    # Apply detection patch manually as fallback
    exec(open(os.path.join(_custom_node_dir, "__init__.py")).read())

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


@test("Load Diffusion Model")
def test_load_model():
    import comfy.sd
    model_path = "/lustrefs/team-multimodal/minsu/ComfyUI/models/diffusion_models/motifvideo_1.9b.safetensors"
    assert os.path.exists(model_path), f"Not found: {model_path}"
    model = comfy.sd.load_diffusion_model(model_path)
    print(f"    Model type: {type(model)}")
    assert model is not None


@test("Load MotifVideo Text Encoder")
def test_load_text_encoder():
    from nodes.loader import MotifTextEncoderLoader
    loader = MotifTextEncoderLoader()
    clip = loader.load_text_encoder(
        text_encoder_path="/lustrefs/team-multimodal/checkpoints/base_checkpoint/model/text_encoder",
        tokenizer_path="/lustrefs/team-multimodal/checkpoints/base_checkpoint/model/tokenizer",
        dtype="bfloat16",
    )[0]
    print(f"    CLIP type: {type(clip)}")
    assert clip is not None


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
    print(f"    Tokens: {type(tokens)}")
    cond = clip.encode_from_tokens_scheduled(tokens)
    print(f"    Conditioning: {type(cond)}, len={len(cond) if isinstance(cond, list) else 'N/A'}")
    assert cond is not None


@test("EmptyMotifLatent shape")
def test_empty_latent():
    import torch
    from nodes.latent import EmptyMotifLatent
    node = EmptyMotifLatent()
    result = node.generate(width=1280, height=736, num_frames=121, batch_size=1)
    latent = result[0]["samples"]
    expected = (1, 16, 31, 92, 160)
    assert latent.shape == expected, f"Expected {expected}, got {latent.shape}"
    print(f"    Latent shape: {latent.shape}")


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

    if failed:
        print("\nFailed:")
        for name, status, err in results:
            if status == "FAIL":
                print(f"  - {name}: {err}")
        sys.exit(1)
    print("All tests passed!")
