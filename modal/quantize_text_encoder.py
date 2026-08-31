"""Build a blockwise-FP8 MiniMax-H3 text encoder so it can stay RESIDENT on 80 GB.

Why this exists
---------------
On 8xH100 80GB the pipeline is 15.62 s against a 14.375 s budget, and the entire 1.24 s
deficit is one stage: the Qwen3-VL text encoder is too large to keep in HBM, so it is
CPU-offloaded and costs 1.596 s per request instead of 0.038 s resident.

At bf16 the (already 50-of-64-layer-truncated) encoder is ~52 GB. Blockwise FP8 halves
that to ~26 GB, which fits beside an FSDP-sharded DiT and the VAE with room to spare.

Format
------
FastVideo loads, but does not create, quantized text encoders. `MiniMaxH3SerializedFP8Config`
requires exactly the DeepSeek-V3 blockwise convention:

    weight            float8_e4m3fn      [N, K]
    weight_scale_inv  float32            [ceil(N/128), ceil(K/128)]   dequant multiplier
    input_scale       absent             (activation_scheme = "dynamic")

    quantization_config = {quant_method: "fp8", fmt: "e4m3",
                           activation_scheme: "dynamic",
                           weight_block_size: [128, 128],
                           modules_to_not_convert: [...vision stack...]}

The vision stack must be excluded (H3 is text-to-video; the tower is small anyway) and
the language stack must be quantized in full -- partial language quantization is rejected.
"""
import json
import os

import modal

app = modal.App("h3-quantize-te")
vol = modal.Volume.from_name("h3-weights")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("safetensors", "numpy"))

BLOCK = 128
E4M3_MAX = 448.0


@app.function(image=image, gpu="H100", volumes={"/weights": vol}, timeout=5400,
              cpu=16, memory=131072)
def quantize(src: str = "/weights/fasth3-syn1300-full/text_encoder",
             dst: str = "/weights/te-fp8-syn1300"):
    import glob
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(dst, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {src}")

    def blockwise_fp8(w: torch.Tensor):
        """[N,K] bf16 -> (fp8 [N,K], float32 scale [ceil(N/128), ceil(K/128)])."""
        n, k = w.shape
        pn = (BLOCK - n % BLOCK) % BLOCK
        pk = (BLOCK - k % BLOCK) % BLOCK
        wp = torch.nn.functional.pad(w, (0, pk, 0, pn)) if (pn or pk) else w
        N, K = wp.shape
        blocks = wp.view(N // BLOCK, BLOCK, K // BLOCK, BLOCK).permute(0, 2, 1, 3)
        amax = blocks.abs().amax(dim=(-2, -1)).clamp_(min=1e-12).float()
        scale = (amax / E4M3_MAX)                        # dequant multiplier
        q = (blocks.float() / scale[..., None, None]).clamp_(-E4M3_MAX, E4M3_MAX)
        q = q.permute(0, 2, 1, 3).reshape(N, K)[:n, :k].to(torch.float8_e4m3fn)
        return q.contiguous(), scale.contiguous()

    skipped, quantized, not_convert = [], 0, set()
    dev = "cuda"
    total_in = total_out = 0
    for si, shard in enumerate(shards):
        out = {}
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                total_in += t.numel() * t.element_size()
                is_vision = ".visual." in key or key.startswith("visual.")
                if is_vision:
                    # record the module prefix so it lands in modules_to_not_convert
                    # Must stay scoped to the vision tower. Emitting a bare top-level
                    # namespace (e.g. "model") would land in modules_to_not_convert and
                    # silently exclude the ENTIRE encoder from quantization.
                    pref = (key.split(".weight")[0].rsplit(".", 1)[0]
                            if key.endswith(".weight") else key.rsplit(".", 1)[0])
                    if "visual" in pref:
                        not_convert.add(pref)
                if (key.endswith(".weight") and t.ndim == 2 and not is_vision
                        and t.dtype in (torch.bfloat16, torch.float16, torch.float32)
                        and "embed" not in key and "lm_head" not in key):
                    q, s = blockwise_fp8(t.to(dev, torch.bfloat16))
                    out[key] = q.cpu()
                    out[key[:-len("weight")] + "weight_scale_inv"] = s.cpu()
                    quantized += 1
                else:
                    out[key] = t
                    skipped.append(key)
        for v in out.values():
            total_out += v.numel() * v.element_size()
        save_file(out, os.path.join(dst, os.path.basename(shard)),
                  metadata={"format": "pt"})
        print(f"  shard {si+1}/{len(shards)}: {quantized} quantized so far")

    # carry over config/tokenizer/etc, then stamp the quantization_config
    import shutil
    for f in os.listdir(src):
        if not f.endswith(".safetensors"):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    cfg_path = os.path.join(dst, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [BLOCK, BLOCK],
        "modules_to_not_convert": sorted(not_convert) or ["visual"],
    }
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    vol.commit()
    res = {"quantized_linears": quantized, "skipped": len(skipped),
           "in_gb": round(total_in / 1e9, 1), "out_gb": round(total_out / 1e9, 1),
           "modules_to_not_convert": sorted(not_convert)[:6], "dst": dst}
    print(json.dumps(res, indent=1))
    return res


@app.local_entrypoint()
def main():
    print(json.dumps(quantize.remote(), indent=1))
