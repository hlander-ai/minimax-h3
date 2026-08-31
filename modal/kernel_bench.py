"""
The decisive kernel experiment.

Our compute-budget model says attention is 66% of dense DiT FLOPs, and the measured
SDPA-FlashAttention rate on this node is only ~353 TFLOPS (35% of peak) versus ~711
TFLOPS for GEMMs. So attention efficiency is THE lever. This benchmarks, at the exact
production shape, every attention path available on SM90a:

  1. torch SDPA (flash backend)          -- the baseline we already measured
  2. FlashAttention-3 (Hopper)           -- if installable; claims ~1.5-2x SDPA on Hopper
  3. block_sparse_attn_sm90  (TK CUDA)   -- hand-tuned Hopper block-sparse, NOT currently
                                            reachable from the H3 backend (it routes to Triton)
  4. block_sparse_attn_256_bshd (Triton) -- what FastH3 actually uses on non-Blackwell

Finding (3) faster than (4) would mean a one-line routing change buys real latency,
which we then spend on quality (more steps / lower sparsity).

Run: modal run modal/kernel_bench.py
"""
import json
import modal

app = modal.App("h3-kernel-bench")

CUDA_TAG = "12.8.1-devel-ubuntu22.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("git", "build-essential", "cmake", "ninja-build", "wget")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "packaging", "wheel", "setuptools", "scikit-build-core", "pybind11",
                 "cmake>=3.31", "ninja", "setuptools_scm", "einops")
    .env({
        "TORCH_CUDA_ARCH_LIST": "9.0a",          # Hopper, with the 'a' suffix TK requires
        "FASTVIDEO_KERNEL_BUILD_TK": "ON",       # force-build the ThunderKittens Hopper kernels
        "MAX_JOBS": "32",
        "CC": "gcc",
        "CXX": "g++",
    })
    .run_commands(
        # Pinned: these results were produced against this upstream commit. FastVideo
        # moves daily and the patches in patches/ do source surgery on specific files,
        # so an unpinned clone will drift out from under them.
        "git clone --filter=blob:none https://github.com/hao-ai-lab/FastVideo.git /opt/FastVideo",
        "cd /opt/FastVideo && git checkout b2db0c0a137e",
        "cd /opt/FastVideo && git submodule update --init --recursive "
        "fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk",
        # Build the kernel package for Hopper. Keep going if it fails so we still get
        # the Triton + SDPA numbers rather than losing the whole run.
        "cd /opt/FastVideo/fastvideo-kernel && "
        "(CMAKE_ARGS='-DFASTVIDEO_KERNEL_BUILD_TK=ON' pip install --no-build-isolation -v . "
        " > /tmp/kernel_build.log 2>&1 && echo KERNEL_BUILD_OK) || echo KERNEL_BUILD_FAILED",
        "tail -40 /tmp/kernel_build.log || true",
        "pip install triton==3.4.0 || true",
    )
)


@app.function(image=image, gpu="H100:1", timeout=3600)
def bench():
    import time
    import torch

    out = {}
    p = torch.cuda.get_device_properties(0)
    out["device"] = p.name
    out["capability"] = f"{p.major}.{p.minor}"

    # ---- what actually got built? -------------------------------------
    avail = {}
    try:
        import fastvideo_kernel
        avail["fastvideo_kernel"] = True
        avail["dir"] = [x for x in dir(fastvideo_kernel) if not x.startswith("_")][:40]
    except Exception as e:
        avail["fastvideo_kernel"] = f"import failed: {e}"
    try:
        import fastvideo_kernel_ops as ops
        avail["ops_symbols"] = [x for x in dir(ops) if "sparse" in x.lower() or "attn" in x.lower()]
    except Exception as e:
        avail["ops_symbols"] = f"import failed: {e}"
    try:
        from fastvideo_kernel.block_sparse_attn import _is_sm90, _get_sm90_ops
        fwd, bwd = _get_sm90_ops()
        avail["_is_sm90"] = _is_sm90()
        avail["sm90_fwd_available"] = fwd is not None
        avail["sm90_bwd_available"] = bwd is not None
    except Exception as e:
        avail["sm90_probe"] = f"failed: {e}"
    out["availability"] = avail

    try:
        with open("/tmp/kernel_build.log") as f:
            log = f.read()
        out["build_log_tail"] = log[-3000:]
        out["build_mentions_h100_kernel"] = "block_sparse_h100" in log
    except Exception:
        pass

    # ---- shapes ---------------------------------------------------------
    # Production: 90,720 video tokens, 56 heads, head_dim 128.
    # Under 8-way Ulysses each GPU owns 7 heads and the FULL sequence.
    SEQ = 90_720
    HEADS_PER_GPU = 7
    D = 128

    def timeit(fn, warmup=2, iters=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts)//2]

    dev = "cuda"
    dt = torch.bfloat16
    dense_flops = 4 * SEQ * SEQ * HEADS_PER_GPU * D

    results = []

    # 1. SDPA flash
    try:
        q = torch.randn(1, HEADS_PER_GPU, SEQ, D, device=dev, dtype=dt)
        k = torch.randn_like(q); v = torch.randn_like(q)
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
            t = timeit(lambda: sdpa(q, k, v))
        results.append({"method": "sdpa_flash", "sparsity": 0.0, "ms": t*1e3,
                        "tflops": dense_flops/t/1e12})
        del q, k, v; torch.cuda.empty_cache()
    except Exception as e:
        results.append({"method": "sdpa_flash", "error": str(e)[:200]})

    # 2. cuDNN backend (another Hopper-optimized dense option)
    try:
        q = torch.randn(1, HEADS_PER_GPU, SEQ, D, device=dev, dtype=dt)
        k = torch.randn_like(q); v = torch.randn_like(q)
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
            t = timeit(lambda: sdpa(q, k, v))
        results.append({"method": "sdpa_cudnn", "sparsity": 0.0, "ms": t*1e3,
                        "tflops": dense_flops/t/1e12})
        del q, k, v; torch.cuda.empty_cache()
    except Exception as e:
        results.append({"method": "sdpa_cudnn", "error": str(e)[:200]})

    # 3. TK sm90 block-sparse, swept over sparsity
    #    Layout for block_sparse_attn_sm90 is BHSD with 64-token blocks.
    BLK = 64
    nblk = SEQ // BLK
    for sparsity in (0.0, 0.25, 0.5, 0.7, 0.8, 0.9):
        try:
            from fastvideo_kernel.block_sparse_attn import (
                block_sparse_attn_sm90, _get_sm90_ops)
            fwd, _ = _get_sm90_ops()
            if fwd is None:
                results.append({"method": "tk_sm90", "sparsity": sparsity,
                                "error": "sm90 ops not built"})
                continue
            keep = max(1, int(round(nblk * (1.0 - sparsity))))
            q = torch.randn(1, HEADS_PER_GPU, nblk*BLK, D, device=dev, dtype=dt)
            k = torch.randn_like(q); v = torch.randn_like(q)
            # each query block attends to `keep` key blocks
            q2k_idx = torch.randint(0, nblk, (1, HEADS_PER_GPU, nblk, keep),
                                    device=dev, dtype=torch.int32)
            q2k_num = torch.full((1, HEADS_PER_GPU, nblk), keep,
                                 device=dev, dtype=torch.int32)
            vbs = torch.full((nblk,), BLK, device=dev, dtype=torch.int32)
            t = timeit(lambda: block_sparse_attn_sm90(q, k, v, q2k_idx, q2k_num, vbs))
            eff = dense_flops * (1.0 - sparsity)
            results.append({"method": "tk_sm90", "sparsity": sparsity, "ms": t*1e3,
                            "eff_tflops": eff/t/1e12,
                            "speedup_vs_dense_sdpa": None})
            del q, k, v, q2k_idx, q2k_num; torch.cuda.empty_cache()
        except Exception as e:
            results.append({"method": "tk_sm90", "sparsity": sparsity, "error": str(e)[:300]})
            torch.cuda.empty_cache()

    # 4. Triton block-sparse (the path FastH3 actually takes on Hopper)
    for sparsity in (0.5, 0.8, 0.9):
        try:
            from fastvideo_kernel.block_sparse_attn import block_sparse_attn_triton
            keep = max(1, int(round(nblk * (1.0 - sparsity))))
            q = torch.randn(1, HEADS_PER_GPU, nblk*BLK, D, device=dev, dtype=dt)
            k = torch.randn_like(q); v = torch.randn_like(q)
            q2k_idx = torch.randint(0, nblk, (1, HEADS_PER_GPU, nblk, keep),
                                    device=dev, dtype=torch.int32)
            q2k_num = torch.full((1, HEADS_PER_GPU, nblk), keep,
                                 device=dev, dtype=torch.int32)
            vbs = torch.full((nblk,), BLK, device=dev, dtype=torch.int32)
            t = timeit(lambda: block_sparse_attn_triton(q, k, v, q2k_idx, q2k_num, vbs))
            eff = dense_flops * (1.0 - sparsity)
            results.append({"method": "triton_bs", "sparsity": sparsity, "ms": t*1e3,
                            "eff_tflops": eff/t/1e12})
            del q, k, v, q2k_idx, q2k_num; torch.cuda.empty_cache()
        except Exception as e:
            results.append({"method": "triton_bs", "sparsity": sparsity, "error": str(e)[:300]})
            torch.cuda.empty_cache()

    out["results"] = results
    print(json.dumps(out, indent=1)[:20000])
    return out


@app.local_entrypoint()
def main():
    import os
    r = bench.remote()
    os.makedirs("results", exist_ok=True)
    with open("results/kernel_bench.json", "w") as f:
        json.dump(r, f, indent=1)
    print("WROTE results/kernel_bench.json")
