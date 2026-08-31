"""
Hardware ground-truth probe for the H3 realtime project.

Measures the things that determine whether <=15s is physically reachable on 8xH100,
independent of any claim in a paper or model card:

  1. Topology            - is this really an NVSwitch full-mesh node?
  2. BF16 / FP8 GEMM      - achievable TFLOPs at DiT-shaped matmuls => real MFU, not spec sheet
  3. HBM bandwidth        - the memory-bound floor for norms/residuals/elementwise
  4. NCCL collectives     - all-reduce / all-gather / reduce-scatter / ALL-TO-ALL.
                            all-to-all is the Ulysses sequence-parallel primitive and is the
                            most likely hidden bottleneck at video-DiT sequence lengths.
  5. Attention            - SDPA throughput swept across video-DiT sequence lengths
  6. Cold start           - container + CUDA init overhead

Run:  modal run modal/probe_hardware.py
"""
import json
import os
import time

import modal

app = modal.App("h3-hw-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "numpy",
        index_url="https://download.pytorch.org/whl/cu128",
        extra_index_url="https://pypi.org/simple",
    )
)

BOOT_T0 = time.time()


def _sync_time(fn, warmup=3, iters=10):
    """Time a CUDA fn with proper warmup + sync. Returns median seconds."""
    import torch

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
    return ts[len(ts) // 2]


@app.function(image=image, gpu="H100:8", timeout=3600)
def probe():
    import subprocess

    import torch

    out = {"import_to_start_s": round(time.time() - BOOT_T0, 2)}

    # ---------------------------------------------------------------- topology
    out["nvidia_smi_topo"] = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
    ).stdout
    out["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pcie.link.gen.max,clocks.max.sm",
         "--format=csv"], capture_output=True, text=True
    ).stdout
    out["torch_version"] = torch.__version__
    out["cuda_version"] = torch.version.cuda
    out["device_count"] = torch.cuda.device_count()
    p = torch.cuda.get_device_properties(0)
    out["device_name"] = p.name
    out["sm_count"] = p.multi_processor_count
    out["capability"] = f"{p.major}.{p.minor}"
    out["hbm_gb"] = round(p.total_memory / 1e9, 1)
    try:
        out["nvlink_status"] = subprocess.run(
            ["nvidia-smi", "nvlink", "-s"], capture_output=True, text=True
        ).stdout[:4000]
    except Exception as e:
        out["nvlink_status"] = f"err: {e}"

    t_cuda0 = time.perf_counter()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    out["cuda_init_s"] = round(time.perf_counter() - t_cuda0, 2)

    dev = torch.device("cuda:0")

    # ------------------------------------------------------------- BF16 GEMM
    # Shapes chosen to look like a 7168-hidden DiT: qkv proj, out proj, MLP up/down.
    HID = 7168
    gemms = []
    for (m, k, n, tag) in [
        (32768, HID, HID * 3, "qkv_proj"),
        (32768, HID, HID, "out_proj"),
        (32768, HID, HID * 4, "mlp_up"),
        (32768, HID * 4, HID, "mlp_down"),
        (16384, 8192, 8192, "square_8k"),
        (65536, 8192, 8192, "tall_8k"),
    ]:
        try:
            a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            b = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
            t = _sync_time(lambda: torch.mm(a, b))
            tflops = (2 * m * k * n) / t / 1e12
            gemms.append({"tag": tag, "m": m, "k": k, "n": n,
                          "ms": round(t * 1e3, 3), "tflops": round(tflops, 1)})
            del a, b
            torch.cuda.empty_cache()
        except Exception as e:
            gemms.append({"tag": tag, "error": str(e)[:200]})
    out["bf16_gemm"] = gemms
    peak = max((g.get("tflops", 0) for g in gemms), default=0)
    out["bf16_peak_tflops_measured"] = peak
    # H100 SXM dense BF16 spec = 989.5 TFLOPs
    out["bf16_mfu_vs_spec"] = round(peak / 989.5, 3) if peak else None

    # -------------------------------------------------------------- FP8 GEMM
    fp8 = []
    try:
        for (m, k, n, tag) in [(32768, HID, HID * 3, "qkv_proj"), (32768, HID * 4, HID, "mlp_down")]:
            a = torch.randn(m, k, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            b = torch.randn(n, k, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
            sa = torch.tensor(1.0, device=dev)
            sb = torch.tensor(1.0, device=dev)
            t = _sync_time(lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb,
                                                    out_dtype=torch.bfloat16))
            fp8.append({"tag": tag, "ms": round(t * 1e3, 3),
                        "tflops": round((2 * m * k * n) / t / 1e12, 1)})
            del a, b
            torch.cuda.empty_cache()
    except Exception as e:
        fp8.append({"error": str(e)[:300]})
    out["fp8_gemm"] = fp8
    fp8_peak = max((g.get("tflops", 0) for g in fp8), default=0)
    out["fp8_speedup_vs_bf16"] = round(fp8_peak / peak, 2) if (peak and fp8_peak) else None

    # --------------------------------------------------------- HBM bandwidth
    try:
        n_el = 1 << 28  # 256M bf16 = 512MB
        src = torch.randn(n_el, device=dev, dtype=torch.bfloat16)
        dst = torch.empty_like(src)
        t = _sync_time(lambda: dst.copy_(src))
        out["hbm_copy_gbps"] = round(2 * src.numel() * 2 / t / 1e9, 1)  # r+w
        t2 = _sync_time(lambda: src.mul(1.0001))
        out["hbm_elementwise_gbps"] = round(2 * src.numel() * 2 / t2 / 1e9, 1)
        del src, dst
        torch.cuda.empty_cache()
    except Exception as e:
        out["hbm_error"] = str(e)[:200]

    # ------------------------------------------------------------- attention
    # Sweep video-DiT sequence lengths. heads=7 is what each GPU owns under
    # 8-way Ulysses if the model really has 56 heads.
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    attn = []
    for seq in [16384, 32768, 65536, 131072, 262144, 393216]:
        for heads in [7, 56]:
            try:
                d = 128
                q = torch.randn(1, heads, seq, d, device=dev, dtype=torch.bfloat16)
                k_ = torch.randn(1, heads, seq, d, device=dev, dtype=torch.bfloat16)
                v = torch.randn(1, heads, seq, d, device=dev, dtype=torch.bfloat16)
                with torch.nn.attention.sdpa_kernel(
                    [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
                ):
                    t = _sync_time(lambda: sdpa(q, k_, v), warmup=2, iters=5)
                flops = 4 * seq * seq * heads * d
                attn.append({"seq": seq, "heads": heads, "ms": round(t * 1e3, 2),
                             "tflops": round(flops / t / 1e12, 1)})
                del q, k_, v
                torch.cuda.empty_cache()
            except Exception as e:
                attn.append({"seq": seq, "heads": heads, "error": str(e)[:120]})
                torch.cuda.empty_cache()
    out["attention_sweep"] = attn

    # ----------------------------------------------------------------- NCCL
    out["nccl"] = _run_nccl()

    print("=" * 100)
    print(json.dumps(out, indent=1))
    print("=" * 100)
    return out


def _run_nccl():
    """Spawn 8 procs and benchmark the collectives a DiT actually uses."""
    import torch.multiprocessing as mp

    mgr = mp.Manager()
    res = mgr.dict()
    try:
        mp.spawn(_nccl_worker, args=(8, res), nprocs=8, join=True)
        return dict(res)
    except Exception as e:
        return {"error": str(e)[:500]}


def _nccl_worker(rank, world, res):
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    def bench(fn, warmup=5, iters=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    local = {}
    # Message sizes bracket a real Ulysses all-to-all payload:
    # (seq/8) x hidden x 2B  ~= 650 MB at seq=363k, hidden=7168
    for mb in [16, 64, 256, 512, 1024]:
        n_el = mb * 1024 * 1024 // 2  # bf16
        try:
            x = torch.randn(n_el, device=dev, dtype=torch.bfloat16)

            t = bench(lambda: dist.all_reduce(x))
            # ring all-reduce moves 2*(N-1)/N * bytes
            local[f"all_reduce_{mb}MB"] = {
                "ms": round(t * 1e3, 3),
                "algo_gbps": round(2 * (world - 1) / world * mb / 1024 / t, 1),
            }

            shard = torch.randn(n_el // world, device=dev, dtype=torch.bfloat16)
            gathered = torch.empty(n_el, device=dev, dtype=torch.bfloat16)
            t = bench(lambda: dist.all_gather_into_tensor(gathered, shard))
            local[f"all_gather_{mb}MB"] = {"ms": round(t * 1e3, 3)}

            t = bench(lambda: dist.reduce_scatter_tensor(shard, x))
            local[f"reduce_scatter_{mb}MB"] = {"ms": round(t * 1e3, 3)}

            # all-to-all: THE Ulysses primitive
            a2a_in = torch.randn(n_el, device=dev, dtype=torch.bfloat16)
            a2a_out = torch.empty_like(a2a_in)
            t = bench(lambda: dist.all_to_all_single(a2a_out, a2a_in))
            local[f"all_to_all_{mb}MB"] = {
                "ms": round(t * 1e3, 3),
                "eff_gbps": round((world - 1) / world * mb / 1024 / t, 1),
            }
            del x, shard, gathered, a2a_in, a2a_out
            torch.cuda.empty_cache()
        except Exception as e:
            local[f"size_{mb}MB"] = {"error": str(e)[:150]}
            torch.cuda.empty_cache()

    # p2p bandwidth rank0 <-> rank1
    if world > 1:
        try:
            x = torch.randn(1 << 28, device=dev, dtype=torch.bfloat16)  # 512MB
            def p2p():
                if rank == 0:
                    dist.send(x, 1)
                elif rank == 1:
                    dist.recv(x, 0)
            t = bench(p2p, warmup=3, iters=10)
            if rank == 0:
                local["p2p_512MB"] = {"ms": round(t * 1e3, 3),
                                      "gbps": round(0.5 / t, 1)}
            del x
        except Exception as e:
            local["p2p"] = {"error": str(e)[:150]}

    if rank == 0:
        for k, v in local.items():
            res[k] = v
    dist.barrier()
    dist.destroy_process_group()


@app.local_entrypoint()
def main():
    t0 = time.time()
    r = probe.remote()
    r["_wallclock_including_coldstart_s"] = round(time.time() - t0, 1)
    os.makedirs("results", exist_ok=True)
    with open("results/hw_probe.json", "w") as f:
        json.dump(r, f, indent=1)
    print(f"\n\nWROTE results/hw_probe.json  (wallclock {r['_wallclock_including_coldstart_s']}s)")
