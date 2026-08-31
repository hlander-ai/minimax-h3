"""
Compute-budget model for MiniMax H3 on an 8-GPU Hopper node.

Everything here is derived from the ACTUAL configs pulled from HuggingFace:
  transformer/config.json : 50 layers, 56 heads x 128 dim, hidden 5376, ffn 14336, patch [1,2,2]
  vae/config.json         : spatial_downsample [2,2,2,2,1,1] = 16x, temporal [1,2,2,1,1,1] = 4x

The point of this module is to answer, BEFORE spending GPU hours:
  "How many DiT evaluations, at what attention sparsity, fit in 15 seconds?"

Re-run with measured MFU once the hardware probe lands to replace the assumed efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- architecture
NUM_LAYERS = 50
NUM_REFINER_LAYERS = 2
NUM_HEADS = 56
HEAD_DIM = 128
HIDDEN = 5376
FFN_DIM = 14336
PATCH = (1, 2, 2)          # t, h, w
VAE_SPATIAL = 16           # 2*2*2*2*1*1
VAE_TEMPORAL = 4           # 1*2*2*1*1*1
ATTN_DIM = NUM_HEADS * HEAD_DIM   # 7168 -- note: != HIDDEN, qkv projects 5376 -> 7168

# transformer/ is 66.3 GB of bf16 safetensors => ~33.2B params
DIT_PARAMS = 33.12e9   # confirmed at load: "Loaded model with 33.12B parameters"

# --------------------------------------------------------------------- hardware
@dataclass
class GPU:
    name: str
    bf16_tflops: float     # dense, no sparsity
    hbm_gb: float
    hbm_tbps: float

H100 = GPU("H100-SXM", 989.5, 80, 3.35)
H200 = GPU("H200-SXM", 989.5, 141, 4.8)     # same SM90a compute, more/faster memory
B200 = GPU("B200", 2250.0, 180, 8.0)        # for comparing against FastH3's published numbers


# H3's causal VAE packs frames in chunks of 17 producing 5 latents each, so num_frames
# must satisfy num_frames % 17 == 5. 360 aligns UP to 362 = 15.083 s, which the pipeline
# rejects against its 15 s ceiling. The true maximum is 345 frames = 14.375 s.
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
MAX_FRAMES = 345


def align_num_frames(n):
    while n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        n += 1
    return n


def video_latent_frames(n):
    return (n - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def latent_tokens(width=1344, height=768, frames=MAX_FRAMES):
    """Video tokens entering the DiT after VAE compression and patchification."""
    lat_w = width // VAE_SPATIAL
    lat_h = height // VAE_SPATIAL
    lat_t = video_latent_frames(align_num_frames(frames))
    tok_w = lat_w // PATCH[2]
    tok_h = lat_h // PATCH[1]
    tok_t = lat_t // PATCH[0]
    return {
        "latent_grid": (lat_t, lat_h, lat_w),
        "token_grid": (tok_t, tok_h, tok_w),
        "tokens": tok_t * tok_h * tok_w,
    }


def flops_per_forward(n_tokens: int, sparsity: float = 0.0):
    """FLOPs for one DiT forward.

    Linear ops scale O(N); attention scales O(N^2) and is what sparsity attacks.
    sparsity=0.9 means 10% of attention blocks are actually computed.
    """
    # Dense/linear: 2 * params * tokens is the standard estimate.
    linear = 2 * DIT_PARAMS * n_tokens

    # Self-attention: QK^T + PV = 4 * N^2 * heads * head_dim per layer.
    attn_per_layer = 4 * (n_tokens ** 2) * NUM_HEADS * HEAD_DIM
    attn = attn_per_layer * NUM_LAYERS * (1.0 - sparsity)

    return {"linear": linear, "attention": attn, "total": linear + attn}


def time_estimate(n_tokens, steps, sparsity, gpu: GPU, n_gpu=8, mfu=0.55,
                  sparse_efficiency=0.65):
    """Seconds of DiT compute.

    mfu                : fraction of peak achieved on the dense GEMMs
    sparse_efficiency  : block-sparse kernels never hit dense MFU. A 90%-sparse kernel
                         doing 10% of the FLOPs does NOT run 10x faster -- gather/scatter,
                         partial tiles, and lower arithmetic intensity eat into it.
                         0.65 is a deliberately non-heroic assumption.
    """
    f = flops_per_forward(n_tokens, sparsity)
    agg = gpu.bf16_tflops * 1e12 * n_gpu

    t_linear = f["linear"] / (agg * mfu)
    eff = mfu * (sparse_efficiency if sparsity > 0 else 1.0)
    t_attn = f["attention"] / (agg * eff)

    per_fwd = t_linear + t_attn
    return {
        "pflops_per_forward": f["total"] / 1e15,
        "attn_share": f["attention"] / f["total"],
        "s_per_forward": per_fwd,
        "s_linear": t_linear * steps,
        "s_attention": t_attn * steps,
        "dit_total_s": per_fwd * steps,
    }


def grid(gpu=H200, n_gpu=8, mfu=0.55, budget_s=MAX_FRAMES/24, overhead_s=3.0):
    """The Phase-5 grid: which (steps, sparsity) combinations fit the budget?

    overhead_s covers text encoding + VAE decode + audio + scheduler glue, i.e. the
    non-DiT part of E2E. Placeholder until measured.
    """
    tk = latent_tokens()["tokens"]
    rows = []
    for steps in (4, 5, 6, 8, 10, 12):
        for sp in (0.0, 0.25, 0.50, 0.70, 0.80, 0.90):
            e = time_estimate(tk, steps, sp, gpu, n_gpu, mfu)
            e2e = e["dit_total_s"] + overhead_s
            rows.append({
                "steps": steps, "sparsity": sp,
                "dit_s": e["dit_total_s"], "e2e_s": e2e,
                "rt": e2e / (MAX_FRAMES/24), "fits": e2e <= budget_s,
            })
    return tk, rows


def report(gpu=H200, n_gpu=8, mfu=0.55, overhead_s=3.0):
    lt = latent_tokens()
    tk = lt["tokens"]
    lines = []
    lines.append(f"Target 1344x768, {MAX_FRAMES} frames @24fps = {MAX_FRAMES/24:.3f} s of video "
                 f"(H3 requires frames%17==5 and <=15s, so 345 is the true max, not 360)")
    lines.append(f"  VAE latent grid (t,h,w) : {lt['latent_grid']}   ({VAE_SPATIAL}x spatial, {VAE_TEMPORAL}x temporal)")
    lines.append(f"  DiT token grid  (t,h,w) : {lt['token_grid']}   (patch {PATCH})")
    lines.append(f"  SEQUENCE LENGTH         : {tk:,} video tokens")
    lines.append("")
    f0 = flops_per_forward(tk, 0.0)
    lines.append(f"Per DiT forward @ dense:  linear {f0['linear']/1e15:6.2f} PF | "
                 f"attention {f0['attention']/1e15:6.2f} PF | total {f0['total']/1e15:6.2f} PF")
    lines.append(f"  attention is {100*f0['attention']/f0['total']:.0f}% of dense FLOPs "
                 f"-> sparsity is the dominant lever")
    lines.append("")
    lines.append(f"Hardware: {n_gpu}x {gpu.name} = {gpu.bf16_tflops*n_gpu/1e3:.1f} PFLOP/s peak bf16, "
                 f"assumed MFU {mfu:.0%}, non-DiT overhead {overhead_s:.1f}s")
    lines.append("")
    hdr = f"{'steps':>5} | " + " | ".join(f"{int(s*100):>3}% sp" for s in (0.0,0.25,0.50,0.70,0.80,0.90))
    lines.append(hdr)
    lines.append("-" * len(hdr))
    _, rows = grid(gpu, n_gpu, mfu, overhead_s=overhead_s)
    by = {}
    for r in rows:
        by.setdefault(r["steps"], []).append(r)
    for steps, rs in by.items():
        cells = []
        for r in rs:
            mark = "*" if r["fits"] else " "
            cells.append(f"{r['e2e_s']:5.1f}{mark}")
        lines.append(f"{steps:>5} | " + " | ".join(cells))
    lines.append("")
    lines.append("  cells = estimated E2E seconds; * = fits the 15s realtime budget")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
