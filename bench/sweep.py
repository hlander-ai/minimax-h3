#!/usr/bin/env python3
"""
Pareto sweep driver.

Runs the (steps x sparsity x backend x precision) grid, records every point into
experiments.jsonl, and reports the frontier.

Design intent, straight from the mission:
  * We are NOT hunting for the fastest config. We are hunting for the HIGHEST-QUALITY
    config whose latency is <= 15 s. So the sweep deliberately spends time on the
    *dense* and *low-sparsity* end, which speed-oriented work skips.
  * Points are ordered cheapest-information-first: establish the reference and the
    two extremes, then fill in the interior only where the frontier is actually
    ambiguous. Blindly running the full cross product wastes GPU hours on configs
    the budget model already rules out.

Usage:
    python bench/sweep.py --plan            # print the plan, run nothing
    python bench/sweep.py --stage 1         # reference + extremes
    python bench/sweep.py --stage 2         # fill the interior
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PROFILE = os.environ.get("MODAL_PROFILE", "")

# The canonical target
FRAMES, STEPS_DEFAULT = 345, 5           # 5 sigma points = 4 DiT forwards

# Stage 1: the endpoints of the frontier. Everything else is interpolation.
# Both dense points go through TORCH_SDPA and differ ONLY in which kernel torch
# dispatches to. That makes the flash-vs-cuDNN comparison a controlled A/B on one
# code path, rather than comparing two different backend implementations.
STAGE1 = [
    ("fasth3_vsa90",  "VIDEO_SPARSE_ATTN_H3", "vsa-datafree",   5, 90, "", ""),
    ("dense_flash",   "TORCH_SDPA",           "dense-datafree", 5,  0, "flash", ""),
    ("dense_cudnn",   "TORCH_SDPA",           "dense-datafree", 5,  0, "cudnn", ""),
]

# Stage 2: adapter comparison at MATCHED settings. This is a zero-training test of the
# mission's explicit question -- "do NOT assume data-free training is optimal".
STAGE2 = [
    (f"adapter_{a}", "VIDEO_SPARSE_ATTN_H3", a, 5, 90, "", "")
    for a in ("vsa-datafree", "vsa-synthetic-step1300", "vsa-synthetic-step1900")
]

# Stage 3: how dense can we AFFORD to stay?
#
# The kernel benchmark makes this concrete and prunes most of the naive grid:
#   * Triton block-sparse holds ~355 TFLOPS effective at every sparsity (efficiency ~0.95)
#   * cuDNN dense holds 597 TFLOPS, so sparse only wins above 40.5% sparsity
#   * below 40% sparsity VSA is STRICTLY SLOWER than dense -- that region is dominated
#     and is not worth a single GPU-minute
#   * the densest that fits 14.375s at 4 forwards is ~64% (bf16)
# So the informative sweep is 60-90%, bracketing the predicted 64% frontier edge,
# plus the dense cuDNN point as the (over-budget) quality ceiling reference.
STAGE3 = [
    (f"vsa_sp{sp}", "VIDEO_SPARSE_ATTN_H3", "vsa-datafree", 5, sp, "", "")
    for sp in (90, 80, 70, 65, 60)
]

# Stage 4: precision. FP8 changes numerics, so it must clear the quality gate.
STAGE4 = [
    ("dense_cudnn_fp8", "TORCH_SDPA", "dense-datafree", 5, 0, "cudnn", "FP8"),
    ("vsa90_fp8",       "VIDEO_SPARSE_ATTN_H3", "vsa-datafree", 5, 90, "", "FP8"),
]

# Stage 5: extra forwards, OFF-DISTRIBUTION (only 4-step adapters exist; DMD bakes its
# sigma grid in). Measured because if quality holds it dominates every 4-forward point.
STAGE5 = [
    (f"dense_steps{s}", "TORCH_SDPA", "dense-datafree", s, 0, "cudnn", "")
    for s in (6, 7)
] + [
    (f"vsa70_steps{s}", "VIDEO_SPARSE_ATTN_H3", "vsa-datafree", s, 70, "", "")
    for s in (6, 7, 9)
]

PROMPTS = {
    "parkour": "A traceur sprints across a rooftop and vaults a concrete ledge, tucking into a "
               "roll and rising into a run. Fast tracking shot follows alongside. Late afternoon "
               "sun, long shadows, dust kicked up on landing, city skyline behind.",
    "occlusion": "A red enamel mug sits on a wooden table. A person in a grey sweater walks "
                 "between the camera and the table, fully hiding the mug for about a second, then "
                 "continues out of frame, revealing the mug exactly where it was. Static camera.",
    "texture": "Macro shot of sunlight moving across a woven wool blanket as a cloud passes. "
               "Individual fibers and slubs visible, the weave pattern crisp.",
}


def run_point(tag, backend, lora, steps, sparsity, sdpa, quant, prompt_key, repeats=3, gpus=8):
    prompt = PROMPTS[prompt_key]
    cmd = [
        "modal", "run", "modal/h3_bench.py",
        "--frames", str(FRAMES), "--steps", str(steps),
        "--backend", backend, "--lora", lora, "--gpus", str(gpus),
        "--sparsity", str(sparsity), "--repeats", str(repeats),
        "--tag", f"{tag}__{prompt_key}", "--prompt", prompt,
    ]
    if sdpa:
        cmd += ["--sdpa-kernel", sdpa]
    if quant:
        cmd += ["--quant", quant]
    env = dict(os.environ, MODAL_PROFILE=PROFILE)
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log = f"results/logs/sweep_{tag}__{prompt_key}.log"
    os.makedirs("results/logs", exist_ok=True)
    with open(log, "w") as f:
        f.write(p.stdout + "\n===STDERR===\n" + p.stderr)
    ok = p.returncode == 0
    print(f"  [{'ok' if ok else 'FAIL'}] {tag}/{prompt_key} in {time.time()-t0:.0f}s -> {log}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--prompts", default="parkour,occlusion,texture")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    stage = {1: STAGE1, 2: STAGE2, 3: STAGE3, 4: STAGE4, 5: STAGE5}[a.stage]
    keys = a.prompts.split(",")

    if a.plan:
        print(f"Stage {a.stage}: {len(stage)} configs x {len(keys)} prompts "
              f"= {len(stage)*len(keys)} runs")
        for t, b, l, st, sp, sd, q in stage:
            print(f"  {t:22s} {b:22s} lora={l or '(base)':22s} "
                  f"{st-1} fwd  sp={sp:3d}%  sdpa={sd or 'default':7s} prec={q or 'bf16'}")
        return

    for t, b, l, st, sp, sd, q in stage:
        for k in keys:
            run_point(t, b, l, st, sp, sd, q, k, repeats=a.repeats)


if __name__ == "__main__":
    main()
