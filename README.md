# Sub-real-time MiniMax H3 on Hopper

A 14.375-second 768p video with native audio, generated in **13.506 s on 8×H100 80 GB** —
0.939× of real time. The published state of the art for this model is Blackwell-only. It
doesn't need to be, and **93% of the win came from outside the model.**

| | |
|---|---|
| **13.506 s** | warm end-to-end, 345 frames @ 1344×768 / 24 fps + audio |
| **0.939×** | of real time, on device-verified H100 80GB HBM3 |
| **1.690×** | over FastH3's own recipe — same node, checkpoint, steps, sparsity |
| **none** | quality claims standing (see below) |

Three warm runs: 13.503 / 13.506 / 13.554 s, σ = 0.03. Loading and compilation excluded,
matching FastVideo's reporting protocol.

## Read this first

**No quality claim here is established.** An earlier version of this work ranked adapters
with reference-based metrics against Base H3. That comparison is invalid: a 49-step dense
sampler and a 4-step DMD sampler map identical noise to *different trajectories*, so the
same seed yields a different video, not a degraded one. DMD2 trains the student to match
the teacher's distribution, never its per-sample output. Every claim resting on that —
adapter rankings, LPIPS/SSIM/PSNR vs base, the "motion energy preserved" figures — is
**retracted in full**.

Absolute metrics can't rescue it either: they're confounded by scene content, and the two
motion estimators disagree in *direction* (RAFT 0.756× vs Farneback 1.31× on the same
pair). The only structurally valid instrument used was CLIP text similarity, at n=1 prompt
— direction only.

Settling it needs distribution-level metrics (FVD/FAD over many samples) or blinded
pairwise preference. `eval/prompts/suite_v1.json` was built for that and has not been run.

## Where the time actually went

| Stage | FastH3 stock | optimized | Δ | what it was |
|---|---:|---:|---:|---|
| VAE decode | 7.00 s | 2.51 s | −4.49 | 10.4 GB CPU offload round-trip, per request |
| Text encode | 1.82 s | 0.04 s | −1.78 | 66.7 GB Qwen3-VL over PCIe, per request |
| Post-decode / encode / save | 3.09 s | 1.98 s | −1.11 | 4.27 GB fp32 frame buffer through IPC |
| Audio decode | 0.37 s | 0.12 s | −0.25 | fell out of residency |
| DiT denoising | 8.86 s | 8.60 s | −0.26 | FP8 arithmetic |

Total **−9.72 s** — measured on one 8×H200 node (23.810 → 14.085 s, 1.690×), same
checkpoint, same 4 forwards, same 90% sparsity. The 13.506 s headline is 8×H100 and adds
the worker-side encode patch; **the two figures are from different hardware and should not
be subtracted from each other.** Steps never changed; sparsity never changed. The model was never the slow part — at the halfway
point the DiT was already generating faster than real time while the pipeline around it
blew the budget.

The first three rows are costs every FastH3 user pays **on any hardware, Blackwell
included.**

## `patches/` — the part worth taking

Runtime patches against [FastVideo](https://github.com/hao-ai-lab/FastVideo), pinned to
upstream commit [`b2db0c0`](https://github.com/hao-ai-lab/FastVideo/commit/b2db0c0a137e).

**Three of the four are numerically lossless** — they change where and how bytes move,
never pixel values. The fourth changes text-encoder precision, and it is worth being
precise about which is which.

- **`patch_postdecode.py`** — the decode stage returns a CPU fp32 buffer (345×3×768×1344×4B
  = 4.27 GB) across the multiproc-executor boundary, and the receiver's next act is to cast
  it to uint8. Cast in the worker instead: 1.07 GB, not 4.27 GB.
- **`patch_worker_encode.py`** — encode the mp4 inside the worker so those pixels never
  cross IPC at all. The main process only ever needed them to write a ~15 MB file.
- **`patch_parallel_encode.py`** — segment-parallel x264 in **threads**, not processes.
  PyAV releases the GIL inside `encode()`; a per-call ProcessPoolExecutor spends more
  spawning interpreters than it saves. With `ultrafast` every segment starts on a keyframe,
  so stream-copy concat decodes bit-identically to a single pass.
- **`patch_fp8_te_sm90.py`** — **not numerically neutral.** FastVideo gates its
  blockwise-FP8 text encoder on Blackwell; only the GEMM is actually Blackwell-specific,
  while activation quantization is Triton and architecture-agnostic. On sm90 this
  dequantizes each weight to bf16 one layer at a time — an exact-arithmetic substitution
  for *FlashInfer's FP8 path*, but the weights still live as FP8 (66.7 → 35.5 GB, 448
  linears), so this is a precision change relative to the stock bf16 encoder. Halving the
  encoder is what makes full residency fit in 80 GB.

## Other findings

**On Hopper, "reduce the sparsity" is backwards.** VSA's fast path is a hand-written
`sm100a` kernel that doesn't compile for Hopper, so SM90a falls through to Triton at 355
TFLOPS against cuDNN dense attention's 597. Measured: every 10 points of density costs
2.24 s (fit `DiT = 6.44 + 22.45·(1−s)`). 90% sparsity isn't aggression — it's the only
operating point that fits the budget. That 1.68× per-FLOP penalty also gives a threshold:
**crossover at 1 − 355/597 = 40.5% sparsity**, below which VSA on Hopper loses outright to
plain dense attention.

**FP8 on the DiT returned −0.24 s against a predicted −2.2 s.** It genuinely uses
`torch._scaled_mm`; the gap is that FastVideo quantizes activations dynamically on every
call — ~1.1 GB read and 0.55 GB write per linear layer, ~200 times per forward, which a
static-scale microbenchmark never pays. Its real value was memory: halving the DiT let FSDP
go, worth −0.79 s.

**At 8-way parallelism the pipeline is no longer DiT-bound.** Fit `T(N) = D/N + S` to
FastVideo's published B200 scaling (47.2 / 15.5 / 12.88 s at 1 / 4 / 8 GPUs) and the
non-scaling term is ≈10.3 s of 12.88. Most of the remaining wall clock is serial work and
sequence-parallel communication, which doesn't care how fast the tensor cores are. That's
why an H100 node keeps pace.

**Predictions that missed**, all from the same error — an isolated kernel benchmark
*bounds* a component, it does not *predict* a pipeline:

| | predicted | actual |
|---|---:|---:|
| Dense 4-step fits the budget | 14.2 s | 17.3–19.5 s |
| Sparsity cost per 10 density points | 0.75 s | 2.24 s |
| FP8 on the DiT | −2.2 s | −0.24 s |
| Non-DiT pipeline overhead | ~2 s | 9.42 s |

## Layout

```
patches/            runtime patches against FastVideo    <- the contribution
src/h3lab/          metrics, budget accounting, experiment records
bench/              sweep harness
modal/              benchmark / hardware-probe / eval jobs
eval/prompts/       30-prompt evaluation suite (unrun)
results/            every measurement, as JSON
                    (bench/ also writes a local experiments.jsonl record)
```

`src/h3lab/metrics.py::classify()` is worth a look: it refuses to grade against a reference
the candidate was never meant to reproduce, and documents why. That guard was correct.
Overriding it is what produced the retracted claims above.

## Reproducing

```
checkpoint   FastVideo/FastH3-4-step-Preview-v1-VSA-Synthetic-Step1300  # merged, not LoRA
frames       345          # 14.375 s @ 24 fps — the maximum legal count
resolution   1344×768     audio  native stereo 32 kHz     seed 1000
forwards     4            # 5 sigma points, DMD2, guidance_scale 1.0
attention    VIDEO_SPARSE_ATTN_H3   sparsity 0.90   tile 64   # Triton on SM90a
precision    bf16 DiT, FSDP-sharded · blockwise-FP8 text encoder (sm90 patch)
residency    text encoder + VAE resident   # never offloaded
parallelism  ulysses sp=8
encode       worker-side, 4-thread segmented x264
upstream     hao-ai-lab/FastVideo @ b2db0c0a137e
```

The upstream pin matters: FastVideo moves daily and these patches do source surgery
on specific files. Runs were executed 2026-08-30 against `--depth 1` clones of main;
`b2db0c0` was HEAD for the bulk of that window, including the headline runs.

The checkpoint is **not** a recommendation — it's there because a now-retracted quality
argument put it there. It costs the same to run as FastVideo's default `vsa-datafree`.

Set `MODAL_PROFILE` to your own Modal workspace before running anything under `bench/`.

## What this is not

Hao AI Lab's [FastH3 preview](https://haoailab.com/blogs/fasth3-preview/) reports 12.88 s
on 8×B200 under the same protocol. **They are 4.9% faster than this work, on 2.3× the
silicon per GPU**, with FlashAttention-4 and a sparse kernel that doesn't exist on Hopper.
This is not a speedup over them and should never be cited as one.

The narrower claim is the useful one: their published numbers are all Blackwell, the
natural reading is that sub-real-time H3 requires Blackwell, and that inference is false.

## License and scope

Code is Apache-2.0. Model weights and generated media are deliberately absent — see
[NOTICE](NOTICE). MiniMax H3 and its derivatives (including the FastH3 checkpoints, which
declare `base_model: MiniMaxAI/MiniMax-H3` and `license_name: minimax-h3-community`) carry
a license whose grant excludes several territories and extends that restriction to model
outputs. This repository ships measurements, code and analysis only.
