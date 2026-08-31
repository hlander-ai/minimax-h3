"""
The main generation + benchmarking harness.

One Modal class per *boot-time configuration* (attention backend, compile flags, ...),
because FastVideo reads those from the environment before the model loads. Modal keeps a
warm container per distinct parameter set, which is exactly what we need: cold-start and
model-load costs are excluded from the measurement, matching how a served deployment behaves.

Each run does one discarded warmup request (compile + autotune) followed by N measured
requests at a fixed seed, and reports median E2E plus per-stage timings.
"""
import json
import os
import time

import modal

app = modal.App("h3-bench")
vol = modal.Volume.from_name("h3-weights", create_if_missing=True)
out_vol = modal.Volume.from_name("h3-outputs", create_if_missing=True)

CUDA_TAG = "12.8.1-devel-ubuntu22.04"
LOCAL_OUT = "/scratch/out"   # container-local disk, not the network volume

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("git", "build-essential", "cmake", "ninja-build", "wget", "ffmpeg")
    .pip_install("torch==2.8.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "packaging", "wheel", "setuptools", "scikit-build-core", "pybind11", "cmake>=3.31", "ninja", "setuptools_scm",
                 "einops", "imageio", "imageio-ffmpeg", "safetensors", "huggingface_hub[hf_transfer]")
    .env({
        "TORCH_CUDA_ARCH_LIST": "9.0a",
        "FASTVIDEO_KERNEL_BUILD_TK": "ON",
        "MAX_JOBS": "32",
        "CC": "gcc",
        "CXX": "g++",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/weights/.hf",
    })
    .run_commands(
        # Pinned: these results were produced against this upstream commit. FastVideo
        # moves daily and the patches in patches/ do source surgery on specific files,
        # so an unpinned clone will drift out from under them.
        "git clone --filter=blob:none https://github.com/hao-ai-lab/FastVideo.git /opt/FastVideo",
        "cd /opt/FastVideo && git checkout b2db0c0a137e",
        "cd /opt/FastVideo && git submodule update --init --recursive "
        "fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk",
        "cd /opt/FastVideo/fastvideo-kernel && "
        "(CMAKE_ARGS='-DFASTVIDEO_KERNEL_BUILD_TK=ON' pip install --no-build-isolation -v . "
        " > /tmp/kbuild.log 2>&1 && echo KERNEL_OK) || (echo KERNEL_FAILED; tail -60 /tmp/kbuild.log)",
        # FastVideo itself, without letting it drag in a different torch
        "cd /opt/FastVideo && pip install --no-build-isolation -e . || pip install -e .",
        # FlashAttention-3 for Hopper: this is the single biggest dense-attention lever.
        "pip install 'flash-attn>=2.7' --no-build-isolation || echo FA2_FAILED",
    )
    # Added LAST so every earlier layer stays cached. Fails the build loudly if the
    # upstream anchor moved -- silently running unpatched would corrupt the A/B.
    .add_local_file("patches/patch_postdecode.py", "/opt/patch_postdecode.py", copy=True)
    .run_commands("python /opt/patch_postdecode.py")
    .add_local_file("patches/patch_fp8_te_sm90.py", "/opt/patch_fp8_te_sm90.py", copy=True)
    .run_commands("python /opt/patch_fp8_te_sm90.py")
    .add_local_file("patches/patch_parallel_encode.py", "/opt/patch_parallel_encode.py", copy=True)
    .run_commands("python /opt/patch_parallel_encode.py")
    .add_local_file("patches/patch_worker_encode.py", "/opt/patch_worker_encode.py", copy=True)
    .run_commands("python /opt/patch_worker_encode.py")
)


@app.cls(image=image, gpu="H100:8", volumes={"/weights": vol, "/outputs": out_vol},
         timeout=5400, scaledown_window=60, cpu=16, memory=131072)
class H3Runner:
    # Boot-time knobs. Modal keeps one warm container per distinct parameter set.
    backend: str = modal.parameter(default="VIDEO_SPARSE_ATTN_H3")   # or FLASH_ATTN
    sparsity_x100: int = modal.parameter(default=90)                  # VSA sparsity * 100
    tile: int = modal.parameter(default=64)
    compile_regional: int = modal.parameter(default=1)
    parallel_vae: int = modal.parameter(default=1)
    fusions: int = modal.parameter(default=1)
    num_gpus: int = modal.parameter(default=8)
    lora: str = modal.parameter(default="vsa-datafree")               # or dense-datafree, or "" for base
    # Full merged FastH3 checkpoint instead of base+adapter. Required for FP8, since
    # FastVideo's LoRA layer conversion and its quantized ReplicatedLinear are mutually
    # exclusive (AttributeError: no attribute 'weight'). Also removes the per-build cost
    # of patching 362 layers. FP8 additionally halves the DiT to ~35GB, so DiT+VAE+TE fit
    # resident at ~112GB WITHOUT FSDP -- avoiding the +0.8s all-gather FSDP costs.
    model_path: str = modal.parameter(default="/weights/base")
    # Force torch SDPA onto a specific kernel. Measured on H100 at seq=90,720:
    #   flash 373 TFLOPS vs cuDNN 596 TFLOPS -> 1.60x. Both compute EXACT attention,
    #   so this is a class-A (effectively lossless) change, not an approximation.
    sdpa_kernel: str = modal.parameter(default="")                    # "", "cudnn", "flash"
    # Transformer weight quantization. Measured FP8 e4m3 GEMM on this node is 1319
    # TFLOPS vs 712 bf16 (1.85x) -- but unlike the cuDNN swap this DOES change numerics,
    # so it must clear the quality gate before it is allowed onto the frontier.
    quant: str = modal.parameter(default="")                          # "", "FP8", "AbsMaxFP8"
    # Quantize the TEXT ENCODER separately from the DiT. On an 80GB part the 66.7GB
    # Qwen3-VL cannot stay resident, and offloading it costs 1.596s of the 1.24s deficit.
    # At FP8 it is ~33GB, which fits beside an FSDP-sharded DiT (52GB total, 28GB free).
    # NOTE: this changes prompt-embedding numerics, so it must clear the quality gate.
    te_quant: str = modal.parameter(default="")
    te_path: str = modal.parameter(default="")   # pre-quantized text encoder directory
    # Residency. Measured: the VAE is shuttled host<->device EVERY request
    # (self.vae.to(device) ... finally self.vae.to("cpu")) and the 66.7GB Qwen3-VL text
    # encoder likewise. With a replicated 66.3GB DiT there is room for the VAE (76.7GB
    # total) but not the text encoder (143.4GB > 143GB). FSDP-sharding the DiT drops it
    # to 8.3GB/rank, so DiT+VAE+TE all fit resident at 85.4GB with 57.6GB to spare --
    # and comm was measured cheap (all-gather 1GB = 4.0ms).
    offload_te: int = modal.parameter(default=1)
    offload_vae: int = modal.parameter(default=1)
    fsdp: int = modal.parameter(default=0)

    @modal.enter()
    def boot(self):
        import torch
        os.makedirs(LOCAL_OUT, exist_ok=True)

        # Environment MUST be set before FastVideo imports/loads.
        env = {
            "FASTVIDEO_ATTENTION_BACKEND": self.backend,
            "FASTVIDEO_VSA_SM100A": "0",          # Blackwell-only; we are sm90a
            "FASTVIDEO_VSA_CUTEDSL": "0",         # ditto
            "FASTVIDEO_FA4": "0",                 # FA4 is Blackwell CuTe; use FA2/FA3 on Hopper
            "FASTVIDEO_NVFP4_FA4": "0",
            "FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN": "0",
            "FASTVIDEO_MINIMAX_H3_FUSIONS": "all" if self.fusions else "0",
            "FASTVIDEO_INFERENCE_TORCH_COMPILE": "1" if self.compile_regional else "0",
            "FASTVIDEO_DISABLE_ATTENTION_COMPILE": "0",
            "FASTVIDEO_VAE_PARALLEL_DECODE": "1" if self.parallel_vae else "0",
            "FASTVIDEO_VAE_PARALLEL_ENCODE": "0",
            "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "gather",
            "FASTVIDEO_ULYSSES_A2A": "off",
            "FASTVIDEO_STAGE_LOGGING": "1",
        }
        os.environ.update(env)
        self.env = env

        if self.sdpa_kernel:
            want_cudnn = self.sdpa_kernel == "cudnn"
            torch.backends.cuda.enable_flash_sdp(not want_cudnn)
            torch.backends.cuda.enable_cudnn_sdp(want_cudnn)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            print(f"[boot] forced SDPA kernel -> {self.sdpa_kernel}")

        from fastvideo import VideoGenerator
        from fastvideo.api import (CompileConfig, ComponentConfig, EngineConfig,
                                   GeneratorConfig, OffloadConfig, ParallelismConfig,
                                   PipelineSelection, QuantizationConfig)

        experimental = {
            "attention_backend": self.backend,
            "inference_torch_compile": bool(self.compile_regional),
            "vae_parallel_decode": bool(self.parallel_vae),
            "vae_parallel_decode_strategy": "gather",
        }
        if self.backend == "VIDEO_SPARSE_ATTN_H3":
            experimental["VSA_sparsity"] = self.sparsity_x100 / 100.0
            experimental["VSA_tile_size"] = self.tile

        lora_path = None
        if self.lora:
            lora_path = f"/weights/fasth3-lora/{self.lora}/adapter_model.safetensors"
            if not os.path.exists(lora_path):
                raise RuntimeError(f"missing adapter {lora_path}")

        cfg = GeneratorConfig(
            model_path=self.model_path,
            pipeline=PipelineSelection(
                components=ComponentConfig(lora_path=lora_path, lora_strength=1.0,
                                           text_encoder_weights=self.te_path or None),
                experimental=experimental,
            ),
            engine=EngineConfig(
                num_gpus=self.num_gpus,
                use_fsdp_inference=bool(self.fsdp),
                parallelism=ParallelismConfig(tp_size=1, sp_size=self.num_gpus),
                offload=OffloadConfig(dit=False, dit_layerwise=False,
                                      text_encoder=bool(self.offload_te),
                                      vae=bool(self.offload_vae), pin_cpu_memory=True),
                compile=CompileConfig(enabled=False, mode=None, vae_enabled=True),
                quantization=(QuantizationConfig(
                                  transformer_quant=self.quant or None,
                                  text_encoder_quant=self.te_quant or None)
                              if (self.quant or self.te_quant) else None),

            ),
        )
        # Record WHICH hardware we actually got. Modal fulfils gpu="H100:8" with either
        # H100 80GB or H200 143GB, and the fully-resident configuration only fits on the
        # latter -- so a result is meaningless without knowing which one ran it.
        p0 = torch.cuda.get_device_properties(0)
        self.hw = {"name": p0.name, "hbm_gb": round(p0.total_memory / 1e9, 1),
                   "capability": f"{p0.major}.{p0.minor}", "count": torch.cuda.device_count()}
        print(f"[boot] HARDWARE: {self.hw}")

        t0 = time.time()
        self.gen = VideoGenerator.from_config(cfg)
        self.load_s = time.time() - t0
        print(f"[boot] loaded in {self.load_s:.1f}s | torch {torch.__version__} | env={env}")

    @modal.method()
    def sweep(self, prompts: list, configs: list, frames: int = 345,
              height: int = 768, width: int = 1344, steps: int = 5,
              seed: int = 1000, repeats: int = 2):
        """Sweep VSA settings inside ONE warm container.

        FastVideo builds the batch per request in the main process
        (`ForwardBatch(..., VSA_sparsity=fastvideo_args.VSA_sparsity)`) and its own
        comment says these are "per-request knobs (sweeps flip these between
        generate_video calls without respawning workers)". So mutating
        fastvideo_args.VSA_sparsity between calls changes sparsity with no reload.

        This matters here: the workspace runs one 8-GPU job at a time and each model
        load is ~6 min, so a 5-point sparsity sweep as separate containers would burn
        ~30 min of 8xGPU purely loading. This does it with a single load.

        Each config is a dict:
            {"sparsity": 0.7, "dense_first_n": 0, "dense_layers": [], "mode": "exempt"}
        `dense_first_n` runs the first N denoising steps fully dense -- FastVideo notes
        early steps "set global structure and are the most damage-prone", which is the
        mission's early-vs-late-denoising question with first-class support.
        """
        import time as _t
        from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig

        out = []
        first = True
        for cfg in configs:
            sp = float(cfg.get("sparsity", 0.9))
            try:
                self.gen.fastvideo_args.VSA_sparsity = sp
            except Exception as e:
                out.append({"config": cfg, "error": f"cannot set sparsity: {e}"})
                continue
            ext = {}
            if cfg.get("dense_first_n"):
                ext["vsa_dense_first_n_steps"] = int(cfg["dense_first_n"])
            if cfg.get("dense_layers"):
                ext["vsa_dense_layers"] = tuple(cfg["dense_layers"])
            if cfg.get("mode"):
                ext["vsa_mode"] = cfg["mode"]

            for item in prompts:
                pid = item["id"] if isinstance(item, dict) else "p"
                txt = item["text"] if isinstance(item, dict) else item
                # Include the config label: without it, two configs with the same
                # sparsity write to the SAME output path and silently overwrite each
                # other, which would destroy any cross-config quality comparison.
                lbl = cfg.get("label", "cfg")
                tag = f"{lbl}__sp{int(sp*100)}_df{cfg.get('dense_first_n',0)}_{pid}"

                def mk(sd, name, _txt=txt, _ext=ext):
                    return GenerationRequest(
                        prompt=_txt, negative_prompt="",
                        sampling=SamplingConfig(height=height, width=width,
                                                num_frames=frames, fps=24,
                                                num_inference_steps=steps,
                                                guidance_scale=1.0, batch_cfg=False,
                                                seed=sd),
                        output=OutputConfig(output_path=f"/outputs/{name}",
                                            save_video=True, return_frames=False),
                        extensions=dict(_ext),
                    )

                rec = {"config": dict(cfg), "prompt_id": pid, "tag": tag,
                       "hardware": self.hw,
                       "frames": frames, "steps": steps, "sparsity": sp,
                       "lora": self.lora, "backend": self.backend,
                       "num_gpus": self.num_gpus}
                if first:   # one discarded warmup absorbs compile + autotune
                    try:
                        self.gen.generate(mk(seed - 1, f"warmup_{tag}"))
                    except Exception as e:
                        rec["warmup_error"] = str(e)[:500]
                    first = False
                ts = []
                for i in range(repeats):
                    t0 = _t.perf_counter()
                    try:
                        res = self.gen.generate(mk(seed, f"{tag}_r{i}"))
                        ts.append(_t.perf_counter() - t0)
                        vp = getattr(res, "video_path", None)
                        if vp:
                            rec["video_path"] = str(vp)
                        gt = getattr(res, "generation_time", None)
                        if gt is not None:
                            rec.setdefault("generation_time_s", []).append(float(gt))
                        li = getattr(res, "logging_info", None)
                        if li is not None and getattr(li, "stages", None):
                            rec["stages"] = {k: (v.get("execution_time")
                                                 if isinstance(v, dict) else v)
                                             for k, v in li.stages.items()}
                    except Exception as e:
                        rec["error"] = str(e)[:900]
                        break
                if ts:
                    ts.sort()
                    rec["e2e_median_s"] = ts[len(ts) // 2]
                    rec["e2e_all_s"] = ts
                    rec["realtime_factor"] = ts[len(ts) // 2] / (frames / 24.0)
                out.append(rec)
                print(json.dumps(rec, default=str)[:900])
        # Persist outputs for evaluation AFTER all timing is done.
        import shutil
        for root, _, files in os.walk(LOCAL_OUT):
            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join("/outputs", os.path.relpath(src, LOCAL_OUT))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
        out_vol.commit()
        return out

    @modal.method()
    def run_suite(self, prompts: list, frames: int = 345, height: int = 768,
                  width: int = 1344, steps: int = 5, seed: int = 1000,
                  repeats: int = 3):
        """Run many prompts inside ONE warm container.

        A 144 GB model load dominates per-config cost, so amortizing it across the whole
        prompt suite (rather than one container per prompt) is the difference between a
        sweep that costs hours of 8xGPU time and one that costs minutes. The first prompt
        absorbs the compile/autotune warmup; the rest are measured directly.
        """
        out = []
        for i, item in enumerate(prompts):
            pid = item["id"] if isinstance(item, dict) else f"p{i}"
            txt = item["text"] if isinstance(item, dict) else item
            r = self.run.local(txt, pid, frames=frames, height=height, width=width,
                               steps=steps, seed=seed, repeats=repeats,
                               warmup=(i == 0))
            out.append(r)
        return out

    @modal.method()
    def run(self, prompt: str, prompt_id: str = "p", frames: int = 345,
            height: int = 768, width: int = 1344, steps: int = 5,
            seed: int = 1000, repeats: int = 3, warmup: bool = True):
        from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig

        def make_req(sd, tag):
            return GenerationRequest(
                prompt=prompt, negative_prompt="",
                sampling=SamplingConfig(height=height, width=width, num_frames=frames,
                                        fps=24, num_inference_steps=steps,
                                        guidance_scale=1.0, batch_cfg=False, seed=sd),
                output=OutputConfig(output_path=f"/outputs/{tag}", save_video=True,
                                    return_frames=False),
            )

        rec = {"prompt_id": prompt_id, "frames": frames, "steps": steps,
               "backend": self.backend, "sparsity": self.sparsity_x100 / 100.0,
               "lora": self.lora, "num_gpus": self.num_gpus,
               "compile_regional": bool(self.compile_regional),
               "sdpa_kernel": self.sdpa_kernel or "default",
               "quant": self.quant or "bf16", "te_quant": self.te_quant or "bf16",
               "model_path": self.model_path,
               "hardware": self.hw,
               "offload_te": bool(self.offload_te), "offload_vae": bool(self.offload_vae),
               "fsdp": bool(self.fsdp),
               "load_s": self.load_s, "env": self.env}

        if warmup:
            t0 = time.time()
            try:
                self.gen.generate(make_req(seed - 1, f"warmup_{prompt_id}"))
                rec["warmup_s"] = time.time() - t0
            except Exception as e:
                rec["warmup_error"] = str(e)[:600]

        times, stages, paths = [], [], []
        for i in range(repeats):
            t0 = time.time()
            try:
                res = self.gen.generate(make_req(seed, f"{prompt_id}_r{i}"))
                dt = time.time() - t0
                times.append(dt)
                gt = getattr(res, "generation_time", None)
                if gt is not None:
                    rec.setdefault("generation_time_s", []).append(float(gt))
                li = getattr(res, "logging_info", None)
                if li is not None and getattr(li, "stages", None):
                    stages.append({k: v for k, v in li.stages.items()})
                vp = getattr(res, "video_path", None)
                if vp:
                    paths.append(str(vp))
            except Exception as e:
                rec["error"] = str(e)[:2000]
                break

        if times:
            s = sorted(times)
            rec.update({
                "e2e_median_s": s[len(s) // 2],
                "e2e_min_s": s[0],
                "e2e_all_s": times,
                "realtime_factor": s[len(s) // 2] / (frames / 24.0),
                "stages": stages[-1] if stages else None,
                "video_paths": paths,
            })
        import torch
        rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out_vol.commit()
        print(json.dumps(rec, default=str)[:4000])
        return rec


@app.local_entrypoint()
def main(prompt: str = "A traceur sprints across a rooftop and vaults a concrete ledge, "
                       "tucking into a roll and rising into a run. Fast tracking shot follows "
                       "alongside. Late afternoon sun, long shadows, dust kicked up on landing.",
         frames: int = 345, steps: int = 5, sparsity: int = 90,
         backend: str = "VIDEO_SPARSE_ATTN_H3", lora: str = "vsa-datafree",
         gpus: int = 8, repeats: int = 3, tag: str = "smoke",
         sdpa_kernel: str = "", quant: str = "",
         offload_te: int = 1, offload_vae: int = 1, fsdp: int = 0,
         model_path: str = "/weights/base"):
    r = H3Runner(backend=backend, sparsity_x100=sparsity, lora=lora,
                 num_gpus=gpus, sdpa_kernel=sdpa_kernel, quant=quant,
                 offload_te=offload_te, offload_vae=offload_vae, fsdp=fsdp,
                 model_path=model_path).run.remote(prompt, tag, frames=frames, steps=steps,
                                           repeats=repeats)
    os.makedirs("results", exist_ok=True)
    with open(f"results/bench_{tag}.json", "w") as f:
        json.dump(r, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in r.items() if k != "env"}, indent=1, default=str)[:3000])


@app.local_entrypoint()
def sweep_main(lora: str = "vsa-datafree", backend: str = "VIDEO_SPARSE_ATTN_H3",
               gpus: int = 8, repeats: int = 2, frames: int = 345, steps: int = 5,
               quant: str = "", offload_te: int = 1, offload_vae: int = 1,
               fsdp: int = 0, model_path: str = "/weights/base",
               sdpa_kernel: str = "", te_quant: str = "", te_path: str = "",
               payload: str = "/tmp/sweep_payload.json", out: str = "results/sweep_phase34.json"):
    """Run a whole (sparsity x schedule x prompt) sweep inside ONE warm container."""
    with open(payload) as f:
        p = json.load(f)
    r = H3Runner(backend=backend, sparsity_x100=90, lora=lora,
                 num_gpus=gpus, quant=quant, offload_te=offload_te,
                 offload_vae=offload_vae, fsdp=fsdp,
                 model_path=model_path, sdpa_kernel=sdpa_kernel,
                 te_quant=te_quant, te_path=te_path).sweep.remote(p["prompts"], p["configs"],
                                             frames=frames, steps=steps, repeats=repeats)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=1, default=str)
    print(f"\nWROTE {out} ({len(r)} records)\n")
    print(f"{'config':<34}{'prompt':<12}{'E2E s':>9}{'RT':>7}")
    for rec in r:
        lab = (rec.get("config") or {}).get("label", "?")
        e2e = rec.get("e2e_median_s")
        rt = rec.get("realtime_factor")
        if e2e:
            print(f"{lab:<34}{rec.get('prompt_id','?'):<12}{e2e:>9.2f}{rt:>7.3f}")
        else:
            print(f"{lab:<34}{rec.get('prompt_id','?'):<12}   FAILED "
                  f"{str(rec.get('error') or rec.get('warmup_error'))[:70]}")


@app.local_entrypoint()
def gen(prompt_file: str, tag: str = "gen", frames: str = "345",
        height: int = 768, width: int = 1344, steps: str = "5", seed: int = 1000,
        backend: str = "VIDEO_SPARSE_ATTN_H3", sparsity: int = 90,
        lora: str = "", gpus: int = 8, quant: str = "",
        model_path: str = "/weights/fasth3-syn1300-full",
        te_path: str = "/weights/te-fp8-syn1300", te_quant: str = "",
        offload_te: int = 0, offload_vae: int = 0, fsdp: int = 1,
        out: str = ""):
    """Generate video(s) for ONE prompt, in the deployed realtime configuration.

    `frames` and `steps` may each be comma-separated lists; the product runs inside the
    SAME warm container so the model load is paid once. Only a new FRAME COUNT is a new
    shape for regional torch.compile -- step count is not -- so warmup is per frame
    count, not per configuration. Otherwise the reported latency would be a compile.
    """
    with open(prompt_file) as f:
        text = f.read().strip()
    runner = H3Runner(backend=backend, sparsity_x100=sparsity, lora=lora,
                      num_gpus=gpus, quant=quant, model_path=model_path,
                      te_path=te_path, te_quant=te_quant, offload_te=offload_te,
                      offload_vae=offload_vae, fsdp=fsdp)
    recs = []
    step_list = [int(x) for x in steps.split(",") if x.strip()]
    for fs in [int(x) for x in frames.split(",") if x.strip()]:
        for i, st in enumerate(step_list):
            pid = f"{tag}_f{fs}_s{st}"
            r = runner.run.remote(text, pid, frames=fs, height=height, width=width,
                                  steps=st, seed=seed, repeats=1, warmup=(i == 0))
            r["duration_s"] = fs / 24.0
            recs.append(r)
            print(f"\n=== {pid}: {fs} frames = {fs/24.0:.3f}s, {st} steps -> "
                  f"{r.get('e2e_median_s')} s  (RT {r.get('realtime_factor')})  "
                  f"{r.get('error') or ''}")
    os.makedirs("results", exist_ok=True)
    path = out or f"results/gen_{tag}.json"
    with open(path, "w") as f:
        json.dump(recs, f, indent=1, default=str)
    print(f"\nWROTE {path}")
    for r in recs:
        print(json.dumps({k: v for k, v in r.items() if k != "env"},
                         indent=1, default=str)[:1500])


@app.function(image=image, cpu=2, memory=4096, timeout=900,
              volumes={"/weights": vol})
def probe_fps():
    """Is `fps` GENERATIVE or just container metadata?

    Decides whether "higher FPS" is a config change or a post-process. Lives on this app
    so it reuses the already-built image; CPU-only, so it costs cents.
    """
    import subprocess
    out = {}

    def sh(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return r.stdout + r.stderr

    out["api_fps"] = sh("grep -rn 'fps' /opt/FastVideo/fastvideo/api/*.py | head -30")
    out["fps_consumers"] = sh(
        "grep -rn 'fps' /opt/FastVideo/fastvideo/pipelines/ --include=*.py | head -40")
    out["fps_vs_numframes"] = sh(
        "grep -rn 'fps' /opt/FastVideo/fastvideo/ --include=*.py "
        "| grep -i 'num_frames\\|duration\\|frame_count\\|temporal' | head -25")
    out["h3_json"] = sh(
        "grep -rn -i 'fps\\|frame_rate' /weights/fasth3-syn1300-full/*.json "
        "/weights/fasth3-syn1300-full/*/config.json 2>/dev/null | head -25")
    out["align"] = sh(
        "grep -rn 'align_num_frames' -A14 /opt/FastVideo/fastvideo/ --include=*.py | head -45")
    out["temporal_compression"] = sh(
        "grep -rn -i 'temporal_compression\\|temporal_downsample\\|frames_per_latent' "
        "/opt/FastVideo/fastvideo/ --include=*.py | head -15")
    return out


@app.local_entrypoint()
def fps_probe():
    r = probe_fps.remote()
    for k, v in r.items():
        print(f"\n{'='*20} {k} {'='*20}\n{v}")


@app.local_entrypoint()
def gen_prompts(prompt_files: str, tag: str = "pvar", frames: int = 345,
                height: int = 768, width: int = 1344, steps: int = 5, seed: int = 1000,
                backend: str = "VIDEO_SPARSE_ATTN_H3", sparsity: int = 90,
                lora: str = "", gpus: int = 8, quant: str = "",
                model_path: str = "/weights/fasth3-syn1300-full",
                te_path: str = "/weights/te-fp8-syn1300", te_quant: str = "",
                offload_te: int = 0, offload_vae: int = 0, fsdp: int = 1,
                repeats: int = 1, out: str = ""):
    """Generate SEVERAL prompt variants in one warm container, same seed and config.

    Prompt wording is the only free quality lever inside the realtime budget: it costs no
    latency at all. Holding seed, frames and config fixed makes the wording the single
    variable, which is the only way an adherence comparison means anything.

    `prompt_files` is a comma-separated list of local paths; the basename becomes the id.

    LATENCY CAVEAT, learned by getting it wrong: prompt length is ALSO a compiled shape,
    not just frame count. Warming up on the first prompt only leaves every later prompt
    paying a recompile inside its measured run -- which showed up as SHORTER prompts
    being 3 s slower, an impossible result. Use repeats>=2 and read the median: the
    first repeat absorbs the recompile, the rest are clean.
    """
    paths = [p.strip() for p in prompt_files.split(",") if p.strip()]
    runner = H3Runner(backend=backend, sparsity_x100=sparsity, lora=lora,
                      num_gpus=gpus, quant=quant, model_path=model_path,
                      te_path=te_path, te_quant=te_quant, offload_te=offload_te,
                      offload_vae=offload_vae, fsdp=fsdp)
    recs = []
    for i, p in enumerate(paths):
        with open(p) as f:
            text = f.read().strip()
        pid = f"{tag}_{os.path.splitext(os.path.basename(p))[0]}"
        r = runner.run.remote(text, pid, frames=frames, height=height, width=width,
                              steps=steps, seed=seed, repeats=repeats, warmup=(i == 0))
        r["prompt_file"] = p
        r["prompt_chars"] = len(text)
        recs.append(r)
        print(f"\n=== {pid}: {r.get('e2e_median_s')} s "
              f"(RT {r.get('realtime_factor')})  {r.get('error') or ''}")
    os.makedirs("results", exist_ok=True)
    path = out or f"results/gen_{tag}.json"
    with open(path, "w") as f:
        json.dump(recs, f, indent=1, default=str)
    print(f"\nWROTE {path}")
