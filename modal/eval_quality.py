"""
Quality evaluation for generated videos.

Runs on ONE GPU (cheap) against videos already written to the h3-outputs volume by
the generation harness. Deliberately separated from generation so that:
  * an 8-GPU node is never held while we compute metrics, and
  * metrics can be recomputed/extended later without regenerating video.

Reference-based metrics compare a candidate against the Base-H3 output for the SAME
prompt and seed. That same-seed comparison is our highest-signal cheap measurement:
it isolates what the acceleration changed, holding the sample path fixed.
"""
import json
import os

import modal

app = modal.App("h3-eval")
out_vol = modal.Volume.from_name("h3-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "imageio[ffmpeg]", "av", "opencv-python-headless",
                 "scikit-image", "lpips", "open_clip_torch", "ftfy", "regex")
    .add_local_dir("src/h3lab", "/root/h3lab")
)


@app.function(image=image, gpu="L40S", volumes={"/outputs": out_vol}, timeout=5400)
def evaluate_many(items: list, prompt: str, ref_rel: str | None = None,
                  ground_truth: bool = False):
    """Score several candidates against ONE reference in a single container.

    The reference here is the DENSEST sparsity we generated. All candidates share the
    prompt, the seed and the 4-step DMD schedule, so the only variable is attention
    sparsity -- which is exactly the question the mission asks.
    """
    import sys, os, json
    sys.path.insert(0, "/root")
    from h3lab.metrics import evaluate as _ev, classify, composite
    out = []
    for rel in items:
        cand = os.path.join("/outputs", rel)
        if not os.path.exists(cand):
            out.append({"candidate": rel, "error": "missing"}); continue
        try:
            m = _ev(cand, prompt, os.path.join("/outputs", ref_rel) if ref_rel else None)
            m["quality_class"] = classify(m, reference_is_ground_truth=ground_truth)
            m["composite"] = composite(m)
        except Exception as e:
            m = {"error": str(e)[:400]}
        m["candidate"] = rel; m["reference"] = ref_rel
        out.append(m)
        print(json.dumps(m, default=str)[:900])
    return out


@app.function(image=image, gpu="L40S", volumes={"/outputs": out_vol}, timeout=3600)
def evaluate(cand_rel: str, prompt: str, ref_rel: str | None = None):
    import sys
    sys.path.insert(0, "/root")
    from h3lab.metrics import evaluate as _ev, classify, composite

    cand = os.path.join("/outputs", cand_rel)
    ref = os.path.join("/outputs", ref_rel) if ref_rel else None
    if not os.path.exists(cand):
        return {"error": f"missing candidate {cand}"}
    if ref and not os.path.exists(ref):
        return {"error": f"missing reference {ref}"}

    m = _ev(cand, prompt, ref)
    m["quality_class"] = classify(m)
    m["composite"] = composite(m)
    m["candidate"] = cand_rel
    m["reference"] = ref_rel
    print(json.dumps(m, indent=1))
    return m


@app.function(image=image, volumes={"/outputs": out_vol}, timeout=600)
def list_outputs():
    found = []
    for root, _, files in os.walk("/outputs"):
        for f in files:
            if f.endswith((".mp4", ".webm", ".mkv")):
                p = os.path.join(root, f)
                found.append({"path": os.path.relpath(p, "/outputs"),
                              "mb": round(os.path.getsize(p) / 1e6, 2)})
    return sorted(found, key=lambda x: x["path"])


@app.local_entrypoint()
def fp8_te_correctness(out: str = "results/quality_fp8te.json"):
    """Is the sm90 dequant path CORRECT, or just fast?

    A wrong dequant produces plausible video at the right speed. LPIPS against the
    bf16-encoder run would NOT catch it -- FP8 shifts embeddings, which shifts the
    denoising trajectory, so a different sample is expected either way. The tell is
    absolute prompt adherence: garbage embeddings collapse CLIP text similarity.
    bf16-encoder reference on this prompt sits at 0.327-0.345.
    """
    P = ("A traceur sprints across a rooftop and vaults a concrete ledge, tucking into a "
         "roll and rising into a run. Fast tracking shot follows alongside. Late afternoon "
         "sun, long shadows, dust kicked up on landing, city skyline behind.")
    V = "/A traceur sprints across a rooftop and vaults a concrete ledge, tucking into a roll and rising into.mp4"
    cands = ["H100_FP8TE_RESIDENT__sp90_df0_parkour_r0" + V,
             "H100_FP8TE_RESIDENT__sp90_df0_parkour_r1" + V]
    r = evaluate_many.remote(cands, P, None, ground_truth=False)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=1, default=str)
    for m in r:
        if m.get("error"):
            print("ERROR:", m["error"][:200]); continue
        c = m.get("clip_text_sim")
        print(f"\n  clip_text_sim   {c:.4f}   (bf16-encoder band 0.327-0.345)")
        print(f"  motion          {m.get('motion_magnitude'):.2f}")
        print(f"  hf_energy       {m.get('hf_energy'):.1f}")
        print(f"  audio           {m.get('has_audio')} {m.get('audio_codec')} "
              f"{m.get('channels')}ch {m.get('audio_duration_s')}s")
        verdict = ("CORRECT -- prompt adherence intact" if c and 0.30 <= c <= 0.38
                   else "SUSPECT -- adherence outside the expected band")
        print(f"\n  => {verdict}")


@app.local_entrypoint()
def adapters(out: str = "results/quality_adapters.json"):
    """Rank the three released VSA adapters against Base H3, at identical latency.

    All three are rank-64 adapters on the same DiT at 90% sparsity, so they cost the same
    to run (measured: 15.07 / 14.64 s). Any quality difference between them is therefore
    free. Two were distilled data-free and two on teacher-generated video, which makes
    this a direct test of "do NOT assume data-free training is optimal".
    """
    PROMPTS = {
        "parkour": ("A traceur sprints across a rooftop and vaults a concrete ledge, tucking "
                    "into a roll and rising into a run. Fast tracking shot follows alongside. "
                    "Late afternoon sun, long shadows, dust kicked up on landing, city "
                    "skyline behind.",
                    "/A traceur sprints across a rooftop and vaults a concrete ledge, tucking into a roll and rising into.mp4",
                    "BASE_H3_REFERENCE_49fwd__sp0_df0_parkour_r0"),
        "texture": ("Macro shot of sunlight moving across a woven wool blanket as a cloud "
                    "passes. Individual fibers and slubs visible, the weave pattern crisp, "
                    "subtle color shift from warm to cool and back.",
                    "/Macro shot of sunlight moving across a woven wool blanket as a cloud passes. Individual fibers and s.mp4",
                    "BASE_H3_REFERENCE_49fwd__sp0_df0_texture_r0"),
    }
    ADAPTERS = ["vsa-datafree", "vsa-synthetic-step1300", "vsa-synthetic-step1900"]
    allr = []
    for pid, (prompt, vname, ref) in PROMPTS.items():
        cands = [f"adapter_{a}__sp90_df0_{pid}_r0{vname}" for a in ADAPTERS]
        allr += evaluate_many.remote(cands, prompt, ref + vname, ground_truth=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(allr, f, indent=1, default=str)
    print(f"\nWROTE {out}\n")
    print(f"{'candidate':<46}{'lpips':>7}{'ssim':>7}{'motion':>8}{'ratio':>7}{'hf':>8}{'clip':>7}")
    for m in allr:
        c = m.get("candidate", "?").split("/")[0][:45]
        if m.get("error"):
            print(f"{c:<46} ERROR {m['error'][:44]}"); continue
        g = lambda k, n=3: (f"%.{n}f" % m[k]) if isinstance(m.get(k), (int, float)) else "-"
        print(f"{c:<46}{g('lpips_vs_ref'):>7}{g('ssim_vs_ref'):>7}{g('motion_magnitude',2):>8}"
              f"{g('motion_ratio_vs_ref'):>7}{g('hf_energy',1):>8}{g('clip_text_sim'):>7}")


@app.local_entrypoint()
def vs_base(out: str = "results/quality_vs_base.json"):
    """THE quality measurement: accelerated configs vs Base H3 at 49 forwards.

    This is the one comparison where reference-based metrics are legitimate. The 4-step
    DMD adapter is explicitly trained to approximate the many-step model, so LPIPS /
    SSIM / motion-ratio against Base H3 measure real degradation -- unlike comparing two
    accelerated configs to each other, where a fixed seed yields a different sample
    rather than a worse one.
    """
    P = ("A traceur sprints across a rooftop and vaults a concrete ledge, tucking into a "
         "roll and rising into a run. Fast tracking shot follows alongside. Late afternoon "
         "sun, long shadows, dust kicked up on landing, city skyline behind.")
    V = "/A traceur sprints across a rooftop and vaults a concrete ledge, tucking into a roll and rising into.mp4"
    ref = "BASE_H3_REFERENCE_49fwd__sp0_df0_parkour_r0" + V
    cands = [
        "v2patch_fp8_sp90__sp90_df0_parkour_r0" + V,   # the DEPLOYED realtime config
        "sp90_df0_parkour_r0" + V,                      # 90% sparse, bf16-era
        "sp80_df0_parkour_r0" + V,                      # 80% sparse
        "sp70_df0_parkour_r0" + V,                      # 70% sparse (densest run)
        "fasth3_vsa90_r0" + V,                          # original bf16 baseline
    ]
    r = evaluate_many.remote(cands, P, ref, ground_truth=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=1, default=str)
    print(f"\nWROTE {out}\n")
    print(f"{'candidate':<42}{'lpips':>7}{'ssim':>7}{'motion':>8}{'ratio':>7}{'clip':>7}{'cls':>5}")
    for m in r:
        c = m.get("candidate", "?").split("/")[0][:41]
        if m.get("error"):
            print(f"{c:<42} ERROR {m['error'][:50]}"); continue
        g = lambda k, n=3: (f"%.{n}f" % m[k]) if isinstance(m.get(k), (int, float)) else "-"
        print(f"{c:<42}{g('lpips_vs_ref'):>7}{g('ssim_vs_ref'):>7}{g('motion_magnitude',2):>8}"
              f"{g('motion_ratio_vs_ref'):>7}{g('clip_text_sim'):>7}{str(m.get('quality_class')):>5}")


@app.local_entrypoint()
def sparsity_quality(out: str = "results/quality_sparsity.json"):
    """Does denser attention actually look better? sp90 / sp80 vs sp70 (densest)."""
    prompt = ("A traceur sprints across a rooftop and vaults a concrete ledge, tucking into "
              "a roll and rising into a run. Fast tracking shot follows alongside. Late "
              "afternoon sun, long shadows, dust kicked up on landing, city skyline behind.")
    V = "/A traceur sprints across a rooftop and vaults a concrete ledge, tucking into a roll and rising into.mp4"
    ref = "sp70_df0_parkour_r0" + V
    cands = ["sp90_df0_parkour_r0" + V, "sp80_df0_parkour_r0" + V,
             "sp90_df0_parkour_r1" + V, "fasth3_vsa90_r0" + V]
    r = evaluate_many.remote(cands, prompt, ref)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=1, default=str)
    print(f"\nWROTE {out}")
    for m in r:
        if m.get("error"):
            print(f"  {m['candidate'][:34]:36s} ERROR {m['error'][:70]}"); continue
        print(f"  {m['candidate'][:34]:36s} lpips={m.get('lpips_vs_ref')} "
              f"motion={m.get('motion_magnitude')} ratio={m.get('motion_ratio_vs_ref')} "
              f"hf={m.get('hf_energy')} clip={m.get('clip_text_sim')} "
              f"class={m.get('quality_class')}")


@app.local_entrypoint()
def main(cand: str = "", prompt: str = "", ref: str = "", ls: bool = False):
    if ls or not cand:
        for f in list_outputs.remote():
            print(f"  {f['mb']:8.2f} MB  {f['path']}")
        return
    r = evaluate.remote(cand, prompt, ref or None)
    os.makedirs("results", exist_ok=True)
    tag = cand.replace("/", "_").replace(".mp4", "")
    with open(f"results/eval_{tag}.json", "w") as f:
        json.dump(r, f, indent=1)
    print(json.dumps(r, indent=1))


@app.local_entrypoint()
def dance_ladder(out: str = "results/quality_dance_ladder.json",
                 dirs: str = "dance_f294_r0,ladder_f294_s8_r0,ladder_f294_s16_r0,ladder_f294_s25_r0",
                 ref_dir: str = "ladder_f294_s25_r0"):
    """Score the denoising-step ladder on the dance prompt.

    The 4-step DMD output and Base H3 at 8/16/25 steps, same prompt and seed. The
    densest-step run is the reference: it is the closest thing to ground truth we have
    for THIS prompt, and motion magnitude against it is the metric that tracks the
    smearing artifact the step count is supposed to fix.

    Paths are discovered rather than hardcoded -- the video filename is derived from the
    prompt text, so assuming it would silently score the wrong file.
    """
    with open("/tmp/prompt_dance.txt") as f:
        prompt = f.read().strip()

    all_files = list_outputs.remote()
    want = [d.strip() for d in dirs.split(",") if d.strip()]
    resolved = {}
    for d in want:
        hits = [f["path"] for f in all_files if f["path"].startswith(d + "/")]
        if hits:
            resolved[d] = hits[0]
        else:
            print(f"  WARNING: no mp4 found under {d}/")
    ref_rel = resolved.get(ref_dir)
    cands = [resolved[d] for d in want if d in resolved]
    if not cands:
        print("no candidates found; nothing to score")
        return

    r = evaluate_many.remote(cands, prompt, ref_rel, ground_truth=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=1, default=str)
    print(f"\nWROTE {out}\n")
    print(f"{'run':<24}{'motion':>9}{'hf_energy':>11}{'CLIP':>8}{'LPIPS':>8}{'SSIM':>8}")
    for m in r:
        name = m.get("candidate", "?").split("/")[0]
        if m.get("error"):
            print(f"{name:<24} ERROR {m['error'][:60]}"); continue
        def g(k, d="-"):
            v = m.get(k)
            return f"{v:.3f}" if isinstance(v, (int, float)) else d
        print(f"{name:<24}{g('motion_magnitude'):>9}{g('hf_energy'):>11}"
              f"{g('clip_text_sim'):>8}{g('lpips_vs_ref'):>8}{g('ssim_vs_ref'):>8}")
