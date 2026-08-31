"""
Video quality metrics for the H3 Pareto search.

Two families, deliberately kept separate:

REFERENCE-BASED (candidate vs Base-H3 at identical prompt+seed) -- our cheapest high-signal
  measure of "how much did the acceleration change the output".
    lpips, dreamsim-style embedding distance, psnr, ssim, clip_frame_sim

ABSOLUTE (candidate alone) -- necessary because reference similarity ALONE is misleading.
  A 4-step model that renders a near-static video scores well on per-frame similarity to a
  reference whose motion it failed to reproduce, if the reference is also fairly static.
  These catch the characteristic few-step failure modes directly:
    motion_magnitude   -- optical-flow energy. Few-step distillation collapses motion to
                          slow-motion/static. THE canonical few-step regression.
    hf_energy          -- high-frequency (Laplacian) energy. Detects texture smoothing,
                          the canonical over-sparsification / over-distillation regression.
    temporal_flicker   -- frame-to-frame instability, detects incoherence
    clip_text_sim      -- prompt adherence

Design note: we never reduce to a single number for a DECISION. `composite()` exists only so
the Pareto frontier code has a scalar to sort on; every gate reads the vector.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ io
def read_video(path, max_frames=None, stride=1):
    """Return uint8 array [T, H, W, 3]."""
    import imageio.v3 as iio
    frames = []
    for i, f in enumerate(iio.imiter(path, plugin="pyav")):
        if i % stride:
            continue
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    return np.stack(frames)


# ------------------------------------------------- absolute (no reference)
def motion_magnitude(frames, device="cuda", sample=32):
    """Mean optical-flow magnitude in px/frame, via RAFT.

    This is the single most important absolute metric we compute. Few-step distilled
    video models systematically under-produce motion; a candidate whose motion_magnitude
    is materially below Base H3's on the same prompt has regressed on the axis viewers
    notice most, even when every per-frame metric looks fine.
    """
    import torch
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    idx = np.linspace(0, len(frames) - 2, min(sample, len(frames) - 1)).astype(int)
    w = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=w, progress=False).to(device).eval()
    tf = w.transforms()

    mags = []
    with torch.no_grad():
        for i in idx:
            a = torch.from_numpy(frames[i]).permute(2, 0, 1)[None].float() / 255.0
            b = torch.from_numpy(frames[i + 1]).permute(2, 0, 1)[None].float() / 255.0
            a, b = tf(a, b)
            flow = model(a.to(device), b.to(device))[-1]
            mags.append(torch.linalg.vector_norm(flow, dim=1).mean().item())
    return float(np.mean(mags))


def hf_energy(frames, sample=32):
    """Laplacian variance averaged over frames -- a texture/sharpness proxy.

    Over-aggressive step reduction and over-sparsified attention both smooth out
    high-frequency detail. This catches that directly, without a reference.
    """
    import cv2
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
    vals = []
    for i in idx:
        g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        vals.append(cv2.Laplacian(g, cv2.CV_64F).var())
    return float(np.mean(vals))


def temporal_flicker(frames, sample=64):
    """Mean abs frame-to-frame delta, normalized. High => incoherent/flickery."""
    idx = np.linspace(0, len(frames) - 2, min(sample, len(frames) - 1)).astype(int)
    d = [np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32)).mean()
         for i in idx]
    return float(np.mean(d))


# ------------------------------------------------- reference-based
def psnr_ssim(cand, ref, sample=32):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    n = min(len(cand), len(ref))
    idx = np.linspace(0, n - 1, min(sample, n)).astype(int)
    ps, ss = [], []
    for i in idx:
        ps.append(psnr(ref[i], cand[i], data_range=255))
        ss.append(ssim(ref[i], cand[i], channel_axis=2, data_range=255))
    return float(np.mean(ps)), float(np.mean(ss))


def lpips_dist(cand, ref, device="cuda", sample=24):
    import lpips as _l
    import torch
    net = _l.LPIPS(net="alex").to(device)
    n = min(len(cand), len(ref))
    idx = np.linspace(0, n - 1, min(sample, n)).astype(int)
    vals = []
    with torch.no_grad():
        for i in idx:
            a = torch.from_numpy(cand[i]).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            b = torch.from_numpy(ref[i]).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            vals.append(net(a, b).item())
    return float(np.mean(vals))


def clip_scores(frames, prompt, device="cuda", sample=16):
    """Returns (text_sim, frame_to_frame_consistency)."""
    import open_clip
    import torch
    from PIL import Image

    model, _, pre = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    tok = open_clip.get_tokenizer("ViT-B-32")
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
    ims = torch.stack([pre(Image.fromarray(frames[i])) for i in idx]).to(device)
    with torch.no_grad():
        f = model.encode_image(ims)
        f = f / f.norm(dim=-1, keepdim=True)
        t = model.encode_text(tok([prompt]).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
        text_sim = (f @ t.T).mean().item()
        consec = (f[:-1] * f[1:]).sum(-1).mean().item()
    return float(text_sim), float(consec)


# ------------------------------------------------------------------- audio
def audio_probe(path):
    """Does the file actually carry audio, and does it line up with the video?

    H3 generates native STEREO audio jointly with video, so an accelerated config that
    silently drops or degrades the audio track would pass every pixel metric while failing
    the actual target. Checked explicitly rather than assumed.
    """
    import json
    import subprocess

    out = {"has_audio": False}
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=120)
        info = json.loads(r.stdout or "{}")
    except Exception as e:
        return {"has_audio": False, "error": str(e)[:200]}

    for st in info.get("streams", []):
        if st.get("codec_type") == "audio":
            out.update({
                "has_audio": True,
                "audio_codec": st.get("codec_name"),
                "sample_rate": int(st.get("sample_rate", 0) or 0),
                "channels": int(st.get("channels", 0) or 0),
                "audio_duration_s": float(st.get("duration") or 0.0),
            })
        elif st.get("codec_type") == "video":
            out["video_duration_s"] = float(st.get("duration") or 0.0)
            out["video_fps"] = st.get("r_frame_rate")
    if out.get("audio_duration_s") and out.get("video_duration_s"):
        out["av_duration_delta_s"] = round(
            out["audio_duration_s"] - out["video_duration_s"], 3)
    return out


def av_sync_score(path, sample_hz=25):
    """Correlation between audio energy envelope and visual motion energy.

    A crude but reference-free proxy for A/V synchronization: in most real footage,
    impacts and onsets co-occur with motion. It will not catch lip-sync errors -- that
    needs a SyncNet-class model -- but it does catch gross desync and silent tracks,
    which are the failure modes acceleration actually introduces.
    """
    import subprocess
    import numpy as _np

    try:
        # decode audio to mono 16k f32
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
             "-ar", "16000", "-"],
            capture_output=True, timeout=300)
        a = _np.frombuffer(r.stdout, dtype=_np.float32)
        if a.size < 16000:
            return {"av_sync": None, "reason": "no/short audio"}
        hop = 16000 // sample_hz
        env = _np.array([_np.sqrt((a[i:i + hop] ** 2).mean() + 1e-12)
                         for i in range(0, len(a) - hop, hop)])

        frames = read_video(path, stride=max(1, round(24 / sample_hz)))
        mot = _np.array([_np.abs(frames[i + 1].astype(_np.float32)
                                 - frames[i].astype(_np.float32)).mean()
                         for i in range(len(frames) - 1)])
        n = min(len(env), len(mot))
        if n < 10:
            return {"av_sync": None, "reason": "too short"}
        e, m = env[:n], mot[:n]
        e = (e - e.mean()) / (e.std() + 1e-9)
        m = (m - m.mean()) / (m.std() + 1e-9)
        best, lag = -2.0, 0
        for L in range(-sample_hz, sample_hz + 1):        # +/- 1 second
            if L < 0:
                c = float((e[-L:] * m[:n + L]).mean())
            elif L > 0:
                c = float((e[:n - L] * m[L:]).mean())
            else:
                c = float((e * m).mean())
            if c > best:
                best, lag = c, L
        return {"av_sync": round(best, 4), "av_lag_frames": lag,
                "av_sync_at_zero": round(float((e * m).mean()), 4)}
    except Exception as ex:
        return {"av_sync": None, "error": str(ex)[:200]}


# ------------------------------------------------------------------ driver
def evaluate(cand_path, prompt, ref_path=None, device="cuda"):
    cand = read_video(cand_path, stride=2)
    m = {
        "n_frames": int(len(cand)),
        "motion_magnitude": motion_magnitude(cand, device),
        "hf_energy": hf_energy(cand),
        "temporal_flicker": temporal_flicker(cand),
    }
    m.update(audio_probe(cand_path))
    m.update(av_sync_score(cand_path))

    try:
        ts, cs = clip_scores(cand, prompt, device)
        m["clip_text_sim"], m["clip_frame_consistency"] = ts, cs
    except Exception as e:
        m["clip_error"] = str(e)[:200]

    if ref_path:
        ref = read_video(ref_path, stride=2)
        try:
            p, s = psnr_ssim(cand, ref)
            m["psnr_vs_ref"], m["ssim_vs_ref"] = p, s
        except Exception as e:
            m["psnr_error"] = str(e)[:200]
        try:
            m["lpips_vs_ref"] = lpips_dist(cand, ref, device)
        except Exception as e:
            m["lpips_error"] = str(e)[:200]
        try:
            rm = motion_magnitude(ref, device)
            m["ref_motion_magnitude"] = rm
            # <1.0 means the candidate moves LESS than the reference -- the classic
            # few-step failure. We track the ratio explicitly because it is the metric
            # most likely to be masked by good per-frame similarity scores.
            m["motion_ratio_vs_ref"] = m["motion_magnitude"] / rm if rm > 1e-6 else None
        except Exception as e:
            m["motion_ref_error"] = str(e)[:200]
    return m


def classify(m, reference_is_ground_truth=False):
    """Quality gate: A effectively lossless / B tiny / C noticeable / D major.

    IMPORTANT -- learned the hard way. This is only meaningful when the reference is
    something the candidate is *supposed to approximate* (i.e. Base H3 vs an accelerated
    variant at the same seed). It is NOT meaningful between two accelerated configs that
    differ in sparsity or step count: changing those changes the denoising trajectory, so
    the same seed yields a DIFFERENT sample rather than a degraded one. Measured across
    sparsity settings, LPIPS came back 0.58-0.63 and SSIM 0.36-0.41 for every pair -- i.e.
    "different video", not "worse video" -- and this function dutifully called all of them
    class D, which was meaningless.

    Callers must pass reference_is_ground_truth=True to get a class back. Otherwise the
    honest answer is None, and the comparison has to be made on absolute metrics or by
    blinded pairwise preference.

    Thresholds remain provisional and MUST be recalibrated against human judgments before
    they gate a real decision.
    """
    if not reference_is_ground_truth:
        return None
    lp = m.get("lpips_vs_ref")
    mr = m.get("motion_ratio_vs_ref")
    if lp is None:
        return None
    if m.get("has_audio") is False:
        return "D"                                  # native audio is part of the target
    motion_bad = mr is not None and mr < 0.80      # lost >20% of motion energy
    motion_poor = mr is not None and mr < 0.90
    if lp < 0.05 and not motion_poor:
        return "A"
    if lp < 0.15 and not motion_bad:
        return "B"
    if lp < 0.30:
        return "C"
    return "D"


def composite(m):
    """Scalar for Pareto sorting ONLY. Never use for a final quality call."""
    score = 100.0
    if (lp := m.get("lpips_vs_ref")) is not None:
        score -= 150.0 * lp
    if (mr := m.get("motion_ratio_vs_ref")) is not None:
        score -= 40.0 * abs(1.0 - min(mr, 1.5))    # penalize both loss and runaway motion
    if (cs := m.get("clip_text_sim")) is not None:
        score += 20.0 * (cs - 0.25)
    return round(score, 2)
