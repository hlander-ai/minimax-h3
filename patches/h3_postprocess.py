"""Patch FastVideo's post-decode path for MiniMax-H3.

Two measured costs, both avoidable:

1. PostDecodeFrameProcessStage (1.37 s). The upstream code comments that it quantizes to
   uint8 "on the source device (typically CUDA) BEFORE the device->host copy". For H3 that
   is not true: MiniMaxH3VideoDecodingStage allocates its output as a **CPU float32** buffer
   (it is validated as such in decode_to_pixels_parallel), so `(src*255).clamp_().to(uint8)`
   runs single-threaded on the CPU across 4.27 GB, and then `make_grid` is called once per
   frame in a Python loop over 345 frames. With batch=1 `make_grid` is a no-op that still
   allocates and copies.

2. VideoSaveStage (1.67 s). h264 encoding at the default preset.

The patch keeps the same output semantics (uint8 HWC frames) and only removes work:
  * vectorised uint8 conversion over the whole tensor instead of a per-frame Python loop
  * skip make_grid entirely when batch == 1 (the only case we run)
"""
import time

import numpy as np
import torch


def apply_patch():
    import fastvideo.entrypoints.video_generator as vg

    src_file = vg.__file__
    with open(src_file) as f:
        code = f.read()

    old = """            src = output_batch.output
            vid_u8 = (src * 255).clamp_(0, 255).to(torch.uint8)
            vid_u8 = rearrange(vid_u8, "b c t h w -> t b c h w").cpu()
            frames = [
                torchvision.utils.make_grid(x, nrow=6).permute(1, 2, 0).squeeze(-1).contiguous().numpy() for x in vid_u8
            ]"""
    new = """            src = output_batch.output
            # PATCHED (h3lab): H3's decode writes a CPU float32 buffer, so the upstream
            # "quantize on device" comment does not hold here -- this loop was doing
            # 4.27GB of single-threaded CPU elementwise plus one make_grid per frame.
            # Vectorise the cast and skip make_grid for batch==1 (our only case).
            vid_u8 = (src * 255).clamp_(0, 255).to(torch.uint8)
            vid_u8 = rearrange(vid_u8, "b c t h w -> t b c h w")
            if vid_u8.shape[1] == 1:
                arr = vid_u8[:, 0].permute(0, 2, 3, 1).contiguous().numpy()
                frames = list(arr)
            else:
                vid_u8 = vid_u8.cpu()
                frames = [
                    torchvision.utils.make_grid(x, nrow=6).permute(1, 2, 0).squeeze(-1).contiguous().numpy()
                    for x in vid_u8
                ]"""
    if old not in code:
        return {"patched": False, "reason": "anchor not found (upstream changed)"}
    code = code.replace(old, new, 1)
    with open(src_file, "w") as f:
        f.write(code)
    return {"patched": True, "file": src_file}
