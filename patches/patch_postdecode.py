"""Applied inside the image. Removes non-model work from H3's output path.

Measured at 90% sparsity, FP8, fully resident (E2E 15.82 s vs a 14.375 s budget):

  denoising            8.60 s   <- model
  VAE decode           2.19 s   <- model
  PostDecodeFrameProc  0.89 s   \\
  VideoSave            1.57 s    >  non-model: 5.02 s total
  unattributed         2.37 s   /

THREE fixes, all lossless (they change where/how bytes move, never pixel values):

1. **uint8 in the worker (the big one).** MiniMaxH3VideoDecodingStage returns a CPU
   **float32** pixel buffer -- 345x3x768x1344x4B = 4.27 GB -- which then crosses the
   multiproc-executor boundary from worker 0 to the main process. 4.27 GB / 2.37 s is
   ~1.8 GB/s, i.e. the unattributed time is that IPC transfer. The very next thing the
   main process does is cast it to uint8. Doing that cast in the worker sends 1.07 GB
   instead of 4.27 GB.

2. **No second full-size allocation in post-decode.** `(src * 255)` allocated another
   4.27 GB fp32 tensor before the cast; scale in place on a buffer we own. (Also skip
   `make_grid`, a no-op at batch==1 that still allocates, once per frame over 345 frames.)

3. **Let x264 thread.** `tune=zerolatency` disables frame-level threading -- correct for
   streaming, wrong for a batch save on a 16-CPU container. Dropping it (keeping
   preset=ultrafast) lets `threads=0` actually parallelise.
"""
import sys

VG = "/opt/FastVideo/fastvideo/entrypoints/video_generator.py"
DEC = "/opt/FastVideo/fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_decoding.py"

# ---- 1. cast to uint8 inside the worker, before the IPC hop -------------------
DEC_OLD = """            batch.output = output if is_output_rank else placeholder
            return batch"""
DEC_NEW = """            # h3lab patch: cast to uint8 HERE, in the worker, before this tensor
            # crosses the multiproc-executor boundary. It was shipping 4.27 GB of
            # fp32 (345x3x768x1344) at ~1.8 GB/s and the main process cast it to
            # uint8 immediately anyway. Same pixels, 4x fewer bytes on the wire.
            if is_output_rank and output is not None and output.dtype == torch.float32:
                output = output.mul_(255).clamp_(0, 255).to(torch.uint8)
            batch.output = output if is_output_rank else placeholder
            return batch"""

# ---- 2. post-decode: accept uint8, avoid the extra fp32 copy ------------------
POST_OLD = '''            src = output_batch.output
            vid_u8 = (src * 255).clamp_(0, 255).to(torch.uint8)
            vid_u8 = rearrange(vid_u8, "b c t h w -> t b c h w").cpu()
            frames = [
                torchvision.utils.make_grid(x, nrow=6).permute(1, 2, 0).squeeze(-1).contiguous().numpy() for x in vid_u8
            ]'''
POST_NEW = '''            src = output_batch.output
            # h3lab patch: the decode stage now hands us uint8 already (see
            # minimax_h3_decoding). Fall back to scaling in place -- NOT `src * 255`,
            # which allocated a second 4.27 GB fp32 tensor -- if it is still float.
            if src.dtype == torch.uint8:
                vid_u8 = src
            else:
                vid_u8 = src.mul_(255).clamp_(0, 255).to(torch.uint8)
            vid_u8 = rearrange(vid_u8, "b c t h w -> t b c h w")
            if vid_u8.shape[1] == 1:
                frames = list(vid_u8[:, 0].permute(0, 2, 3, 1).contiguous().numpy())
            else:
                vid_u8 = vid_u8.cpu()
                frames = [
                    torchvision.utils.make_grid(x, nrow=6).permute(1, 2, 0).squeeze(-1).contiguous().numpy()
                    for x in vid_u8
                ]'''

# ---- 3. allow x264 frame threading -------------------------------------------
ENC_OLD = '''            video_stream.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
            }'''
ENC_NEW = '''            video_stream.options = {
                "preset": "ultrafast",
                # h3lab patch: dropped tune=zerolatency. It disables frame-level
                # threading (right for streaming, wrong for a batch save) and was
                # why threads=0 bought almost nothing on a 16-CPU container.
                "threads": "0",
            }
            try:
                video_stream.thread_type = "AUTO"
                video_stream.thread_count = 0
            except Exception:
                pass'''


def sub(path, old, new, name, marker):
    with open(path) as f:
        s = f.read()
    if marker in s:
        print(f"{name}_ALREADY_PATCHED")
        return True
    if old not in s:
        print(f"{name}_ANCHOR_MISSING", file=sys.stderr)
        return False
    with open(path, "w") as f:
        f.write(s.replace(old, new, 1))
    print(f"{name}_PATCHED")
    return True


def main():
    ok = True
    ok &= sub(DEC, DEC_OLD, DEC_NEW, "DECODE_U8", "cast to uint8 HERE, in the worker")
    ok &= sub(VG, POST_OLD, POST_NEW, "POSTDECODE", "the decode stage now hands us uint8")
    ok &= sub(VG, ENC_OLD, ENC_NEW, "ENCODER", "dropped tune=zerolatency")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
