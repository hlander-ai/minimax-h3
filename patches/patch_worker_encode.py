"""Encode the video inside the worker, so 1.07 GB of pixels never crosses the IPC boundary.

The measurement that motivates this
-----------------------------------
On H100 80GB the best configuration is 14.694 s against a 14.375 s budget. Of that,
**0.53 s is unattributed to any stage** -- it is `worker.pipe.send()` pickling the decoded
video (345 x 3 x 768 x 1344 uint8 = 1.07 GB) from worker 0 to the main process. Confirmed
by scaling: before the uint8 patch the same transfer was 4.27 GB of fp32 and cost 2.37 s;
4.0x fewer bytes gave 2.96x less time.

The main process uses those pixels for exactly one thing: writing a ~15 MB mp4.

The change
----------
`video_decoding_stage` runs before `audio_decoding_stage`, so the worker cannot mux audio
yet -- but it can encode the video. So:

  worker : decode -> uint8 -> encode a VIDEO-ONLY mp4 to container-local disk
           -> put the path in batch.extra, leave batch.output a 1-element placeholder
  main   : sees the path, skips frame post-processing entirely, and muxes the audio in
           with a stream copy (milliseconds on a 15 MB file)

Net effect is not the encode -- that still costs the same, just on the other side of the
pipe -- it is the 1.07 GB transfer, which disappears. Expected ~-0.5 s, against a 0.319 s
gap.

Falls back silently to the stock path if anything fails, so a broken encode cannot ship a
corrupt video; the main process simply finds no path and does what it always did.
"""
import sys

DEC = "/opt/FastVideo/fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_decoding.py"
VG = "/opt/FastVideo/fastvideo/entrypoints/video_generator.py"

DEC_OLD = '''            if is_output_rank and output is not None and output.dtype == torch.float32:
                output = output.mul_(255).clamp_(0, 255).to(torch.uint8)
            batch.output = output if is_output_rank else placeholder
            return batch'''

DEC_NEW = '''            if is_output_rank and output is not None and output.dtype == torch.float32:
                output = output.mul_(255).clamp_(0, 255).to(torch.uint8)
            # h3lab: encode HERE, in the worker. Otherwise this 1.07 GB tensor is pickled
            # through a pipe to the main process purely so it can write a 15 MB file.
            if is_output_rank and output is not None and output.dtype == torch.uint8:
                _p = _h3lab_worker_encode(output, int(getattr(batch, "fps", 24) or 24))
                if _p is not None:
                    batch.extra["h3lab_video"] = _p
                    # Reuse the module's own verifier-compatible empty tensor:
                    # verify_output requires 5 dims (V.with_dims(5)), so a 1-D
                    # stub fails StageVerificationError. Nothing large crosses.
                    output = placeholder
            batch.output = output if is_output_rank else placeholder
            return batch'''

DEC_HELPER = '''

def _h3lab_worker_encode(video_u8, fps):
    """Encode [B,C,T,H,W] uint8 to a video-only mp4 on local disk. None on any failure."""
    import os as _os
    import subprocess as _sp
    import tempfile as _tf
    from concurrent.futures import ThreadPoolExecutor

    import numpy as _np
    try:
        arr = video_u8[0].permute(1, 2, 3, 0).contiguous().numpy()   # T,H,W,C
        t, h, w, _c = arr.shape
        tmp = _tf.mkdtemp(prefix="h3lab_wenc_")
        n_seg = 4
        step = (t + n_seg - 1) // n_seg

        def _seg(i):
            chunk = arr[i * step:(i + 1) * step]
            if chunk.shape[0] == 0:
                return None
            path = _os.path.join(tmp, f"s{i:02d}.mp4")
            p = _sp.Popen(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
                           "-c:v", "libx264", "-preset", "ultrafast", "-g", "1",
                           "-pix_fmt", "yuv420p", "-threads", "0", "-y", path],
                          stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            p.stdin.write(memoryview(_np.ascontiguousarray(chunk)))
            p.stdin.close()
            return path if p.wait() == 0 else None

        with ThreadPoolExecutor(max_workers=n_seg) as ex:
            paths = [q for q in ex.map(_seg, range(n_seg)) if q]
        if not paths:
            return None
        listing = _os.path.join(tmp, "l.txt")
        with open(listing, "w") as fh:
            for q in paths:
                fh.write(f"file '{q}'\\n")
        out = _os.path.join(tmp, "video.mp4")
        r = _sp.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
                     "-i", listing, "-c", "copy", "-y", out], capture_output=True)
        return out if r.returncode == 0 and _os.path.exists(out) else None
    except Exception:
        return None
'''

# --- main process, two surgical edits -----------------------------------------
# 1) do not touch output_batch.output when the worker already encoded (it is now a
#    1-element placeholder and `rearrange` would raise)
# NOTE: this must go where output_batch EXISTS. `needs_frame_output` is computed BEFORE
# the forward runs, so testing output_batch there raises "cannot access local variable
# 'output_batch'". The frame branch below is the first point after the forward.
POST_OLD = """        elif not needs_frame_output:
            frames = None"""
POST_NEW = """        elif not needs_frame_output or output_batch.extra.get("h3lab_video"):
            # h3lab: the worker already encoded the video, so batch.output is a
            # placeholder and there are no frames to post-process.
            frames = None"""

# 2) in the save block, mux audio into the worker's mp4 instead of re-encoding
SAVE_OLD = """        save_to_disk = batch.save_video and not is_latent_output
        save_video_time = 0.0
        audio_mux_time = 0.0
        if save_to_disk:
            if audio_only:"""
SAVE_NEW = """        save_to_disk = batch.save_video and not is_latent_output
        save_video_time = 0.0
        audio_mux_time = 0.0
        _h3_pre = output_batch.extra.get("h3lab_video")
        if save_to_disk and _h3_pre and not audio_only:
            # h3lab: the worker encoded the video already. Mux the audio in with a
            # stream copy -- milliseconds on a ~15 MB file -- and skip re-encoding.
            import shutil as _sh
            import subprocess as _sp
            import wave as _wave
            _t0 = time.perf_counter()
            _ok = False
            try:
                _aud = output_batch.extra.get("audio")
                _sr = output_batch.extra.get("audio_sample_rate")
                if _aud is not None and _sr:
                    _a16, _nch = self._audio_to_int16(_aud)
                    _wav = _h3_pre + ".wav"
                    with _wave.open(_wav, "wb") as _w:
                        _w.setnchannels(_nch)
                        _w.setsampwidth(2)
                        _w.setframerate(int(_sr))
                        _w.writeframes(_a16.tobytes())
                    _ok = _sp.run(["ffmpeg", "-v", "error", "-i", _h3_pre, "-i", _wav,
                                   "-c:v", "copy", "-c:a", "aac", "-shortest",
                                   "-y", output_path],
                                  capture_output=True).returncode == 0
                else:
                    _sh.copy2(_h3_pre, output_path)
                    _ok = True
            except Exception:
                _ok = False
            if _ok:
                save_video_time = time.perf_counter() - _t0
                logger.info("Saved video to %s", output_path)
                save_to_disk = False        # done; fall past the stock path
            else:
                _h3_pre = None
        if save_to_disk:
            if audio_only:"""

def sub(path, old, new, name, marker, append=None):
    with open(path) as f:
        s = f.read()
    if marker in s:
        print(f"{name}_ALREADY"); return True
    if old not in s:
        print(f"{name}_ANCHOR_MISSING", file=sys.stderr); return False
    s = s.replace(old, new, 1)
    if append:
        s += append
    with open(path, "w") as f:
        f.write(s)
    print(f"{name}_PATCHED")
    return True


def main():
    ok = sub(DEC, DEC_OLD, DEC_NEW, "WENC_DECODE", "_h3lab_worker_encode", DEC_HELPER)
    ok &= sub(VG, POST_OLD, POST_NEW, "WENC_POST",
              'elif not needs_frame_output or output_batch.extra.get("h3lab_video"):')
    ok &= sub(VG, SAVE_OLD, SAVE_NEW, "WENC_SAVE", "_h3_pre = output_batch.extra.get")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
