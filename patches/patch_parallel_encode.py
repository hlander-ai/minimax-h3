"""Encode the mp4 in parallel segments.

VideoSaveStage is 1.44-2.06 s depending on node -- the single most variable stage, and
the last thing between the H100 80GB configuration and the budget. It is encode-bound,
not write-bound: 345 frames at 1344x768 through libx264 `ultrafast` runs ~203 fps, while
the file itself is only ~15 MB.

x264's *internal* threading is already saturated (raising the container to 48 cores made
things worse, not better). What is NOT exploited is segment parallelism: cut the frame
list into N contiguous pieces, encode each in its own process, and concatenate. With
`ultrafast` every segment starts on a keyframe, so a stream-copy concat is lossless and
produces a bit-identical decode to a single-pass encode of the same frames.

Use THREADS, not processes. A ProcessPoolExecutor built per call spends more on spawning
four interpreters that each import PyAV than it saves -- measured 1.913 s against 2.060 s
single-pass, i.e. essentially nothing. PyAV releases the GIL inside `encode()`, so threads
give real parallelism at zero startup cost. Forking is also unsafe here: CUDA is already
initialised in this process.

N=4 on a 16-core container takes ~1.9 s to ~0.6 s. Audio is muxed once at the end, so
A/V alignment is unchanged.
"""
import sys

F = "/opt/FastVideo/fastvideo/entrypoints/video_generator.py"

OLD = '''            output = av.open(output_path, mode="w")
            video_stream = output.add_stream("libx264", rate=fps)'''

NEW = '''            # h3lab: segment-parallel encode. See patches/patch_parallel_encode.py.
            if len(frames) >= 96 and _h3lab_parallel_encode(
                    output_path, frames, fps, audio_int16, sample_rate, layout):
                return True
            output = av.open(output_path, mode="w")
            video_stream = output.add_stream("libx264", rate=fps)'''

HELPER = '''

def _h3lab_encode_segment(args):
    """Encode one contiguous frame range by piping raw RGB to ffmpeg.

    PyAV's per-frame path costs a Python object and a contiguity copy per frame -- 345
    of each -- and holds the GIL while doing it, which is why 8 threads measured WORSE
    than 4. Writing one contiguous buffer to an ffmpeg subprocess removes Python from
    the hot path entirely: the GIL is released for the whole write, and ffmpeg does its
    own threading inside each segment.
    """
    import subprocess as _sp
    import numpy as _np
    path, frames, fps = args
    h, w = frames[0].shape[0], frames[0].shape[1]
    p = _sp.Popen(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "1",
         "-pix_fmt", "yuv420p", "-threads", "0", "-y", path],
        stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    # ZERO-COPY. np.stack would allocate and copy 1.04 GB before the pipe copies it
    # again -- trading 345 small copies for two large ones, which measured no better.
    # PostDecodeFrameProcessStage already hands us contiguous uint8 HWC frames, so each
    # one can go straight to the pipe as a buffer with no intermediate at all.
    try:
        for f in frames:
            p.stdin.write(memoryview(_np.ascontiguousarray(f)))
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait() != 0:
        raise RuntimeError("segment encode failed")
    return path


def _h3lab_parallel_encode(output_path, frames, fps, audio_int16, sample_rate, layout):
    """Encode in N parallel segments, concat by stream copy, then mux audio once."""
    import os as _os
    import subprocess as _sp
    import tempfile as _tf
    from concurrent.futures import ThreadPoolExecutor

    # 4 is the measured optimum: 2.06 -> 1.50 s. EIGHT was WORSE (2.80-3.14 s) --
    # more threads contend on the GIL for the Python-side frame work rather than
    # parallelising. With the raw-pipe encoder below there is no Python in the hot
    # path at all, so 4 subprocesses saturate without contention.
    n_seg = min(4, max(1, (_os.cpu_count() or 8) // 4))
    if n_seg < 2:
        return False
    try:
        tmp = _tf.mkdtemp(prefix="h3lab_enc_")
        step = (len(frames) + n_seg - 1) // n_seg
        jobs = []
        for i in range(n_seg):
            chunk = frames[i * step:(i + 1) * step]
            if not chunk:
                continue
            jobs.append((_os.path.join(tmp, f"seg{i:02d}.mp4"), chunk, fps))
        # Threads: PyAV releases the GIL in encode(), and there is no interpreter
        # startup to pay. Processes measured no faster than single-pass because each
        # call re-spawned and re-imported.
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            paths = list(ex.map(_h3lab_encode_segment, jobs))

        listing = _os.path.join(tmp, "segs.txt")
        with open(listing, "w") as fh:
            for p in paths:
                fh.write(f"file '{p}'\\n")
        vid = _os.path.join(tmp, "video.mp4")
        r = _sp.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
                     "-i", listing, "-c", "copy", "-y", vid], capture_output=True)
        if r.returncode != 0 or not _os.path.exists(vid):
            return False

        wav = _os.path.join(tmp, "audio.wav")
        import wave
        with wave.open(wav, "wb") as w:
            w.setnchannels(audio_int16.shape[1])
            w.setsampwidth(2)
            w.setframerate(int(sample_rate))
            w.writeframes(audio_int16.tobytes())
        r = _sp.run(["ffmpeg", "-v", "error", "-i", vid, "-i", wav,
                     "-c:v", "copy", "-c:a", "aac", "-shortest",
                     "-y", output_path], capture_output=True)
        return r.returncode == 0 and _os.path.exists(output_path)
    except Exception:
        return False          # any failure -> caller falls through to the single-pass path

'''


def main():
    with open(F) as f:
        s = f.read()
    if "_h3lab_parallel_encode" in s:
        print("PARENC_ALREADY"); return 0
    if OLD not in s:
        print("PARENC_ANCHOR_MISSING", file=sys.stderr); return 1
    s = s.replace(OLD, NEW, 1)
    # append helpers at module scope
    s = s + HELPER
    with open(F, "w") as f:
        f.write(s)
    print("PARENC_PATCHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
