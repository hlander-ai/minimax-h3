"""
Pull the weights we need onto a Modal Volume.

Download strategy (deliberate, not exhaustive):
  * MiniMax-H3 base, T2AV subset only  (~144 GB) -- we skip FL2VA/, Ref2VA/, transformer_ref/,
    which are the first-last-frame and reference-conditioned pipelines (288 GB) we don't need.
  * FastH3 LoRA bundle (17.5 GB) -- contains ALL FOUR ablations (dense-datafree, vsa-datafree,
    vsa-synthetic-1300, vsa-synthetic-1900) as adapters on top of base. Downloading these instead
    of four 148 GB full checkpoints saves ~575 GB and gives the same configurations.

Run:  modal run modal/download_weights.py
"""
import modal

app = modal.App("h3-download")

vol = modal.Volume.from_name("h3-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]==0.35.3", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

BASE_REPO = "MiniMaxAI/MiniMax-H3"
LORA_REPO = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA"

# Everything the text->audio+video path needs, and nothing else.
BASE_ALLOW = [
    "transformer/*", "text_encoder/*", "vae/*", "audio_vae/*",
    "tokenizer/*", "processor/*", "scheduler/*", "audio_scheduler/*",
    "model_index.json", "modular_model_index.json", "LICENSE", "README.md",
    "docs/*",
]
BASE_IGNORE = ["FL2VA/*", "Ref2VA/*", "transformer_ref/*", "assets/*", "scripts/*"]


@app.function(image=image, volumes={"/weights": vol}, timeout=7200,
              cpu=16, memory=32768, secrets=[modal.Secret.from_name("huggingface-token")])
def fetch(repo: str, subdir: str, allow=None, ignore=None):
    import os, time
    from huggingface_hub import snapshot_download

    dest = f"/weights/{subdir}"
    os.makedirs(dest, exist_ok=True)
    t0 = time.time()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(
        repo_id=repo, local_dir=dest,
        allow_patterns=allow, ignore_patterns=ignore,
        max_workers=16, token=tok,
    )
    dt = time.time() - t0
    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(dest) for f in fs
        if os.path.exists(os.path.join(r, f))
    )
    vol.commit()
    msg = f"{repo} -> {dest}: {total/1e9:.1f} GB in {dt/60:.1f} min ({total/1e9/dt*1000:.0f} MB/s)"
    print(msg)
    return msg


@app.local_entrypoint()
def main():
    jobs = [
        (BASE_REPO, "base", BASE_ALLOW, BASE_IGNORE),
        (LORA_REPO, "fasth3-lora", None, None),
    ]
    for r in fetch.starmap(jobs):
        print("DONE:", r)
