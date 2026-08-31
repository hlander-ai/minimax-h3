#!/usr/bin/env python3
"""Drive the Phase 3+4 sweep through a single warm container.

Two families at roughly matched latency cost, which is the whole point:
  * UNIFORM sparsity -- attention spread evenly across all 4 denoising steps
  * MIXED schedules  -- first step dense (where FastVideo says global structure is
    set and damage is most likely), remaining steps pushed harder

`uniform 65%` (14.26 s, 35% mean density) and `dense_first_n=1 + 90%` (13.72 s, 32.5%)
cost about the same but allocate attention completely differently. Which wins is a
measurement.
"""
import json
import os
import subprocess
import sys

PROMPTS = [
    {"id": "parkour",   # fast motion -- the axis few-step distillation kills first
     "text": "A traceur sprints across a rooftop and vaults a concrete ledge, tucking into "
             "a roll and rising into a run. Fast tracking shot follows alongside. Late "
             "afternoon sun, long shadows, dust kicked up on landing, city skyline behind."},
    {"id": "occlusion",  # THE sparse-attention canary: object permanence across occlusion
     "text": "A red enamel mug sits on a wooden table. A person in a grey sweater walks "
             "between the camera and the table, fully hiding the mug for about a second, "
             "then continues out of frame, revealing the mug exactly where it was. Static "
             "camera, even daylight."},
    {"id": "texture",    # fine detail -- degrades under over-aggressive approximation
     "text": "Macro shot of sunlight moving across a woven wool blanket as a cloud passes. "
             "Individual fibers and slubs visible, the weave pattern crisp, subtle color "
             "shift from warm to cool and back."},
]

CONFIGS = [
    {"sparsity": 0.90, "dense_first_n": 0, "label": "uniform90_fasth3_baseline"},
    {"sparsity": 0.80, "dense_first_n": 0, "label": "uniform80"},
    {"sparsity": 0.70, "dense_first_n": 0, "label": "uniform70"},
    {"sparsity": 0.65, "dense_first_n": 0, "label": "uniform65_predicted_frontier"},
    {"sparsity": 0.90, "dense_first_n": 1, "label": "mixed_dense1_then90"},
]


def main():
    n_prompts = int(os.environ.get("N_PROMPTS", "3"))
    prompts = PROMPTS[:n_prompts]
    payload = {"prompts": prompts, "configs": CONFIGS}
    print(f"Sweep: {len(CONFIGS)} configs x {len(prompts)} prompts, ONE container")
    for c in CONFIGS:
        print(f"   {c['label']:32s} sparsity={c['sparsity']:.2f} dense_first_n={c['dense_first_n']}")
    with open("/tmp/sweep_payload.json", "w") as f:
        json.dump(payload, f)
    cmd = ["modal", "run", "modal/h3_bench.py::sweep_main"]
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", ""))
    sys.exit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
