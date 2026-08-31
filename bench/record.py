#!/usr/bin/env python3
"""Ingest results/bench_*.json (+ optional results/eval_*.json) into experiments.jsonl."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from h3lab.exp import Experiment, record, load, pareto, summary_table, best_realtime, \
    highest_quality, fastest  # noqa: E402


def ingest(path):
    with open(path) as f:
        r = json.load(f)
    if isinstance(r, list):
        return [ingest_one(x, path) for x in r]
    return [ingest_one(r, path)]


def ingest_one(r, path):
    tag = os.path.basename(path).replace("bench_", "").replace(".json", "")
    steps = r.get("steps")
    e = Experiment(
        exp_id=tag,
        checkpoint="/weights/base (MiniMaxAI/MiniMax-H3)",
        lora=r.get("lora") or None,
        lora_strength=1.0 if r.get("lora") else None,
        scheduler="dmd",
        num_steps=steps,
        # FastVideo counts SIGMA GRID POINTS; N points = N-1 DiT forwards.
        dit_evals=(steps - 1) if steps else None,
        cfg_scale=1.0,
        attention_method=r.get("backend", ""),
        attention_sparsity=r.get("sparsity"),
        precision=r.get("quant", "bf16"),
        gpu_count=r.get("num_gpus", 8),
        gpu_type="H200-143GB (Modal 'H100:8')",
        parallelism=f"ulysses{r.get('num_gpus', 8)}",
        compile_mode="regional" if r.get("compile_regional") else None,
        frames=r.get("frames", 360),
        audio=True,
        e2e_latency_s=r.get("e2e_median_s"),
        peak_vram_gb=r.get("peak_vram_gb"),
        n_timing_runs=len(r.get("e2e_all_s") or []),
        status="ok" if r.get("e2e_median_s") else "failed",
        error=(r.get("error") or r.get("warmup_error")),
        notes=f"sdpa={r.get('sdpa_kernel','default')} load={r.get('load_s',0):.0f}s",
    )
    if r.get("e2e_all_s") and len(r["e2e_all_s"]) > 1:
        import statistics
        e.latency_stddev_s = statistics.stdev(r["e2e_all_s"])
    # attach quality if an eval exists for the same tag
    for ev in glob.glob(f"results/eval_*{r.get('prompt_id','')}*.json"):
        try:
            with open(ev) as f:
                q = json.load(f)
            e.quality.update(q)
            e.quality_class = q.get("quality_class")
        except Exception:
            pass
    record(e)
    return e.exp_id


def main():
    files = sys.argv[1:] or sorted(glob.glob("results/bench_*.json"))
    seen = {r["exp_id"] for r in load()}
    n = 0
    for f in files:
        for eid in ingest(f):
            if eid in seen:
                continue
            n += 1
    rows = load()
    print(f"ingested {n} new; {len(rows)} total\n")
    print(summary_table(rows))
    print()
    for name, fn in (("BEST REALTIME", best_realtime), ("HIGHEST QUALITY", highest_quality),
                     ("FASTEST", fastest)):
        r = fn(rows)
        if r:
            print(f"{name:16s}: {r['exp_id']}  {r.get('e2e_latency_s')}s  "
                  f"rt={r.get('realtime_factor')}  q={(r.get('quality') or {}).get('composite')}")


if __name__ == "__main__":
    main()
