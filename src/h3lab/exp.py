"""
Experiment database + Pareto frontier for the H3 realtime project.

The whole project is a constrained optimization:
    maximize quality  subject to  e2e_latency_s <= video_duration_s

so the two things this module must get right are (a) recording every field needed to
reproduce a run, and (b) computing which configurations are non-dominated.

Usage:
    from h3lab.exp import Experiment, record, load, pareto, render_perf_log
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(ROOT, "experiments.jsonl")

# The canonical target. Everything is judged against this.
TARGET_DURATION_S = 14.375   # 345 frames @24fps; H3 requires num_frames%17==5 and <=15s
TARGET_RES = "1344x768"
TARGET_FPS = 24
TARGET_FRAMES = 345


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=5
        ).stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


@dataclass
class Experiment:
    # --- identity -------------------------------------------------------
    exp_id: str
    hypothesis: str = ""
    notes: str = ""
    git_commit: str = field(default_factory=_git_commit)
    timestamp: float = field(default_factory=time.time)

    # --- model configuration -------------------------------------------
    checkpoint: str = ""            # exact HF repo id or local path
    lora: str | None = None
    lora_strength: float | None = None
    scheduler: str = ""
    num_steps: int | None = None     # denoising steps
    dit_evals: int | None = None     # actual DiT forwards (differs from steps under CFG)
    cfg_scale: float | None = None
    attention_method: str = ""       # dense_fa3 | vsa | sta | sdpa | ...
    attention_sparsity: float | None = None   # 0.0 = dense
    sparsity_schedule: str | None = None      # uniform | per-layer | per-timestep | adaptive
    precision: str = "bf16"
    vae_precision: str | None = None

    # --- systems configuration -----------------------------------------
    gpu_type: str = "H100"
    gpu_count: int = 8
    parallelism: str = ""           # e.g. "ulysses8" | "ring8" | "tp8"
    compile_mode: str | None = None
    cuda_graphs: bool = False

    # --- output spec ----------------------------------------------------
    resolution: str = TARGET_RES
    fps: int = TARGET_FPS
    frames: int = TARGET_FRAMES
    duration_s: float = field(init=False, default=0.0)
    audio: bool = False

    # --- measured latency (seconds, warm) -------------------------------
    e2e_latency_s: float | None = None
    text_encode_s: float | None = None
    dit_s: float | None = None
    attention_s: float | None = None
    mlp_s: float | None = None
    vae_s: float | None = None
    audio_s: float | None = None
    comm_s: float | None = None
    other_s: float | None = None
    latency_stddev_s: float | None = None
    n_timing_runs: int | None = None

    # --- measured utilization -------------------------------------------
    peak_vram_gb: float | None = None
    sm_utilization: float | None = None
    hbm_bw_utilization: float | None = None
    achieved_tflops: float | None = None

    # --- quality ---------------------------------------------------------
    # Reference-based (vs Base H3 at same prompt+seed) and absolute metrics.
    quality: dict[str, Any] = field(default_factory=dict)
    quality_class: str | None = None   # A=lossless B=tiny C=noticeable D=major
    human_eval: dict[str, Any] = field(default_factory=dict)

    status: str = "ok"   # ok | failed | oom | partial
    error: str | None = None

    def __post_init__(self):
        self.duration_s = self.frames / self.fps if self.fps else 0.0

    # -- derived ---------------------------------------------------------
    @property
    def realtime_factor(self) -> float | None:
        """<=1.0 means faster than realtime. This is the hard constraint."""
        if self.e2e_latency_s is None or not self.duration_s:
            return None
        return self.e2e_latency_s / self.duration_s

    @property
    def meets_constraint(self) -> bool:
        rf = self.realtime_factor
        return rf is not None and rf <= 1.0

    @property
    def quality_score(self) -> float | None:
        """Single scalar for Pareto math only. Never use it to make a final call --
        the goal explicitly warns against collapsing quality into one number too early."""
        q = self.quality.get("composite")
        return float(q) if q is not None else None

    def to_json(self) -> dict:
        d = asdict(self)
        d["realtime_factor"] = self.realtime_factor
        d["meets_constraint"] = self.meets_constraint
        return d


def record(exp: Experiment, db: str = DB) -> Experiment:
    with open(db, "a") as f:
        f.write(json.dumps(exp.to_json()) + "\n")
    return exp


def load(db: str = DB) -> list[dict]:
    if not os.path.exists(db):
        return []
    rows = []
    with open(db) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pareto(rows: list[dict]) -> list[dict]:
    """Non-dominated set on (latency down, quality up).

    A dominates B iff A is at least as fast AND at least as high quality,
    and strictly better on at least one.
    """
    pts = [r for r in rows
           if r.get("e2e_latency_s") is not None
           and (r.get("quality") or {}).get("composite") is not None
           and r.get("status") == "ok"]
    front = []
    for a in pts:
        qa, la = a["quality"]["composite"], a["e2e_latency_s"]
        dominated = False
        for b in pts:
            if b is a:
                continue
            qb, lb = b["quality"]["composite"], b["e2e_latency_s"]
            if lb <= la and qb >= qa and (lb < la or qb > qa):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda r: r["e2e_latency_s"])


def best_realtime(rows: list[dict]) -> dict | None:
    """Highest quality config that satisfies latency <= duration. THE answer."""
    ok = [r for r in rows
          if r.get("meets_constraint")
          and (r.get("quality") or {}).get("composite") is not None
          and r.get("status") == "ok"]
    return max(ok, key=lambda r: r["quality"]["composite"]) if ok else None


def highest_quality(rows: list[dict]) -> dict | None:
    ok = [r for r in rows if (r.get("quality") or {}).get("composite") is not None
          and r.get("status") == "ok"]
    return max(ok, key=lambda r: r["quality"]["composite"]) if ok else None


def fastest(rows: list[dict]) -> dict | None:
    ok = [r for r in rows if r.get("e2e_latency_s") is not None and r.get("status") == "ok"]
    return min(ok, key=lambda r: r["e2e_latency_s"]) if ok else None


def _fmt(v, n=2, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def summary_table(rows: list[dict]) -> str:
    front_ids = {r["exp_id"] for r in pareto(rows)}
    hdr = ("| Config | Steps | Attention | Sparsity | Precision | E2E (s) | RT factor | "
           "Quality | Class | Pareto | Notes |")
    sep = "|---|---:|---|---:|---|---:|---:|---:|:-:|:-:|---|"
    lines = [hdr, sep]
    for r in sorted(rows, key=lambda x: (x.get("e2e_latency_s") or 1e9)):
        q = (r.get("quality") or {}).get("composite")
        rf = r.get("realtime_factor")
        flag = "✅" if r.get("meets_constraint") else ("❌" if rf else "—")
        lines.append(
            f"| {r['exp_id']} | {_fmt(r.get('num_steps'),0)} | {r.get('attention_method') or '—'} "
            f"| {_fmt(r.get('attention_sparsity'),2)} | {r.get('precision')} "
            f"| {_fmt(r.get('e2e_latency_s'))} | {_fmt(rf)} {flag} | {_fmt(q,1)} "
            f"| {r.get('quality_class') or '—'} | {'★' if r['exp_id'] in front_ids else ''} "
            f"| {(r.get('notes') or '')[:60]} |"
        )
    return "\n".join(lines)
