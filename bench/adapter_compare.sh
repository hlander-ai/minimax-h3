#!/usr/bin/env bash
# Compare the three released VSA adapters at IDENTICAL cost (4 forwards, 90% sparse).
#
# This is the only remaining quality lever that does not touch latency: all three are
# rank-64 adapters on the same base DiT at the same sparsity, so they generate in the same
# time. Two were distilled data-free, two on teacher-generated ("synthetic") video at
# different checkpoint steps -- which makes this a training-free test of the mission's
# explicit question, "do NOT assume data-free training is optimal".
#
# Run at bf16 + FSDP (not the deployed FP8 config) because FastVideo's FP8 path and its
# runtime-LoRA loading are mutually exclusive. Latency here is irrelevant -- adapter choice
# does not change it. If a synthetic adapter wins, we pull its full merged checkpoint
# (148 GB) and re-measure in the deployed FP8 configuration.
set -uo pipefail
export MODAL_PROFILE="${MODAL_PROFILE:?set MODAL_PROFILE to your Modal workspace}"

cat > /tmp/sweep_payload.json <<'JSON'
{"prompts": [
  {"id":"parkour","text":"A traceur sprints across a rooftop and vaults a concrete ledge, tucking into a roll and rising into a run. Fast tracking shot follows alongside. Late afternoon sun, long shadows, dust kicked up on landing, city skyline behind."},
  {"id":"texture","text":"Macro shot of sunlight moving across a woven wool blanket as a cloud passes. Individual fibers and slubs visible, the weave pattern crisp, subtle color shift from warm to cool and back."}],
 "configs": [{"sparsity":0.90,"dense_first_n":0,"label":"ADAPTER"}]}
JSON

for A in vsa-datafree vsa-synthetic-step1300 vsa-synthetic-step1900; do
  echo "=== adapter: $A ==="
  sed -i.bak "s/\"label\": *\"[^\"]*\"/\"label\": \"adapter_${A}\"/" /tmp/sweep_payload.json
  modal run modal/h3_bench.py::sweep_main \
    --model-path /weights/base --lora "$A" --quant "" \
    --backend VIDEO_SPARSE_ATTN_H3 \
    --offload-te 0 --offload-vae 0 --fsdp 1 --repeats 1 \
    --out "results/adapter_${A}.json" > "results/logs/adapter_${A}.log" 2>&1
  echo "  exit=$? -> results/adapter_${A}.json"
done
