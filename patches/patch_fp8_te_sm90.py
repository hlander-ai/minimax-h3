"""Enable the blockwise-FP8 text encoder on Hopper (sm90).

Why
---
On 8xH100 80GB the pipeline is 15.62 s against a 14.375 s budget and the entire deficit
is one stage: the ~52 GB bf16 Qwen3-VL text encoder cannot stay resident, so it is
CPU-offloaded at 1.596 s/request against 0.038 s resident. Blockwise FP8 halves it to
~26 GB, which fits (26 + 8.3 FSDP-sharded DiT + 10.4 VAE = 44.7 GB, leaving ~35 GB for
activations). FastVideo rejects the checkpoint on sm90.

What is actually Blackwell-specific
-----------------------------------
Only the GEMM. Reading the module:
  * activation quantization  -> `_h3_per_token_group_quant_fp8_row_major`, a TRITON
                                kernel, architecture-agnostic
  * the matmul               -> `flashinfer.gemm.gemm_fp8_nt_groupwise`, Blackwell-only
So the gate is broader than the actual limitation.

The substitution
----------------
Here FP8 buys MEMORY, not speed: the text encoder runs once per request and costs 0.038 s
resident. So on sm90 we dequantize each weight back to bf16 one layer at a time and use a
normal matmul. Peak extra memory is one layer's weight (~0.26 GB for the largest), the
weights still LIVE as fp8 (~26 GB), and the dequant is memory-bound -- roughly 0.1 ms per
linear, ~50 ms across all 448. Against a 1.558 s offload stall that is a trade worth making.

This is numerically the dequantization the FlashInfer path performs internally, so it is
an exact-arithmetic substitution rather than an approximation on top of the quantization.
"""
import sys

F = "/opt/FastVideo/fastvideo/models/encoders/minimax_h3_checkpoint_fp8.py"

# There are THREE capability gates, not one: get_min_capability() feeds a check in
# validate_runtime AND another in apply(), on top of the explicit (10, 12) test.
# Lowering the floor to sm90 is the single change that clears the first two; the
# (10, 12) test and the apply() dispatch are handled separately below.
MINCAP_OLD = """    def get_min_capability(cls) -> int:
        return 100"""
MINCAP_NEW = """    def get_min_capability(cls) -> int:
        # h3lab: Hopper is served by the dequantize-then-bf16-matmul path in apply().
        return 90"""

RUNTIME_OLD = '''        if capability[0] not in (10, 12):
            raise RuntimeError("MiniMax-H3 serialized blockwise FP8 currently adapts SGLang's Blackwell "
                               f"FlashInfer path; got unsupported sm{capability_number}")'''
RUNTIME_NEW = '''        if capability[0] not in (9, 10, 12):
            raise RuntimeError("MiniMax-H3 serialized blockwise FP8 currently adapts SGLang's Blackwell "
                               f"FlashInfer path; got unsupported sm{capability_number}")
        # h3lab: sm90 (Hopper) is served by the dequantize-then-bf16-matmul path in
        # apply(). Only the FlashInfer GEMM is Blackwell-specific; the activation
        # quantization is Triton and runs anywhere.'''

APPLY_OLD = '''        capability = torch.cuda.get_device_capability(x.device)
        capability_number = capability[0] * 10 + capability[1]
        if capability_number < MiniMaxH3SerializedFP8Config.get_min_capability():
            raise RuntimeError("MiniMax-H3 serialized blockwise FP8 requires GPU capability "
                               f"sm{MiniMaxH3SerializedFP8Config.get_min_capability()} or newer, "
                               f"got sm{capability_number}")

        if not x.is_contiguous():
            x = x.contiguous()
        return _flashinfer_gemm_w8a8_block_fp8_linear_with_fallback('''
APPLY_NEW = '''        capability = torch.cuda.get_device_capability(x.device)
        capability_number = capability[0] * 10 + capability[1]

        # h3lab: Hopper path. FP8 is used here for RESIDENCY, not throughput -- this
        # encoder runs once per request and costs 0.038 s resident against 1.596 s
        # CPU-offloaded. Dequantize each weight to bf16 one layer at a time and use a
        # normal matmul; the weights stay fp8 in memory, which is the whole point.
        if capability[0] == 9:
            if not x.is_contiguous():
                x = x.contiguous()
            w = layer.weight
            scale = layer.weight_scale_inv
            bn, bk = self.weight_block_size
            n, k = w.shape
            pn = (bn - n % bn) % bn
            pk = (bk - k % bk) % bk
            wb = torch.nn.functional.pad(w.to(x.dtype), (0, pk, 0, pn)) if (pn or pk) \\
                else w.to(x.dtype)
            N, K = wb.shape
            wb = wb.view(N // bn, bn, K // bk, bk)
            wb = wb * scale.to(wb.dtype)[:, None, :, None]
            wb = wb.view(N, K)[:n, :k]
            return torch.nn.functional.linear(x, wb, bias)

        if capability_number < MiniMaxH3SerializedFP8Config.get_min_capability():
            raise RuntimeError("MiniMax-H3 serialized blockwise FP8 requires GPU capability "
                               f"sm{MiniMaxH3SerializedFP8Config.get_min_capability()} or newer, "
                               f"got sm{capability_number}")

        if not x.is_contiguous():
            x = x.contiguous()
        return _flashinfer_gemm_w8a8_block_fp8_linear_with_fallback('''


def sub(s, old, new, name, marker):
    if marker in s:
        print(f"{name}_ALREADY"); return s, True
    if old not in s:
        print(f"{name}_ANCHOR_MISSING", file=sys.stderr); return s, False
    print(f"{name}_PATCHED"); return s.replace(old, new, 1), True


def main():
    with open(F) as f:
        s = f.read()
    ok = True
    s, o0 = sub(s, MINCAP_OLD, MINCAP_NEW, "FP8TE_MINCAP", "return 90")
    s, o1 = sub(s, RUNTIME_OLD, RUNTIME_NEW, "FP8TE_RUNTIME", "capability[0] not in (9, 10, 12)")
    s, o2 = sub(s, APPLY_OLD, APPLY_NEW, "FP8TE_APPLY", "h3lab: Hopper path")
    ok = o0 and o1 and o2
    if not ok:
        return 1
    with open(F, "w") as f:
        f.write(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
