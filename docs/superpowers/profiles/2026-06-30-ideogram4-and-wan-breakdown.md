# Per-op GPU-time breakdown — 2026-06-30

Profiles collected with `ASFP8_PROFILE=1 ASFP8_PROFILE_INTERVAL=20` after the G1
seam additions (activation + rotary seams). Each run serializes the GPU; wall time is
~2-4× longer than normal; the *relative shares* are what matter for prioritization.

---

## Task 0 probe result

nn.SiLU dispatches through F.silu: [FILL: YES / NO]
nn.GELU dispatches through F.gelu: [FILL: YES / NO]
Additional wraps applied (if bypass detected): [FILL: none / nn.SiLU.forward + nn.GELU.forward]

---

## Run 1 — Ideogram 4 (int8 convrot, ASFP8_INT8_EXT=1)

**Config:** [FILL: resolution, steps, sampler, ComfyUI version, mtlflashattn ON/OFF]

**Launch command:**
```
ASFP8_PROFILE=1 ASFP8_PROFILE_INTERVAL=20 ASFP8_INT8_EXT=1 \
  python main.py --listen --port 8188 2>&1 | tee /tmp/ideogram4_profile.log
```

**Profiler output (paste the final cumulative dump from terminal):**
```
[FILL: paste [AppleSilicon-FP8/mps_profile] GPU-time breakdown block here]
[Label: phase = denoising, step N of M, or "full run if no crash"]
```

**Parsed table:**

| Op bucket     | Time (s) | Share (%) | Calls | ms/call |
|---------------|----------|-----------|-------|---------|
| attention     | [FILL]   | [FILL]    |       |         |
| linear        | [FILL]   | [FILL]    |       |         |
| conv2d        | [FILL]   | [FILL]    |       |         |
| conv3d        | [FILL]   | [FILL]    |       |         |
| rmsnorm       | [FILL]   | [FILL]    |       |         |
| layernorm     | [FILL]   | [FILL]    |       |         |
| activation    | [FILL]   | [FILL]    |       |         |
| rotary        | [FILL]   | [FILL]    |       |         |
| matmul        | [FILL]   | [FILL]    |       |         |
| bmm           | [FILL]   | [FILL]    |       |         |
| gguf_dequant  | [FILL]   | [FILL]    |       |         |
| \<unwrapped\> | [FILL]   | [FILL]    |  n/a  |   n/a   |

**COMMIT_AND_WAIT hypothesis:**
- Dominant linear shape (M, K, N): [FILL]
- ASFP8_INT8_EXT active: [FILL: YES/NO]
- linear ms/call observed: [FILL]
- Hypothesis: if ms/call > 2ms for M≤64, COMMIT_AND_WAIT at int8_matmul2d_tuned.mm:117 may be firing. Requires per-shape confirmation before acting.

---

## Run 2 — Wan 2.2 720p (4–6 steps)

> NOTE: Wan 2.2 visual output may be incorrect (separate correctness issue).
> The time distribution is valid regardless of output quality.
> Confirm the run completes without Python exception before trusting these numbers.
> If a crash occurs, label the dump with the phase/step reached (see Step C2).

**Config:** [FILL: resolution=720p, steps, sampler, model variant, mtlflashattn ON/OFF]

**Launch command:**
```
ASFP8_PROFILE=1 ASFP8_PROFILE_INTERVAL=20 \
  python main.py --listen --port 8188 2>&1 | tee /tmp/wan_profile.log
```

**Profiler output (paste the final cumulative dump):**
```
[FILL]
[Label: phase = denoising step N of M / crashed after step N / full run]
```

**Parsed table:**

| Op bucket     | Time (s) | Share (%) | Calls | ms/call |
|---------------|----------|-----------|-------|---------|
| attention     | [FILL]   | [FILL]    |       |         |
| linear        | [FILL]   | [FILL]    |       |         |
| conv2d        | [FILL]   | [FILL]    |       |         |
| conv3d        | [FILL]   | [FILL]    |       |         |
| rmsnorm       | [FILL]   | [FILL]    |       |         |
| layernorm     | [FILL]   | [FILL]    |       |         |
| activation    | [FILL]   | [FILL]    |       |         |
| rotary        | [FILL]   | [FILL]    |       |         |
| matmul        | [FILL]   | [FILL]    |       |         |
| bmm           | [FILL]   | [FILL]    |       |         |
| gguf_dequant  | [FILL]   | [FILL]    |       |         |
| \<unwrapped\> | [FILL]   | [FILL]    |  n/a  |   n/a   |

---

## Diagnosis checklist

Before reading the ranking:
- [ ] `rotary` shows > 0% for Wan (it must: Wan uses rope_apply / apply_rope_comfy1 etc.)
- [ ] `rotary` for ideogram4: if 0%, confirm `_ideogram4_apply_rope_lowp` was the active path (check KJNodes node used in workflow); 0% may be correct if that node was not used
- [ ] `activation` > 0% for both models (non-zero confirms F.silu/gelu/glu wrap fired)
- [ ] If `activation` ≈ 0% but `<unwrapped>` is large, Task 0 probe likely showed bypass — add nn.SiLU.forward/nn.GELU.forward wraps and re-run
- [ ] Linear ms/call vs shape hypothesis documented (COMMIT_AND_WAIT section above)
- [ ] Partial dumps (if run crashed) are labeled with phase/step and only used for ranking if at least one full denoising step completed

---

## Combined ranking and priority override

[FILL after both tables are complete]

Priority ranking by GPU-time share (combined ideogram4 + Wan, descending):
1. [FILL] — X%  →  [roadmap item]
2. [FILL] — X%  →  [roadmap item]
3. ...

**Override verdict:** [Does this ranking agree with the default A→B→C→D→E roadmap order?
List any items whose priority changes. Example: "C (rotary) should move ahead of B (conv)
because rotary is 18% vs conv 4% in Wan."]
