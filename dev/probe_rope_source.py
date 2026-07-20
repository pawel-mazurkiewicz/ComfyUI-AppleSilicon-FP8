# dev/probe_rope_source.py — confirm the slow-path hypothesis from the installed source.
import inspect, comfy_kitchen.backends.eager.rope as r
src = inspect.getsource(r)
assert "view_as_complex" not in src and "polar" not in src, \
    "hypothesis wrong: eager DOES use complex multiply — revisit the kernel design"
assert "addcmul_" in src and "movedim" in src, "eager source changed; re-read before building"
print("CONFIRMED: eager rope is real-2x2-matrix; interleaved uses reshape+mul+addcmul_+cast,")
print("           split-half uses reshape+movedim+mul+mul+add+movedim+cast (multi-launch).")
print("apply_rope1 src:\n", inspect.getsource(r.apply_rope1))
print("apply_rope_split_half1 src:\n", inspect.getsource(r.apply_rope_split_half1))
