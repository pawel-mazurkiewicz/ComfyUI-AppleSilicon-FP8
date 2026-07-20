"""Pytest for the G2 matmul2d dtype probe.

The probe's table generation is the unit under test. We do NOT skip the suite when a
candidate fails to compile — a candidate compile failure is a recorded capability finding,
asserted as a structured row. The ONLY legitimate skips are: no MPS device, or no Xcode/
ninja toolchain. signed_char is the control + spy: it must compile, run, and match the
int32 reference EXACTLY (a silent fallback would leave C all-zero and fail this).
"""
import pytest
import torch

import importlib.util, pathlib
_probe_path = pathlib.Path(__file__).parent.parent / "dev" / "probe_matmul2d_dtypes.py"
_spec = importlib.util.spec_from_file_location("probe_matmul2d_dtypes", _probe_path)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)

import shutil
from _patches import _caps
_NO_MPS = not torch.backends.mps.is_available()
_NO_TOOLCHAIN = shutil.which("xcrun") is None

pytestmark = [
    pytest.mark.skipif(_NO_MPS, reason="needs MPS device"),
    pytest.mark.skipif(_NO_TOOLCHAIN, reason="needs Xcode CLI tools (xcrun)"),
    # Match the docstring: a missing ninja toolchain is a legitimate skip, not a build
    # failure (build_extension() would return None and fail the probe_mod assertion).
    pytest.mark.skipif(not _caps.ninja_available(), reason="needs ninja to build the ObjC++ extension"),
]


@pytest.fixture(scope="module")
def probe_mod():
    mod = _probe.build_extension()
    # Extension build failure is a TEST FAILURE (toolchain/MPS already skip above).
    assert mod is not None, "ObjC++ host extension failed to build (see stderr)"
    return mod


def test_signed_char_control_runs_exact(probe_mod):
    """SPY/CONTROL: real int8 kernel must run and match int32 reference exactly."""
    r = _probe.run_probe(probe_mod, "signed_char")
    assert r["compile_ok"], f"signed_char compile failed: {r['detail']}"
    assert r["has_fn"], "signed_char kernel absent — host/template broken"
    assert r["run_ok"], f"signed_char run failed: {r['detail']}"
    assert r["exact"], f"signed_char not bit-exact (kernel did not really run?): {r['detail']}"
    assert r["verdict"] == "PASS", r["detail"]


@pytest.mark.parametrize("dtype", _probe.CANDIDATES)
def test_every_candidate_produces_structured_row(probe_mod, dtype):
    """Every candidate yields a structured row; verdict is one of the known states."""
    r = _probe.run_probe(probe_mod, dtype)
    assert r["verdict"] in {
        "PASS", "FAIL", "COMPILE_FAIL", "COMPILE_SKIP", "RUN_FAIL"
    }, f"{dtype}: unexpected verdict {r['verdict']!r}"
    # If it ran, the spy already enforced C != all-zero inside run_probe before PASS.
    if r["verdict"] == "PASS":
        assert r["run_ok"] and r["has_fn"] and r["compile_ok"]
    # A FAIL/COMPILE_FAIL/RUN_FAIL must carry a non-empty diagnostic.
    if r["verdict"] in {"FAIL", "COMPILE_FAIL", "RUN_FAIL"}:
        assert r["detail"], f"{dtype}: {r['verdict']} with no detail"


def test_table_generation_covers_all_candidates(probe_mod):
    """The INVESTIGATION_FACTS section must contain a row for every candidate."""
    rows = [{"name": d, "res": _probe.run_probe(probe_mod, d)} for d in _probe.CANDIDATES]
    section = _probe.build_section(rows)
    for d in _probe.CANDIDATES:
        assert f"| {d:<12} |" in section, f"missing row for {d}"
    # W4A8 must NOT be claimed from this matrix (Codex MAJOR 6).
    assert "NOT W4A8" in section
