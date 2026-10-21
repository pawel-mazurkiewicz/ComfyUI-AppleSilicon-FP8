"""NVFP4 dequantize shader: bit-exact against comfy_kitchen's eager result, CPU path otherwise."""
import sys

import pytest
import torch

from conftest import requires_mps

from _patches import _caps
from _patches import comfykitchen_fp8 as ck

try:
    import comfy_kitchen
    import comfy_kitchen.backends.eager.quantization as q
    from comfy_kitchen.float_utils import to_blocked
    # captured at import: collection finishes before any test installs the patch
    ORIG = q.dequantize_nvfp4
    _HAS_CK = ORIG.__module__ == q.__name__
except Exception:
    _HAS_CK = False

requires_ck = pytest.mark.skipif(not _HAS_CK, reason="needs an unpatched comfy_kitchen eager backend")
DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# (rows, 16-wide blocks per row): ragged against the 128-row and 4-block swizzle tiles
SHAPES = [(1, 1), (1, 4), (3, 5), (127, 4), (128, 4), (129, 5), (130, 6),
          (200, 37), (256, 8), (257, 9), (512, 64)]


def _bits(t):
    return t.view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def assert_exact(got, ref):
    """Bit for bit (signed zeros too); NaNs must sit in the same places."""
    assert got.dtype == ref.dtype and got.shape == ref.shape
    got, ref = got.cpu(), ref.cpu()
    nan = ref.isnan()
    assert torch.equal(got.isnan(), nan)
    assert torch.equal(_bits(got)[~nan], _bits(ref)[~nan])
    if not nan.any():
        assert torch.equal(got, ref)


def make(R, nb, seed=0, flatten=False, scale_bytes=None):
    g = torch.Generator().manual_seed(seed)
    qx = torch.randint(0, 256, (R, nb * 8), generator=g, dtype=torch.uint8)
    if scale_bytes is None:
        scale_bytes = torch.randint(0, 256, (R, nb), generator=g, dtype=torch.uint8)
    bs = to_blocked(scale_bytes, flatten=flatten).view(torch.float8_e4m3fn)
    return qx, bs


def mps(*ts):
    return tuple(t.to("mps") for t in ts)


@pytest.fixture(scope="module")
def lib():
    if not (torch.backends.mps.is_available() and _caps.has_compile_shader()):
        pytest.skip("needs torch.mps.compile_shader")
    return torch.mps.compile_shader(ck._NVFP4_MSL)


# --- the kernel itself ------------------------------------------------------


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("R,nb", SHAPES)
@requires_mps
@requires_ck
def test_shader_matches_eager_on_arbitrary_bytes(lib, R, nb, dt):
    """Random payload and random scale bytes, E4M3 NaN included."""
    qx, bs = make(R, nb, seed=R * 1000 + nb)
    ts = torch.tensor(0.37)
    for hi in (True, False):
        ref = ORIG(qx, ts, bs, dt, hi)
        got = ck._nvfp4_mps(lib, *mps(qx, ts), bs.to("mps"), dt, hi)
        assert got.device.type == "mps"
        assert_exact(got, ref)


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("R,nb", SHAPES)
@requires_mps
@requires_ck
def test_shader_matches_eager_on_real_quantized_weights(lib, R, nb, dt):
    """Valid NVFP4 from comfy_kitchen's own quantizer; no NaN, so plain torch.equal applies."""
    torch.manual_seed(R + nb)
    x = torch.randn(R, nb * 16) * 0.3
    ts = torch.tensor(float(x.abs().max()) / (6.0 * 448.0) + 1e-8)
    qx, bs = q.quantize_nvfp4(x, ts)
    ref = ORIG(qx, ts, bs, dt)
    got = ck._nvfp4_mps(lib, qx.to("mps"), ts.to("mps"), bs.to("mps"), dt, True)
    assert not ref.isnan().any()
    assert torch.equal(got.cpu(), ref)
    assert_exact(got, ref)


@pytest.mark.parametrize("ts_val", [0.37, 0.0, 1e-30, 448.0, 3e4])
@pytest.mark.parametrize("dt", DTYPES)
@requires_mps
@requires_ck
def test_every_payload_and_scale_byte(lib, dt, ts_val):
    """All 256 E4M3 scale bytes (subnormals, +-0, NaN) and all 256 payload bytes. A large
    tensor scale overflows fp16 to inf, so 0 * inf = NaN has to match too."""
    qx = (torch.arange(256 * 8) % 256).to(torch.uint8).view(256, 8)
    sb = torch.arange(256, dtype=torch.uint8).view(256, 1)
    bs = to_blocked(sb, flatten=False).view(torch.float8_e4m3fn)
    ts = torch.tensor(ts_val)
    for hi in (True, False):
        ref = ORIG(qx, ts, bs, dt, hi)
        got = ck._nvfp4_mps(lib, qx.to("mps"), ts.to("mps"), bs.to("mps"), dt, hi)
        assert_exact(got, ref)


@requires_mps
@requires_ck
def test_flat_and_2d_swizzle_agree(lib):
    qx, bs2 = make(130, 6, seed=3, flatten=False)
    _, bs1 = make(130, 6, seed=3, flatten=True)
    ts = torch.tensor(0.5)
    a = ck._nvfp4_mps(lib, qx.to("mps"), ts.to("mps"), bs2.to("mps"), torch.bfloat16, True)
    b = ck._nvfp4_mps(lib, qx.to("mps"), ts.to("mps"), bs1.to("mps"), torch.bfloat16, True)
    assert_exact(a, ORIG(qx, ts, bs2, torch.bfloat16, True))
    assert_exact(b, a)


@requires_mps
@requires_ck
def test_views_with_storage_offset_and_strides(lib):
    qx, bs = make(130, 6, seed=4, flatten=True)
    ts = torch.tensor(0.25)
    ref = ORIG(qx, ts, bs, torch.float16, True)

    # block scales at a nonzero storage offset
    pad = torch.zeros(37, dtype=torch.uint8)
    bs_off = torch.cat([pad, bs.view(torch.uint8)]).to("mps")[37:].view(torch.float8_e4m3fn)
    assert bs_off.storage_offset() == 37
    # payload at an offset, and as a column slice (not contiguous)
    qx_off = torch.cat([torch.zeros(5, qx.shape[1], dtype=torch.uint8), qx]).to("mps")[5:]
    qx_cols = torch.cat([qx, torch.zeros_like(qx)], dim=1).to("mps")[:, : qx.shape[1]]
    assert not qx_cols.is_contiguous()

    for qv in (qx_off, qx_cols):
        got = ck._nvfp4_mps(lib, qv, ts.to("mps"), bs_off, torch.float16, True)
        assert_exact(got, ref)


@requires_mps
@requires_ck
def test_tensor_scale_on_cpu_or_shaped_1(lib):
    qx, bs = make(64, 4, seed=5)
    ref = ORIG(qx, torch.tensor(0.7), bs, torch.bfloat16, True)
    for ts in (torch.tensor(0.7), torch.tensor([0.7])):  # CPU scalar, 1-element vector
        got = ck._nvfp4_mps(lib, qx.to("mps"), ts, bs.to("mps"), torch.bfloat16, True)
        assert_exact(got, ref)


@requires_mps
def test_shader_args_accepts_only_what_it_is_verified_on():
    qx = torch.zeros(130, 24, dtype=torch.uint8, device="mps")           # C = 48 -> 3 blocks
    bs = torch.zeros(2 * 1 * 512, dtype=torch.uint8, device="mps").view(torch.float8_e4m3fn)
    ts = torch.tensor(1.0, device="mps")
    ok = ck._nvfp4_shader_args(qx, ts, bs, torch.bfloat16, True)
    assert ok is not None

    assert ck._nvfp4_shader_args(qx.cpu(), ts, bs, torch.bfloat16, True) is None        # CPU payload
    assert ck._nvfp4_shader_args(qx, ts, bs.cpu(), torch.bfloat16, True) is None        # CPU scales
    assert ck._nvfp4_shader_args(qx.view(1, 130, 24), ts, bs, torch.bfloat16, True) is None
    assert ck._nvfp4_shader_args(qx.to(torch.int16), ts, bs, torch.bfloat16, True) is None
    assert ck._nvfp4_shader_args(qx, ts.bfloat16(), bs, torch.bfloat16, True) is None   # not fp32
    assert ck._nvfp4_shader_args(qx, ts.repeat(2), bs, torch.bfloat16, True) is None
    assert ck._nvfp4_shader_args(qx, ts, bs.view(torch.uint8), torch.bfloat16, True) is None
    assert ck._nvfp4_shader_args(qx, ts, bs[:-1], torch.bfloat16, True) is None         # mis-sized swizzle
    assert ck._nvfp4_shader_args(qx, ts, bs, torch.float64, True) is None
    assert ck._nvfp4_shader_args(qx[:, :23], ts, bs, torch.bfloat16, True) is None      # C % 16 != 0
    assert ck._nvfp4_shader_args(qx[:0], ts, bs, torch.bfloat16, True) is None          # empty


# --- dispatch, verification and fallback ------------------------------------


@pytest.fixture
def kernel_state(monkeypatch):
    """Start with this kernel untried, and put the process-wide memo back afterwards."""
    name = ck._NVFP4_KERNEL
    had = name in _caps._kernel_ready
    saved = _caps._kernel_ready.get(name)
    _caps._kernel_ready.pop(name, None)
    monkeypatch.setattr(ck, "_nvfp4_lib", ck._nvfp4_lib)
    yield name
    if had:
        _caps._kernel_ready[name] = saved
    else:
        _caps._kernel_ready.pop(name, None)


class Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *a, **k):
        self.calls += 1
        return self.fn(*a, **k)


@pytest.fixture
def dispatch(kernel_state, monkeypatch):
    """(dequantize_nvfp4 wrapper, shader spy, cpu-path spy) around the pristine eager fn."""
    shader = Spy(ck._nvfp4_mps)
    monkeypatch.setattr(ck, "_nvfp4_mps", shader)
    cpu = Spy(ck._cpu_dequant_on_mps(ORIG))
    return ck._nvfp4_dequant(ORIG, cpu), shader, cpu


def _call_args(R=130, nb=6, seed=7):
    qx, bs = make(R, nb, seed=seed)
    ts = torch.tensor(0.37)
    return (qx, ts, bs), mps(qx, ts, bs)


@requires_mps
@requires_ck
def test_mps_call_runs_in_the_shader_and_matches(dispatch):
    fn, shader, cpu = dispatch
    ref_args, mps_args = _call_args()
    for dt in DTYPES:
        for hi in (True, False):
            ref = ORIG(*ref_args, dt, hi)
            assert_exact(fn(*mps_args, dt, hi), ref)                        # positional
            assert_exact(fn(*mps_args, output_type=dt, hi_first=hi), ref)   # as the registry calls it
    assert cpu.calls == 0
    assert shader.calls > 0
    assert _caps._kernel_ready[ck._NVFP4_KERNEL] is True


@requires_mps
@requires_ck
def test_cpu_inputs_never_touch_the_shader(dispatch, kernel_state):
    fn, shader, cpu = dispatch
    ref_args, _ = _call_args()
    assert_exact(fn(*ref_args, torch.bfloat16, True), ORIG(*ref_args, torch.bfloat16, True))
    assert shader.calls == 0
    assert kernel_state not in _caps._kernel_ready      # not even verified


@requires_mps
@requires_ck
def test_fallback_when_shader_does_not_compile(dispatch, monkeypatch):
    fn, shader, cpu = dispatch
    attempts = []

    def boom(src):
        attempts.append(1)
        raise RuntimeError("no metal")

    monkeypatch.setattr(torch.mps, "compile_shader", boom)
    ref_args, mps_args = _call_args()
    ref = ORIG(*ref_args, torch.bfloat16, True)

    assert_exact(fn(*mps_args, torch.bfloat16, True), ref)
    assert_exact(fn(*mps_args, torch.bfloat16, True), ref)
    assert _caps._kernel_ready[ck._NVFP4_KERNEL] is False
    assert (shader.calls, cpu.calls) == (0, 2)
    assert len(attempts) == 1                       # latched: no rebuild per layer


@requires_mps
@requires_ck
def test_fallback_when_shader_fails_its_exact_check(dispatch, monkeypatch):
    fn, shader, cpu = dispatch
    real = shader.fn

    def off_by_one_bit(lib, qx, ts, bs, dt, hi):
        out = real(lib, qx, ts, bs, dt, hi)
        out.view(torch.int32 if dt == torch.float32 else torch.int16)[0, 1] ^= 1
        return out

    shader.fn = off_by_one_bit
    ref_args, mps_args = _call_args()
    ref = ORIG(*ref_args, torch.float16, True)

    got = fn(*mps_args, torch.float16, True)
    assert_exact(got, ref)
    assert _caps._kernel_ready[ck._NVFP4_KERNEL] is False
    assert cpu.calls == 1
    calls_after_verify = shader.calls
    fn(*mps_args, torch.float16, True)
    assert shader.calls == calls_after_verify       # never retried


@requires_mps
@requires_ck
def test_fallback_when_verification_raises(dispatch):
    fn, shader, cpu = dispatch

    def raiser(*a, **k):
        raise RuntimeError("device lost")

    shader.fn = raiser
    ref_args, mps_args = _call_args()
    assert_exact(fn(*mps_args, torch.bfloat16, True), ORIG(*ref_args, torch.bfloat16, True))
    assert _caps._kernel_ready[ck._NVFP4_KERNEL] is False
    assert cpu.calls == 1


@requires_mps
@requires_ck
def test_latches_off_after_a_failure_in_use(dispatch):
    fn, shader, cpu = dispatch
    ref_args, mps_args = _call_args()
    ref = ORIG(*ref_args, torch.bfloat16, True)

    assert_exact(fn(*mps_args, torch.bfloat16, True), ref)           # verifies, then runs the shader
    assert _caps._kernel_ready[ck._NVFP4_KERNEL] is True
    assert cpu.calls == 0

    def raiser(*a, **k):
        raise RuntimeError("encoder failed")

    shader.fn = raiser
    assert_exact(fn(*mps_args, torch.bfloat16, True), ref)           # no exception reaches the render
    assert _caps._kernel_ready[ck._NVFP4_KERNEL] is False
    assert cpu.calls == 1

    seen = shader.calls
    assert_exact(fn(*mps_args, torch.bfloat16, True), ref)
    assert shader.calls == seen                                      # latched
    assert cpu.calls == 2


@requires_mps
@requires_ck
def test_out_of_contract_calls_take_the_old_path(dispatch):
    fn, shader, cpu = dispatch
    (qx, ts, bs), _ = _call_args()

    # mis-sized swizzle: eager raises, and still does (the shader would read past the buffer)
    with pytest.raises((RuntimeError, ValueError)):
        fn(*mps(qx, ts, bs.view(torch.uint8)[:-512].view(torch.float8_e4m3fn)), torch.bfloat16, True)
    # a signature this wrapper does not know goes to eager untouched
    with pytest.raises(TypeError):
        fn(*mps(qx, ts, bs), torch.bfloat16, True, bogus=1)
    # a tensor scale that is not fp32 is outside the registry contract
    ref = ORIG(qx, ts.bfloat16(), bs, torch.bfloat16, True)
    assert_exact(fn(*mps(qx, ts.bfloat16(), bs), torch.bfloat16, True), ref)

    assert shader.calls == 0
    assert cpu.calls == 3
    assert ck._NVFP4_KERNEL not in _caps._kernel_ready


# --- install() wiring -------------------------------------------------------

_PATCHED_NAMES = (
    "dequantize_per_tensor_fp8", "quantize_per_tensor_fp8",
    "dequantize_nvfp4", "dequantize_mxfp8", "to_blocked",
)


@pytest.fixture
def ck_state():
    from comfy_kitchen.registry import registry
    import comfy_kitchen.float_utils as fumod

    mods = [registry._backends.get("eager"), q, comfy_kitchen, fumod]
    saved = [(m, n, getattr(m, n)) for m in mods if m is not None for n in _PATCHED_NAMES if hasattr(m, n)]
    was_installed = ck._installed
    yield
    for m, n, v in saved:
        setattr(m, n, v)
    ck._installed = was_installed


@requires_mps
@requires_ck
def test_install_routes_nvfp4_dequant_through_the_shader(ck_state, kernel_state, monkeypatch):
    from comfy_kitchen.registry import registry

    shader = Spy(ck._nvfp4_mps)
    monkeypatch.setattr(ck, "_nvfp4_mps", shader)
    ck._installed = False
    ck.install()

    eager = registry._backends.get("eager")
    ref_args, mps_args = _call_args(200, 37)
    ref = ORIG(*ref_args, torch.bfloat16, True)

    assert_exact(eager.dequantize_nvfp4(*mps_args, torch.bfloat16, True), ref)
    assert q.dequantize_nvfp4 is eager.dequantize_nvfp4
    # the public op resolves through the registry by getattr at call time
    assert_exact(comfy_kitchen.dequantize_nvfp4(*mps_args, torch.bfloat16, True), ref)
    assert _caps._kernel_ready[ck._NVFP4_KERNEL] is True
    assert shader.calls >= 3                        # the verification passes plus both calls


@requires_mps
@requires_ck
def test_install_without_compile_shader_keeps_the_cpu_path(ck_state, kernel_state, monkeypatch):
    from comfy_kitchen.registry import registry

    monkeypatch.setattr(_caps, "has_compile_shader", lambda: False)

    def must_not_wrap(*a, **k):
        raise AssertionError("shader wrapper installed without compile_shader")

    monkeypatch.setattr(ck, "_nvfp4_dequant", must_not_wrap)
    ck._installed = False
    ck.install()

    eager = registry._backends.get("eager")
    ref_args, mps_args = _call_args()
    out = eager.dequantize_nvfp4(*mps_args, torch.bfloat16, True)
    assert out.device.type == "mps"
    assert_exact(out, ORIG(*ref_args, torch.bfloat16, True))
    assert kernel_state not in _caps._kernel_ready


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple only")
def test_shader_source_has_all_three_output_kernels():
    for suffix in ck._NVFP4_SUFFIX.values():
        assert f"nvfp4_{suffix}" in ck._NVFP4_MSL
