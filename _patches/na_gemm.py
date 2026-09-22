"""Optional Neural-Accelerator bf16 GEMM via Metal Performance Primitives matmul2d.

Compiles a tiled `mpp::tensor_ops::matmul2d` kernel through torch.mps.compile_shader.
Takes bf16 operands and returns float32; `available()` and `self_check_ok()` gate use,
and any failure disables the backend.
"""

import torch

TAG = "[AppleSilicon-FP8/na_gemm]"

_BM, _BN, _BK, _NSG = 64, 64, 64, 4   # tile: BM, BN, BK, simdgroups

_GEMM_MSL = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BM  = @BM@;
constant constexpr int BN  = @BN@;
constant constexpr int BK  = @BK@;
constant constexpr int NSG = @NSG@;

kernel void gemm(
    device bfloat* A  [[buffer(0)]],   // [M,K] row-major
    device bfloat* B  [[buffer(1)]],   // [K,N] row-major (NN)
    device float*  C  [[buffer(2)]],   // [M,N] row-major
    device int*    SH [[buffer(3)]],   // [M,N,K]
    uint3 tgid [[threadgroup_position_in_grid]])
{
    const int M = SH[0], N = SH[1], K = SH[2];
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;
    if (m0 >= M || n0 >= N) return;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto mA0 = tensor(A + ulong(m0)*K, dextents<int,2>{BK, min(BM, M - m0)}, array<int,2>{1, K});
    auto mB0 = tensor(B + n0,          dextents<int,2>{min(BN, N - n0), BK}, array<int,2>{1, N});
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA0)>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(mB0)>;
    auto cC = op.get_destination_cooperative_tensor<AT, BT, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cC.get_capacity(); ++i)
        if (cC.is_valid_element(i)) cC[i] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int kk = min(BK, K - k0);
        auto mA = tensor(A + ulong(m0)*K + k0, dextents<int,2>{kk, min(BM, M - m0)}, array<int,2>{1, K});
        auto mB = tensor(B + ulong(k0)*N + n0, dextents<int,2>{min(BN, N - n0), kk}, array<int,2>{1, N});
        op.run(mA, mB, cC);
    }

    device float* Cb = C + ulong(m0)*N + n0;
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cC.get_capacity(); ++i) {
        if (!cC.is_valid_element(i)) continue;
        auto idx = cC.get_multidimensional_index(i);
        const int r = int(idx[1]), c = int(idx[0]);
        if (m0 + r >= M || n0 + c >= N) continue;
        Cb[ulong(r)*N + c] = cC[i];
    }
}
"""

_lib = None
_compiled = None        # None=untried, False=failed, True=ok
_self_check = None      # None=untried, then bool


def _get_lib():
    global _lib, _compiled
    if _compiled is not None:
        return _lib if _compiled else None
    if not hasattr(torch.mps, "compile_shader"):
        _compiled = False
        return None
    try:
        src = (_GEMM_MSL
               .replace("@BM@", str(_BM)).replace("@BN@", str(_BN))
               .replace("@BK@", str(_BK)).replace("@NSG@", str(_NSG)))
        _lib = torch.mps.compile_shader(src)
        _compiled = True
    except Exception as e:
        print(f"{TAG} matmul2d kernel did not compile; NA GEMM disabled ({e!r}).")
        _lib = None
        _compiled = False
    return _lib


def available():
    """True iff the matmul2d kernel compiles on this OS/SDK/PyTorch build."""
    return _get_lib() is not None


def reset_cache():
    """Test hook: drop the compile + numeric memos so the next call re-probes.

    _caps.reset_cache() delegates here, since its own memo reads through to this one.
    """
    global _lib, _compiled, _self_check
    _lib = None
    _compiled = None
    _self_check = None


def na_matmul(a, b):
    """C[M,N] f32 = A[M,K] @ B[K,N]; a,b are bf16, on MPS.

    Made contiguous here, so a LUT-decoded operand's inherited strides are fine.
    """
    lib = _get_lib()
    if lib is None:
        raise RuntimeError("NA matmul2d unavailable")
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dims disagree: {a.shape} @ {b.shape}"
    c = torch.zeros(M, N, device="mps", dtype=torch.float32)
    sh = torch.tensor([M, N, K], dtype=torch.int32, device="mps")
    ntg_x = -(-M // _BM)
    ntg_y = -(-N // _BN)
    # `threads` is the TOTAL thread count per dimension, so threadgroups per
    # dimension = threads / group_size: ntg_x over M tiles, ntg_y over N tiles
    lib.gemm(a, b, c, sh, threads=(ntg_x * 128, ntg_y, 1), group_size=(128, 1, 1))
    return c


def self_check_ok():
    """Cached one-time numeric check: NA result must match a@b within tolerance."""
    global _self_check
    if _self_check is not None:
        return _self_check
    if not available():
        _self_check = False
        return False
    try:
        # local generator, never torch.manual_seed: this runs at plugin import, and
        # reseeding there would move the host RNG under every later draw
        g = torch.Generator().manual_seed(0)
        a = (torch.randn(64, 256, generator=g) * 0.5).to(torch.bfloat16).to("mps").contiguous()
        b = (torch.randn(256, 96, generator=g) * 0.5).to(torch.bfloat16).to("mps").contiguous()
        ref = (a.float() @ b.float())
        out = na_matmul(a, b)
        rel = (out - ref).abs().max() / (ref.abs().max() + 1e-9)
        _self_check = bool(rel.item() < 5e-2)
        if not _self_check:
            print(f"{TAG} self-check failed (rel={rel.item():.4f}); NA GEMM disabled.")
    except Exception as e:
        print(f"{TAG} self-check raised; NA GEMM disabled ({e!r}).")
        _self_check = False
    return _self_check
