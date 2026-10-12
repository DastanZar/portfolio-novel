"""epi-norm Triton kernel attempt (guarded: safe on CPU-only machines).

If `triton` + `torch` + CUDA are all available, `triton_rmsnorm_fwd` runs a
real fused Triton RMSNorm forward. Otherwise it falls back to the numpy
reference in `core.py` and reports why. Importing this module must NEVER
crash on a machine without Triton.
"""
from __future__ import annotations

try:
    import torch  # type: ignore

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    HAS_TORCH = False

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    HAS_TRITON = False

_TRITON_REASON = []
if not HAS_TORCH:
    _TRITON_REASON.append("torch not available")
if not HAS_TRITON:
    _TRITON_REASON.append("triton not available")


def _define_kernel():
    if not (HAS_TORCH and HAS_TRITON):
        return None
    try:

        @triton.jit
        def _rmsnorm_fwd_kernel(
            X_ptr, W_ptr, Y_ptr, R_ptr,
            N, D,
            stride_xm, stride_xd,
            stride_ym, stride_yd,
            BLOCK: tl.constexpr,
            EPS: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offs = tl.arange(0, BLOCK)
            mask = offs < D
            x = tl.load(X_ptr + pid * stride_xm + offs * stride_xd, mask=mask, other=0.0)
            w = tl.load(W_ptr + offs, mask=mask, other=1.0)
            ms = tl.sum(x * x, axis=0) / D
            rr = 1.0 / tl.sqrt(ms + EPS)
            y = x * rr * w
            tl.store(Y_ptr + pid * stride_ym + offs * stride_yd, y, mask=mask)
            tl.store(R_ptr + pid, rr)

        return _rmsnorm_fwd_kernel
    except Exception as e:  # pragma: no cover
        _TRITON_REASON.append(f"kernel define failed: {e}")
        return None


_RMSNORM_KERNEL = _define_kernel()


def triton_rmsnorm_fwd(x, weight, eps: float = 1e-6):
    """Attempt Triton RMSNorm forward; fall back to numpy/torch ref.

    Returns (y, rrms, info_dict). info_dict has backend in
    {"triton", "torch-ref", "numpy-ref"} and a reason string.
    """
    import numpy as np

    if _RMSNORM_KERNEL is not None and HAS_TORCH:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available; Triton needs GPU")
            xt = torch.as_tensor(np.asarray(x, dtype=np.float32), device="cuda")
            wt = torch.as_tensor(np.asarray(weight, dtype=np.float32), device="cuda")
            N, D = xt.shape[0], xt.shape[1]
            yt = torch.empty_like(xt)
            rt = torch.empty((N,), device="cuda", dtype=torch.float32)
            BLOCK = 1
            while BLOCK * 2 <= max(D, 1) and BLOCK * 2 <= 1024:
                BLOCK *= 2
            grid = (N,)
            _RMSNORM_KERNEL[grid](
                xt, wt, yt, rt, N, D,
                xt.stride(0), xt.stride(1), yt.stride(0), yt.stride(1),
                BLOCK=BLOCK, EPS=float(eps),
            )
            y = yt.cpu().numpy().astype(np.float64)
            rr = rt.cpu().numpy().astype(np.float64)[:, None]
            return y, rr, {"backend": "triton", "reason": "ok"}
        except Exception as e:
            pass  # fall through to reference
            fallback_reason = f"triton attempt failed ({e}); using reference"
    else:
        fallback_reason = "; ".join(_TRITON_REASON) + "; using reference"

    # Reference path (torch if available else numpy)
    if HAS_TORCH:
        try:
            import torch.nn.functional as F  # noqa

            xt = torch.as_tensor(np.asarray(x, dtype=np.float64))
            wt = torch.as_tensor(np.asarray(weight, dtype=np.float64))
            ms = xt.pow(2).mean(dim=-1, keepdim=True)
            rr = 1.0 / torch.sqrt(ms + eps)
            y = (xt * rr * wt).numpy()
            return y, rr.numpy(), {"backend": "torch-ref", "reason": fallback_reason}
        except Exception:
            pass
    from .core import rmsnorm_ref

    y, rr = rmsnorm_ref(np.asarray(x), np.asarray(weight), eps)
    return y, rr, {"backend": "numpy-ref", "reason": fallback_reason}


def kernel_status() -> dict:
    """Report Triton/torch availability (for demos and receipts)."""
    return {
        "has_triton": bool(HAS_TRITON),
        "has_torch": bool(HAS_TORCH),
        "reason": "; ".join(_TRITON_REASON) if _TRITON_REASON else "ok",
        "has_kernel": _RMSNORM_KERNEL is not None,
    }
