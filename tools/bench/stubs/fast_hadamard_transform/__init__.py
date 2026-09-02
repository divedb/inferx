"""Stub for the fast_hadamard_transform wheel (CUDA extension, no wheel for
this platform). Import-time needs a callable; calling it raises, so the
TokenSpeed registry's triton fallback serves the hadamard benchmark."""
def hadamard_transform(*_args, **_kwargs):
    raise NotImplementedError("fast_hadamard_transform stub: not installed")
