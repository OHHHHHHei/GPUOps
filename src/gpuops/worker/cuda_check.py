"""Isolated CUDA readiness check for one UUID-scoped GPU."""

from __future__ import annotations

import ctypes
import os
import sys


class _CudaUuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


def _call(function: object, name: str, *args: object) -> None:
    result = function(*args)  # type: ignore[operator]
    if result != 0:
        raise RuntimeError(f"{name} failed with CUDA error {result}")


def verify_cuda_device(expected_uuid: str) -> str:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match the requested GPU")
    library = ctypes.CDLL("libcuda.so.1")
    library.cuInit.argtypes = [ctypes.c_uint]
    library.cuInit.restype = ctypes.c_int
    library.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.cuDeviceGetCount.restype = ctypes.c_int
    library.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    library.cuDeviceGet.restype = ctypes.c_int
    get_uuid = getattr(library, "cuDeviceGetUuid_v2", None)
    if get_uuid is None:
        get_uuid = library.cuDeviceGetUuid
    get_uuid.argtypes = [ctypes.POINTER(_CudaUuid), ctypes.c_int]
    get_uuid.restype = ctypes.c_int

    _call(library.cuInit, "cuInit", 0)
    count = ctypes.c_int()
    _call(library.cuDeviceGetCount, "cuDeviceGetCount", ctypes.byref(count))
    if count.value != 1:
        raise RuntimeError(f"CUDA exposed {count.value} devices; expected one")
    device = ctypes.c_int()
    _call(library.cuDeviceGet, "cuDeviceGet", ctypes.byref(device), 0)
    uuid = _CudaUuid()
    _call(get_uuid, "cuDeviceGetUuid", ctypes.byref(uuid), device.value)
    hexadecimal = bytes(uuid.bytes).hex()
    observed_uuid = (
        f"GPU-{hexadecimal[:8]}-{hexadecimal[8:12]}-{hexadecimal[12:16]}-"
        f"{hexadecimal[16:20]}-{hexadecimal[20:]}"
    )
    if observed_uuid.casefold() != expected_uuid.casefold():
        raise RuntimeError(f"CUDA device UUID {observed_uuid} did not match {expected_uuid}")
    return expected_uuid


def main() -> int:
    if len(sys.argv) != 2:
        print("expected one GPU UUID", file=sys.stderr)
        return 2
    try:
        print(verify_cuda_device(sys.argv[1]))
    except (OSError, AttributeError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
