from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from gpuops.worker import cuda_check

UUID = "GPU-00112233-4455-6677-8899-aabbccddeeff"


class CudaFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def fake_cuda_library(*, init_code=0, count=1, uuid_hex="00112233445566778899aabbccddeeff"):
    def device_count(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = count
        return 0

    def get_device(pointer, ordinal):
        assert ordinal == 0
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = 7
        return 0

    def get_uuid(pointer, device):
        assert device == 7
        value = ctypes.cast(pointer, ctypes.POINTER(cuda_check._CudaUuid)).contents
        value.bytes[:] = bytes.fromhex(uuid_hex)
        return 0

    return SimpleNamespace(
        cuInit=CudaFunction(lambda flags: init_code),
        cuDeviceGetCount=CudaFunction(device_count),
        cuDeviceGet=CudaFunction(get_device),
        cuDeviceGetUuid_v2=CudaFunction(get_uuid),
    )


def test_cuda_child_verifies_visible_physical_uuid(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    monkeypatch.setattr(cuda_check.ctypes, "CDLL", lambda _: fake_cuda_library())

    assert cuda_check.verify_cuda_device(UUID) == UUID


@pytest.mark.parametrize(
    ("library", "message"),
    [
        (fake_cuda_library(init_code=3), "cuInit failed with CUDA error 3"),
        (fake_cuda_library(count=0), "CUDA exposed 0 devices"),
        (
            fake_cuda_library(uuid_hex="10112233445566778899aabbccddeeff"),
            "did not match",
        ),
    ],
)
def test_cuda_child_rejects_unready_or_wrong_gpu(monkeypatch, library, message) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    monkeypatch.setattr(cuda_check.ctypes, "CDLL", lambda _: library)

    with pytest.raises(RuntimeError, match=message):
        cuda_check.verify_cuda_device(UUID)


def test_cuda_child_rejects_unscoped_visibility(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(RuntimeError, match="does not match"):
        cuda_check.verify_cuda_device(UUID)
