import subprocess
from types import SimpleNamespace

from gpuops.worker import cuda_readiness
from gpuops.worker.cuda_readiness import CudaReadiness

GPU = SimpleNamespace(index=0, uuid="GPU-0")


class Process:
    def __init__(self, *, stdout="", stderr="", code=0, hang=False):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = None if hang else code
        self.hang = hang
        self.killed = False

    def communicate(self, timeout):
        if self.hang:
            raise subprocess.TimeoutExpired("cuda_check", timeout)
        return self.stdout, self.stderr

    def kill(self):
        self.killed = True

    def poll(self):
        return self.returncode


def readiness(*processes):
    now = [0.0]
    launched = []
    queue = list(processes)

    def popen(command, **kwargs):
        assert command[-1] == kwargs["env"]["CUDA_VISIBLE_DEVICES"] == GPU.uuid
        launched.append(command)
        return queue.pop(0)

    return CudaReadiness(monotonic_clock=lambda: now[0], popen=popen), now, launched


def test_ready_gpu_is_cached():
    check, now, launched = readiness(Process(stdout="GPU-0\n"), Process(stdout="GPU-0\n"))

    assert check(GPU) and check(GPU)
    now[0] = cuda_readiness.READY_CACHE_SECONDS + 1
    assert check(GPU)
    assert len(launched) == 2


def test_failed_gpu_backs_off_before_retrying():
    check, now, launched = readiness(
        Process(code=1, stderr="cuInit failed with CUDA error 3"),
        Process(stdout="GPU-OTHER\n"),
        Process(stdout="GPU-0\n"),
    )

    assert not check(GPU)
    now[0] = 59
    assert not check(GPU)
    assert len(launched) == 1
    now[0] = 61
    assert not check(GPU)  # a different device identity is also a failure
    now[0] = 61 + 119
    assert not check(GPU)
    now[0] = 61 + 121
    assert check(GPU)
    assert len(launched) == 3


def test_stuck_check_is_not_repeated_while_it_is_alive():
    stuck = Process(hang=True)
    check, now, launched = readiness(stuck, Process(stdout="GPU-0\n"))

    assert not check(GPU)
    assert stuck.killed
    now[0] = 1000
    assert not check(GPU)
    assert len(launched) == 1
    stuck.returncode = -9
    assert check(GPU)
    assert len(launched) == 2
