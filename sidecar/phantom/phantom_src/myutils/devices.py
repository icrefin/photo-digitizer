"""Minimal device helper for the Phantom sidecar (MPS / CUDA / CPU).

Replaces dreamoving/Phantom's myutils/devices.py, which is an
AUTOMATIC1111 excerpt that fails to import on macOS (`from modules
import mac_specific`). Only the members vaehook.py actually uses are
implemented: `device`, `torch_gc`, `get_optimal_device`,
`test_for_nans`.
"""

import sys

import torch


def has_mps() -> bool:
    return sys.platform == "darwin" and torch.backends.mps.is_available()


def get_optimal_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if has_mps():
        return "mps"
    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


device = get_optimal_device()


class NansException(Exception):
    pass


def torch_gc() -> None:
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def test_for_nans(x: torch.Tensor, where: str) -> None:
    if not torch.all(torch.isnan(x)).item():
        return
    raise NansException(f"A tensor with all NaNs was produced in {where}.")
