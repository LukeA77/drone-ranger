"""Describe the compute environment for `results/env.json` ({{COMPUTE_PROFILE}})."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

PACKAGES = ("numpy", "scipy", "pyyaml", "h5py", "pillow", "typer", "pytest", "ruff")


def _cpu_name() -> str:
    """Best-effort CPU model string; falls back to platform.processor()."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        elif sys.platform == "linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _memory_bytes() -> int | None:
    """Total physical memory in bytes, or None if it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MemoryStatus()
            stat.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return None


def _gpus() -> list[str]:
    """GPU names reported by nvidia-smi, or an empty list."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def env_report() -> dict[str, Any]:
    """Collect CPU, memory, GPU, OS and library versions."""
    mem = _memory_bytes()
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu_model": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "memory_gb": round(mem / 2**30, 2) if mem else None,
        "gpus": _gpus(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": _versions(),
    }
