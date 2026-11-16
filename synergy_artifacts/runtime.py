"""Process peak resident memory without an additional runtime dependency."""
import os
import sys


def peak_memory_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ["peak_working_set", "working_set", "peak_paged", "paged",
                                                   "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile"]]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.peak_working_set)
        return None
    import resource
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)
