"""Windows only: keep the machine from sleeping while a job runs.

Holds ES_CONTINUOUS | ES_SYSTEM_REQUIRED until the marker appears in the log file, or three
hours pass. Usage: python scripts/queues/keep_awake.py cache/logs/phase5.log "bench done"
"""

import ctypes
import sys
import time
from pathlib import Path

log, marker = Path(sys.argv[1]), sys.argv[2]
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
t0 = time.time()
while time.time() - t0 < 3 * 3600:
    if log.exists() and marker in log.read_text(errors="ignore"):
        break
    time.sleep(30)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
