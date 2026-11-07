"""Windows only: keep the machine from sleeping while a job runs.

Holds ES_CONTINUOUS | ES_SYSTEM_REQUIRED until the marker appears in the log file, or the
time limit passes (hours, default 3). Usage:
python scripts/queues/keep_awake.py cache/logs/phase5.log "bench done" [hours]
"""

import ctypes
import sys
import time
from pathlib import Path

log, marker = Path(sys.argv[1]), sys.argv[2]
hours = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
t0 = time.time()
while time.time() - t0 < hours * 3600:
    if log.exists() and marker in log.read_text(errors="ignore"):
        break
    time.sleep(30)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
