# Running on ARM64 and a Raspberry Pi 5

The package needs only its own pinned environment (`requirements-lock.txt`): NumPy, SciPy, OpenCV, ONNX Runtime and a few small libraries, with no PyTorch and no AGPL code. Every one of those packages publishes a ready-made Linux ARM64 wheel for Python 3.12 and 3.13 (checked by downloading the ARM64 wheel of each pinned version, D-726), so the same lockfile installs on a Raspberry Pi 5.

## The container

`docker/Dockerfile` builds the runtime image the same way for x86-64 and ARM64. Model weights and frames are not in the image; they are mounted at run time.

```
make containers          # aerialrange:amd64 and aerialrange:arm64 (Docker with buildx)
docker run --rm --platform linux/arm64 aerialrange:arm64 aerialrange demo --out /tmp/demo
```

On an x86 machine the ARM64 image runs under QEMU emulation. That proves the pipeline installs and runs on ARM64 and gives the same outputs; it says nothing about speed. CI builds the ARM64 image and runs the synthetic demo in it on every push (`.github/workflows/ci.yml`).

## Same outputs on both

```
make platform-check FRAMES=<folder of JPEG frames>
```

This runs `scripts/platform_check.py` in both images on the same frames and the model `configs/run.yaml` names, then compares them. Each run covers the whole pipeline twice: on the synthetic demo flight (tracking, ranging, the Kalman filter bank, outputs), and with the ONNX detector on the first 60 real frames. Detections and output rows are matched frame by frame (same class, IoU of at least 0.5). The check passes if, on each side, at least 99 % of detections and rows are matched; matched boxes differ by at most 0.5 px and scores by at most 0.005; at least 99 % of matched rows keep a one-to-one mapping of track ids; and matched ranges differ by at most 0.1 %. Float32 arithmetic differs between x86 and ARM builds of ONNX Runtime in the last bits, so a detection right at the score floor may appear on one side only; the tolerances allow that and nothing larger. The result is written to `results/edge/`, as `platform_compare_amd64_arm64.json`.

## Speed on a Pi 5

No Pi 5 has been measured yet. `results/edge/pi_estimate.json` is an estimate, labelled as such: laptop timings scaled by the ratio between a published Pi 5 timing and the laptop's timing of the same reference model (D-723).

When a Pi 5 is available, install the package on it (Raspberry Pi OS, 64-bit) and run:

```
make pi-benchmark FRAMES=<folder of JPEG frames> NOTE="anything else running"
```

`scripts/benchmark_v2.py` runs the same timing protocol as the laptop benchmark (warm-up, timed frames, repeats; per-stage medians and percentiles) on the configured pipeline and writes its result to `results/edge/`, as `benchmark_v2_pi5.json`.
