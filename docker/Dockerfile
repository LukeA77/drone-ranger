# The package's runtime image, built the same way for x86-64 and ARM64 (Raspberry Pi 5).
# Only the package's own pinned environment (requirements-lock.txt): no PyTorch, no AGPL code.
# Model weights and frames are not in the image; mount them at run time (docs/edge.md).
#
#   docker buildx build --platform linux/arm64 -f docker/Dockerfile -t aerialrange:arm64 --load .
#   docker buildx build --platform linux/amd64 -f docker/Dockerfile -t aerialrange:amd64 --load .
#
# On an x86 laptop the ARM64 image runs under QEMU emulation: it proves the pipeline runs and
# gives the same outputs on ARM64; it says nothing about speed on a Pi.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-lock.txt .
RUN python -m pip install -r requirements-lock.txt

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src src
COPY configs configs
COPY scripts scripts
# Editable, so the package finds configs/ and writes results/ under /app, as in a checkout.
RUN python -m pip install --no-deps -e . \
    && useradd --create-home --uid 1000 app \
    && mkdir -p cache results weights \
    && chown -R app /app
USER app

CMD ["aerialrange", "--help"]
