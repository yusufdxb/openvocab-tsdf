# Offline pipeline reproducibility image for openvocab-tsdf.
#
# Covers the offline pipeline (fuse, encode, eval, ground). ROS 2 Humble is
# excluded — it would triple the image size and is owned by a separate colcon
# workspace. See docs/decisions.md for the scope rationale.

FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir uv && \
    uv sync --extra dev --extra dense --extra scannet --extra lseg

ENTRYPOINT ["uv", "run", "openvocab-tsdf"]
