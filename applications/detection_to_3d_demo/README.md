# Detection to 3D Demo

A minimal, hardware-free demo of the [`detection_to_3d`](../../operators/detection_to_3d) operator.
It generates a synthetic organized depth image (a gently tilting plane) plus a few fixed detection
boxes entirely on the GPU, lifts each box to a 3D centroid, and prints the per-box Z each frame. No
camera, model, or dataset is required.

## Run

```bash
./holohub run detection_to_3d_demo
# or directly:
python3 applications/detection_to_3d_demo/detection_to_3d_demo.py --frames 100
```

Expected output (per frame):

```text
[detection_to_3d_demo] detections=3 valid=3 z=[1.xxx, 1.xxx, 1.xxx] m
```

## Pipeline

```text
SyntheticSceneOp  --detections (boxes/scores/labels)-->  DetectionTo3DOp  --detections_3d-->  Print3DOp
                  --depth--------------------------->
```

## Using a real detector + depth source

Replace the synthetic scene with a real 2D detector (emitting a `boxes` tensor `[N, 4]`, plus
optional `scores`/`labels`) and a depth source (e.g. the
[`realsense_camera`](../../operators/realsense_camera) operator's `depth_buffer`). Provide the depth
camera intrinsics as `fx/fy/cx/cy` (or wire the `intrinsics` port), and set `depth_scale=0.001` for
uint16-millimeter depth. Set `normalized=True` if your detector emits boxes in `[0, 1]`.

```python
lift = DetectionTo3DOp(
    self, name="detection_to_3d", allocator=...,
    fx=fx, fy=fy, cx=cx, cy=cy,
    depth_scale=0.001, depth_min=0.1, depth_max=10.0,
    normalized=False, patch_size=5,
)
self.add_flow(detector, lift, {("detections", "detections")})
self.add_flow(camera, lift, {("depth_buffer", "depth")})
```

## Requirements

- Holoscan SDK ≥ 4.0.0, CUDA, CuPy. Builds the `detection_to_3d` operator (declared as a
  dependency). Platforms: `x86_64`, `aarch64`.
