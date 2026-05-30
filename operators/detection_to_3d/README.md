# Detection to 3D Operator

Lift 2D object detections into 3D on the GPU. For each detection bounding box, the operator
samples a depth at the box center and deprojects it into a 3D point in the camera optical frame —
turning any 2D detector (YOLO, SSD, …) into a 3D localizer for manipulation pick targets, obstacle
localization, and sensor fusion. It is detector-agnostic: it consumes a generic `boxes` tensor, not
a specific model head. Companion to the [`depth_to_point_cloud`](../depth_to_point_cloud) operator.

## What it computes

For each box `[x1, y1, x2, y2]` (xyxy; absolute pixels, or `[0, 1]` when `normalized` is set):

- the center pixel `(px, py)` is the rounded box center, clamped to the image;
- the metric depth `Z` is the **median** of valid depths in a `patch_size × patch_size` window at
  `(px, py)`, where a sample is valid iff `raw != 0`, `Z` is finite, and `depth_min ≤ Z ≤ depth_max`
  (with `Z = raw * depth_scale`);
- the 3D point in the **camera optical frame** (x-right, y-down, z-forward) is:

```text
X = (px - cx) * Z / fx
Y = (py - cy) * Z / fy
Z = Z
```

A box with no valid depth in its window is written as `(invalid_value, invalid_value, invalid_value)`
(default `NaN`). The kernel runs one CUDA thread per detection; depth pixels and boxes stay
GPU-resident.

> Frame note: the output is in the optical frame. For a ROS body frame (x-forward, y-left, z-up)
> apply the standard optical→body rotation downstream.

## Ports

| Port | Direction | Type | Notes |
| --- | --- | --- | --- |
| `detections` | in | `Entity` w/ tensors | `boxes` `[N, 4]` float32 xyxy (required, exactly 4 columns); `scores` `[N]`, `labels` `[N]` (optional, passed through) |
| `depth` | in | `Entity` w/ 2D tensor | `uint16` (scaled by `depth_scale`) or `float32` (meters at `depth_scale=1.0`), `[H, W]` or `[H, W, 1]`, device memory |
| `intrinsics` | in (optional) | `Entity` w/ `float32[4]` | `[fx, fy, cx, cy]`; overrides the params for that frame |
| `detections_3d` | out | `Entity` | `positions` `float32 [N, 3]` (optical frame) + passed-through `scores`/`labels` |

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `fx`, `fy`, `cx`, `cy` | `0.0` | Pinhole intrinsics in pixels (used when `intrinsics` port is unconnected) |
| `depth_scale` | `0.001` | Raw-depth → meters multiplier (`0.001` for uint16 mm; `1.0` for float32 m) |
| `depth_min`, `depth_max` | `0.0`, `100.0` | Valid metric depth range (meters) |
| `normalized` | `false` | If true, boxes are `[0, 1]` and scaled by `(W, H)`; else absolute pixels |
| `patch_size` | `5` | Median-window size at each box center (clamped to `[1, 15]`; even values are rounded up to the next odd; `1` = center pixel) |
| `invalid_value` | `NaN` | Value written to X/Y/Z for a box with no valid depth |
| `boxes_tensor_name` / `scores_tensor_name` / `labels_tensor_name` | `"boxes"` / `"scores"` / `"labels"` | Input tensor names in the `detections` message |
| `depth_tensor_name` | `""` | Depth tensor name (empty = first tensor) |
| `output_positions_name` | `"positions"` | Name of the emitted positions tensor |
| `allocator` | — | Device allocator for the output tensors (e.g. `BlockMemoryPool` sized to the pipeline depth for production; `UnboundedAllocator` for demos/tests) |

## Usage

### Python

```python
from holohub.detection_to_3d import DetectionTo3DOp

lift = DetectionTo3DOp(
    self,
    name="detection_to_3d",
    allocator=UnboundedAllocator(self, name="pool"),
    fx=fx, fy=fy, cx=cx, cy=cy,
    depth_scale=0.001,          # uint16 millimeters
    depth_min=0.1, depth_max=10.0,
    normalized=False,           # boxes are absolute pixels
    patch_size=5,
)
# detector -> lift -> consumer
self.add_flow(detector, lift, {("detections", "detections")})
self.add_flow(depth_source, lift, {("depth", "depth")})
```

### C++

```cpp
auto lift = make_operator<ops::DetectionTo3DOp>(
    "detection_to_3d",
    Arg("allocator", make_resource<UnboundedAllocator>("pool")),
    Arg("fx", fx), Arg("fy", fy), Arg("cx", cx), Arg("cy", cy),
    Arg("depth_scale", 0.001f));
```

## Testing

`test/test_lift.cu` is a standalone golden-reference unit test that depends only on the CUDA
runtime (no Holoscan SDK). It verifies the box-center deprojection (pixel and normalized layouts),
the uint16 `depth_scale` path, out-of-range → NaN handling, and patch-median robustness when the
center pixel is a hole:

```bash
nvcc -O2 -arch=sm_<cc> -o test_lift test/test_lift.cu lift.cu && ./test_lift
```

It is also registered with CTest (`detection_to_3d_test`) when the project is built with
`BUILD_TESTING` enabled. `test_detection_to_3d.py` adds pytest coverage of the Python bindings
(construction, ports, error handling, parameters).

## Requirements

- Holoscan SDK ≥ 4.0.0, CUDA. Platforms: `x86_64`, `aarch64` (Jetson). No third-party dependencies
  beyond the CUDA runtime — the lift runs as a custom CUDA kernel.
