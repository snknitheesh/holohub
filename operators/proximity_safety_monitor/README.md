# Proximity Safety Monitor Operator

A GPU "virtual safety curtain": it reduces an incoming depth image to a proximity decision —
classifying each valid (optionally masked) pixel's distance into **STOP / WARN** zones and emitting
the overall **status**, the **minimum distance**, and per-zone **violating-pixel counts**. Useful as
a perception-based proximity warning for collaborative robots and humanoids, and it composes with
depth producers (RealSense, stereo, `point_cloud_from_depth` sources).

> ## ⚠️ NOT a certified safety device
> This operator is a **soft, perception-based proximity monitor/assist only**. It is **not** a
> certified functional-safety device: no SIL/PL rating, no redundancy, and GPU + camera latency is
> **not deterministic**. It must **never** replace hardware safety — safety-rated LiDAR scanners,
> light curtains, emergency stop, and a safety PLC. Do not use it as the sole protective measure for
> human safety.

## What it computes

Per valid pixel, distance `d` is either the metric depth `Z` (`distance_mode: z`, default) or the
Euclidean distance of the back-projected point `sqrt(X²+Y²+Z²)` (`distance_mode: radial`, uses
`fx/fy/cx/cy`). It reduces to:

- `stop_count` = pixels with `d < stop_distance`
- `warn_count` = pixels with `d < warn_distance` (includes the stop band)
- `min_distance` = minimum `d` over valid pixels

A zone trips only when at least `min_trigger_pixels` violate it (noise floor). Status:
`STOP` if `stop_count ≥ min_trigger_pixels`, else `WARN` if `warn_count ≥ min_trigger_pixels`, else
`CLEAR`. Reduction is block-local then one atomic per block; the decision is finalized on the GPU
(zero host round-trip).

## Ports

| Port | Direction | Type | Notes |
|---|---|---|---|
| `depth` | in | `Entity` w/ 2D tensor | `uint16` (×`depth_scale`) or `float32` meters, device |
| `mask` | in (optional) | `Entity` w/ `uint8` HxW | consider only `mask != 0` pixels (e.g. a human mask) |
| `status` | out | `Entity` | `status` int32 (0/1/2), `min_distance` f32 (m), `stop_count`/`warn_count` int32 |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `stop_distance` / `warn_distance` | `0.3` / `0.6` | Zone thresholds (m) |
| `depth_scale` | `0.001` | Raw depth → meters (`0.001` uint16 mm; `1.0` float32 m) |
| `depth_min` / `depth_max` | `0.0` / `100.0` | Valid depth range (m) |
| `min_trigger_pixels` | `50` | Pixels needed to trip a zone (noise floor) |
| `distance_mode` | `"z"` | `"z"` or `"radial"` |
| `fx,fy,cx,cy` | `0.0` | Intrinsics (radial mode only) |
| `depth_tensor_name` / `mask_tensor_name` | `""` | Input tensor names (empty = first) |
| `allocator` | — | Device allocator for outputs |
| `cuda_stream_pool` | `nullptr` | Optional `CudaStreamPool` |

## Testing

`test/test_safety_monitor.cu` is a standalone golden-reference test (CUDA runtime only, no Holoscan)
covering the Z and radial modes, `uint16`/`float32`, mask gating, invalid/out-of-range handling,
min-distance, and the decision rule. Registered with CTest as `proximity_safety_monitor_test`.

```bash
nvcc -O2 -arch=sm_<cc> -o t test/test_safety_monitor.cu safety_monitor.cu && ./t
```

## Requirements

Holoscan SDK ≥ 3.5.0, CUDA. Platforms: `x86_64`, `aarch64`. No third-party dependencies beyond the
CUDA runtime.
