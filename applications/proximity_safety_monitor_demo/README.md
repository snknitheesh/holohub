# Proximity Safety Monitor Demo

A hardware-free demo of the [`proximity_safety_monitor`](../../operators/proximity_safety_monitor)
operator. It synthesizes a depth scene where an object approaches the camera over time, feeds it to
the monitor, and prints the safety status (CLEAR → WARN → STOP) and minimum distance each frame.

> ⚠️ The operator is a perception-based proximity assist, **not** a certified safety device. See the
> [operator README](../../operators/proximity_safety_monitor/README.md).

## Run

```bash
./holohub run proximity_safety_monitor_demo
# or directly:
python3 applications/proximity_safety_monitor_demo/proximity_safety_monitor_demo.py --frames 12
```

Expected output (object crosses the warn then stop thresholds as it approaches):

```
[proximity_safety_monitor] status=CLEAR min_distance=1.500 m stop_px=0 warn_px=0
...
[proximity_safety_monitor] status=WARN  min_distance=0.600 m stop_px=0 warn_px=6400
...
[proximity_safety_monitor] status=STOP  min_distance=0.150 m stop_px=6400 warn_px=6400
```

## Pipeline

```
ApproachingObjectDepthOp --depth--> ProximitySafetyMonitorOp --status--> SafetyStatusSinkOp
```

To use a real depth camera, replace the generator with `realsense_camera` (`depth_buffer`,
`depth_scale=0.001`), and optionally connect a human-segmentation mask to the monitor's `mask` input.

## Requirements

Holoscan SDK ≥ 3.5.0, CUDA, CuPy. Builds the `proximity_safety_monitor` operator (declared
dependency). Platforms: `x86_64`, `aarch64`.
