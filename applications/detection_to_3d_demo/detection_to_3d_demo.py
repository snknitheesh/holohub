# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo for DetectionTo3DOp.

Generates a synthetic organized depth image (a gently tilting plane) and a few fixed
detection boxes entirely on the GPU, lifts each box to a 3D centroid with DetectionTo3DOp,
and prints the results. No camera, model, or dataset is required, so the app runs in CI.
See README.md for wiring a real 2D detector and depth source.
"""

import argparse

import cupy as cp
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.resources import UnboundedAllocator

from holohub.detection_to_3d import DetectionTo3DOp


class SyntheticSceneOp(Operator):
    """Emit a synthetic float32 depth image (meters) and a few detection boxes (xyxy pixels)."""

    def __init__(self, fragment, *args, width=640, height=480, **kwargs):
        self.width = width
        self.height = height
        self.frame = 0
        ys, xs = cp.meshgrid(
            cp.arange(height, dtype=cp.float32),
            cp.arange(width, dtype=cp.float32),
            indexing="ij",
        )
        self._xs = xs
        self._ys = ys
        # Three fixed boxes (xyxy, absolute pixels) within a 640x480 frame.
        self._boxes = cp.asarray(
            [
                [80.0, 60.0, 160.0, 140.0],
                [300.0, 200.0, 380.0, 280.0],
                [480.0, 320.0, 560.0, 400.0],
            ],
            dtype=cp.float32,
        )
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("detections")
        spec.output("depth")

    def compute(self, op_input, op_output, context):
        t = self.frame * 0.05
        # A tilted plane in meters with a slow oscillation so successive frames differ.
        depth = (
            1.0 + 0.5 * (self._xs / self.width) + 0.4 * (self._ys / self.height) + 0.3 * cp.sin(t)
        ).astype(cp.float32)

        n = int(self._boxes.shape[0])
        scores = cp.full((n,), 0.9, dtype=cp.float32)
        labels = cp.arange(n, dtype=cp.int32)

        op_output.emit({"boxes": self._boxes, "scores": scores, "labels": labels}, "detections")
        op_output.emit({"depth": depth}, "depth")
        self.frame += 1


class Print3DOp(Operator):
    """Pull the 3D detections and print each box's 3D centroid (CI-friendly sink)."""

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("in")
        pos = cp.asarray(msg["positions"])  # [N, 3] float32
        n = int(pos.shape[0])
        valid = int((~cp.isnan(pos[:, 2])).sum().get())
        zs = ", ".join(f"{z:.3f}" for z in cp.asnumpy(pos[:, 2]))
        print(f"[detection_to_3d_demo] detections={n} valid={valid} z=[{zs}] m")


class DetectionTo3DDemoApp(Application):
    def __init__(self, frames=100, width=640, height=480):
        super().__init__()
        self._frames = frames
        self._width = width
        self._height = height

    def compose(self):
        scene = SyntheticSceneOp(
            self,
            CountCondition(self, count=self._frames),
            name="scene",
            width=self._width,
            height=self._height,
        )

        lift = DetectionTo3DOp(
            self,
            name="detection_to_3d",
            allocator=UnboundedAllocator(self, name="pool"),
            # Pinhole intrinsics for the synthetic camera (principal point at image center).
            fx=float(self._width) * 0.8,
            fy=float(self._width) * 0.8,
            cx=(self._width - 1) / 2.0,
            cy=(self._height - 1) / 2.0,
            depth_scale=1.0,  # synthetic depth is already in meters
            depth_min=0.1,
            depth_max=10.0,
            normalized=False,  # boxes are absolute pixels
            patch_size=5,
        )

        sink = Print3DOp(self, name="print")

        self.add_flow(scene, lift, {("detections", "detections")})
        self.add_flow(scene, lift, {("depth", "depth")})
        self.add_flow(lift, sink, {("detections_3d", "in")})


def main():
    parser = argparse.ArgumentParser(description="DetectionTo3DOp synthetic demo")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    app = DetectionTo3DDemoApp(frames=args.frames, width=args.width, height=args.height)
    app.run()


if __name__ == "__main__":
    main()
