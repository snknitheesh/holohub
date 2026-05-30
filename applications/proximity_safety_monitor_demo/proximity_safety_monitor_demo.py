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

"""Demo for ProximitySafetyMonitorOp.

Generates a synthetic depth scene where an object approaches the camera over time and feeds it to
ProximitySafetyMonitorOp, printing the safety status (CLEAR/WARN/STOP) and minimum distance each
frame. No camera required, so it runs in CI.

NOT a certified safety device — see the operator README.
"""

import argparse

import cupy as cp
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType

from holohub.proximity_safety_monitor import ProximitySafetyMonitorOp

_STATUS = {0: "CLEAR", 1: "WARN", 2: "STOP"}


class ApproachingObjectDepthOp(Operator):
    """Emit a float32 depth image (meters): far background with a central object approaching."""

    def __init__(self, fragment, *args, width=640, height=480, block=80, **kwargs):
        self.width = width
        self.height = height
        self.block = block
        self.frame = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("depth")

    def compute(self, op_input, op_output, context):
        depth = cp.full((self.height, self.width), 5.0, dtype=cp.float32)  # far background
        obj_distance = max(0.15, 1.5 - self.frame * 0.15)  # approaches from 1.5 m toward 0.15 m
        cy, cx = self.height // 2, self.width // 2
        h = self.block // 2
        depth[cy - h : cy + h, cx - h : cx + h] = obj_distance
        op_output.emit({"depth": depth}, "depth")
        self.frame += 1


class SafetyStatusSinkOp(Operator):
    """Print the safety status and minimum distance each frame."""

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("in")
        status = int(cp.asarray(msg["status"]).get()[0])
        min_d = float(cp.asarray(msg["min_distance"]).get()[0])
        stop_c = int(cp.asarray(msg["stop_count"]).get()[0])
        warn_c = int(cp.asarray(msg["warn_count"]).get()[0])
        print(
            f"[proximity_safety_monitor] status={_STATUS.get(status, status)} "
            f"min_distance={min_d:.3f} m stop_px={stop_c} warn_px={warn_c}"
        )


class ProximitySafetyMonitorDemoApp(Application):
    def __init__(self, frames=12, width=640, height=480):
        super().__init__()
        self._frames = frames
        self._width = width
        self._height = height

    def compose(self):
        generator = ApproachingObjectDepthOp(
            self,
            CountCondition(self, count=self._frames),
            name="generator",
            width=self._width,
            height=self._height,
        )

        monitor = ProximitySafetyMonitorOp(
            self,
            name="monitor",
            allocator=BlockMemoryPool(
                self,
                name="pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=256,
                num_blocks=8,
            ),
            stop_distance=0.3,
            warn_distance=0.6,
            depth_scale=1.0,  # synthetic depth already in meters
            depth_min=0.05,
            depth_max=10.0,
            min_trigger_pixels=50,
            distance_mode="z",
            cuda_stream_pool=CudaStreamPool(self, name="stream_pool", num_streams=4),
        )

        sink = SafetyStatusSinkOp(self, name="sink")

        self.add_flow(generator, monitor, {("depth", "depth")})
        self.add_flow(monitor, sink, {("status", "in")})


def main():
    parser = argparse.ArgumentParser(description="ProximitySafetyMonitorOp synthetic demo")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    ProximitySafetyMonitorDemoApp(
        frames=args.frames, width=args.width, height=args.height
    ).run()


if __name__ == "__main__":
    main()
