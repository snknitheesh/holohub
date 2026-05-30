/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN_OPERATORS_DETECTION_TO_3D_LIFT_HPP
#define HOLOSCAN_OPERATORS_DETECTION_TO_3D_LIFT_HPP

#include <cuda_runtime.h>

#include <cstdint>

// This header is intentionally free of any Holoscan dependency so the lift kernel
// can be unit-tested in isolation with nvcc (see test/test_lift.cu).

namespace holoscan::ops {

/// Supported element types of the input depth image.
enum class DepthDType : int { kUint16 = 0, kFloat32 = 1 };

/// Pinhole camera intrinsics in pixels.
struct CameraIntrinsics {
  float fx;
  float fy;
  float cx;
  float cy;
};

/// Largest supported median window (patch_size is clamped to this).
constexpr int kMaxPatchSize = 15;

/**
 * @brief Lift N 2D detection boxes into 3D points by sampling depth at each box center.
 *
 * One CUDA thread per detection. For detection i with box [x1, y1, x2, y2] (xyxy):
 *   - if `normalized`, the box is scaled by (width, height);
 *   - the center pixel (px, py) is the rounded box center, clamped to the image;
 *   - the metric depth Z is the median of valid depths in a `patch_size x patch_size`
 *     window centered at (px, py), where a sample is valid iff
 *     `raw != 0 && isfinite(Z) && depth_min <= Z <= depth_max`, with `Z = raw * depth_scale`;
 *   - if no sample is valid, `out_xyz[i] = (invalid_value, invalid_value, invalid_value)`;
 *   - otherwise the optical-frame point (x-right, y-down, z-forward) is
 *     `X = (px - cx) * Z / fx`, `Y = (py - cy) * Z / fy`, `Z = Z`.
 *
 * @param boxes         device pointer to `num_boxes * 4` float32 box coords (xyxy)
 * @param num_boxes     number of detections N
 * @param depth         device pointer to the HxW depth image (element type = `dtype`)
 * @param dtype         depth element type (uint16 or float32)
 * @param depth_scale   multiply raw depth by this to get meters (e.g. 0.001 for uint16 mm)
 * @param normalized    if true, boxes are in [0, 1] and scaled by (width, height)
 * @param intr          pinhole intrinsics in pixels
 * @param depth_min     minimum valid metric depth (meters), inclusive
 * @param depth_max     maximum valid metric depth (meters), inclusive
 * @param patch_size    odd window size for the median (clamped to [1, kMaxPatchSize])
 * @param invalid_value value written to X/Y/Z for boxes with no valid depth (e.g. NaN)
 * @param out_xyz       device pointer to `num_boxes` float3 outputs
 * @param width         depth image width in pixels
 * @param height        depth image height in pixels
 * @param stream        CUDA stream to launch on
 * @return cudaPeekAtLastError() result after the launch
 */
cudaError_t launch_lift(const float* boxes, int num_boxes, const void* depth, DepthDType dtype,
                        float depth_scale, bool normalized, CameraIntrinsics intr, float depth_min,
                        float depth_max, int patch_size, float invalid_value, float3* out_xyz,
                        int width, int height, cudaStream_t stream);

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_DETECTION_TO_3D_LIFT_HPP */
