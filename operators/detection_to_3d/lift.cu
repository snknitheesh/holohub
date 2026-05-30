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

#include "lift.hpp"

namespace holoscan::ops {

// Median of vals[0..n) via insertion sort (n is small: <= kMaxPatchSize^2). Sorts in place.
__device__ inline float median_inplace(float* vals, int n) {
  for (int i = 1; i < n; ++i) {
    const float key = vals[i];
    int j = i - 1;
    while (j >= 0 && vals[j] > key) {
      vals[j + 1] = vals[j];
      --j;
    }
    vals[j + 1] = key;
  }
  return (n & 1) ? vals[n / 2] : 0.5f * (vals[n / 2 - 1] + vals[n / 2]);
}

// Pattern: one thread per detection. Each thread samples a small depth window and deprojects.
template <typename T>
__global__ void __launch_bounds__(128) lift_kernel(
    const float* __restrict__ boxes, int num_boxes, const T* __restrict__ depth, float depth_scale,
    bool normalized, CameraIntrinsics k, float depth_min, float depth_max, int patch_size,
    float invalid_value, float3* __restrict__ out_xyz, int width, int height) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_boxes) { return; }

  float x1 = boxes[i * 4 + 0];
  float y1 = boxes[i * 4 + 1];
  float x2 = boxes[i * 4 + 2];
  float y2 = boxes[i * 4 + 3];
  if (normalized) {
    x1 *= width;
    x2 *= width;
    y1 *= height;
    y2 *= height;
  }

  // Box center pixel, clamped to the image.
  int px = static_cast<int>(roundf(0.5f * (x1 + x2)));
  int py = static_cast<int>(roundf(0.5f * (y1 + y2)));
  px = min(max(px, 0), width - 1);
  py = min(max(py, 0), height - 1);

  // Gather valid metric depths in the patch_size x patch_size window.
  // Note: this per-thread buffer is non-constant-indexed, so it lives in (DRAM-backed)
  // local memory -- a fixed ~900 B/thread regardless of the runtime patch_size. That is
  // fine here because the workload is small (one thread per detection, N typically <= a
  // few hundred), so occupancy is irrelevant; do not reuse this kernel for very large N
  // without revisiting the sampling strategy.
  const int half = patch_size / 2;
  float samples[kMaxPatchSize * kMaxPatchSize];
  int n = 0;
  for (int dy = -half; dy <= half; ++dy) {
    const int yy = py + dy;
    if (yy < 0 || yy >= height) { continue; }
    for (int dx = -half; dx <= half; ++dx) {
      const int xx = px + dx;
      if (xx < 0 || xx >= width) { continue; }
      const T raw = depth[yy * width + xx];
      const float z = static_cast<float>(raw) * depth_scale;
      if (raw != T(0) && isfinite(z) && z >= depth_min && z <= depth_max) { samples[n++] = z; }
    }
  }

  if (n == 0) {
    out_xyz[i] = make_float3(invalid_value, invalid_value, invalid_value);
    return;
  }
  const float z = median_inplace(samples, n);
  out_xyz[i] = make_float3((px - k.cx) * z / k.fx, (py - k.cy) * z / k.fy, z);
}

cudaError_t launch_lift(const float* boxes, int num_boxes, const void* depth, DepthDType dtype,
                        float depth_scale, bool normalized, CameraIntrinsics intr, float depth_min,
                        float depth_max, int patch_size, float invalid_value, float3* out_xyz,
                        int width, int height, cudaStream_t stream) {
  if (num_boxes <= 0) { return cudaSuccess; }

  // Defensive clamp (the operator also clamps): bounds the per-thread sample buffer.
  if (patch_size < 1) { patch_size = 1; }
  if (patch_size > kMaxPatchSize) { patch_size = kMaxPatchSize; }

  constexpr int kBlock = 128;
  const int grid = (num_boxes + kBlock - 1) / kBlock;

  switch (dtype) {
    case DepthDType::kUint16:
      lift_kernel<uint16_t><<<grid, kBlock, 0, stream>>>(
          boxes, num_boxes, static_cast<const uint16_t*>(depth), depth_scale, normalized, intr,
          depth_min, depth_max, patch_size, invalid_value, out_xyz, width, height);
      break;
    case DepthDType::kFloat32:
      lift_kernel<float><<<grid, kBlock, 0, stream>>>(
          boxes, num_boxes, static_cast<const float*>(depth), depth_scale, normalized, intr,
          depth_min, depth_max, patch_size, invalid_value, out_xyz, width, height);
      break;
  }

  return cudaPeekAtLastError();
}

}  // namespace holoscan::ops
