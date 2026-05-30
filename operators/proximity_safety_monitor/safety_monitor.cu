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

#include "safety_monitor.hpp"

namespace holoscan::ops {

// 0x7f7f7f7f == FLT_MAX bit pattern; used as the "no valid pixel" min sentinel. Distances are
// positive, so the int bit pattern of a float is monotonic and atomicMin on bits yields the min.
static constexpr int kMinSentinel = 0x7f7f7f7f;

template <typename T>
__global__ void __launch_bounds__(256) safety_monitor_kernel(
    const T* __restrict__ depth, float depth_scale, const uint8_t* __restrict__ mask,
    DistanceMode mode, CameraIntrinsics k, SafetyZones z, int* __restrict__ g_stop,
    int* __restrict__ g_warn, int* __restrict__ g_min, int width, int height) {
  __shared__ int s_stop;
  __shared__ int s_warn;
  __shared__ int s_min;
  if (threadIdx.x == 0) {
    s_stop = 0;
    s_warn = 0;
    s_min = kMinSentinel;
  }
  __syncthreads();

  const int n = width * height;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    if (mask != nullptr && mask[i] == 0) { continue; }

    const T raw = depth[i];
    if (raw == T(0)) { continue; }
    const float Z = static_cast<float>(raw) * depth_scale;
    if (!isfinite(Z) || Z < z.depth_min || Z > z.depth_max) { continue; }

    float d = Z;
    if (mode == DistanceMode::kRadial) {
      const int u = i % width;
      const int v = i / width;
      const float X = (u - k.cx) * Z / k.fx;
      const float Y = (v - k.cy) * Z / k.fy;
      d = sqrtf(X * X + Y * Y + Z * Z);
    }

    if (d < z.warn_distance) {
      atomicAdd(&s_warn, 1);
      if (d < z.stop_distance) { atomicAdd(&s_stop, 1); }
    }
    atomicMin(&s_min, __float_as_int(d));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (s_stop != 0) { atomicAdd(g_stop, s_stop); }
    if (s_warn != 0) { atomicAdd(g_warn, s_warn); }
    atomicMin(g_min, s_min);
  }
}

cudaError_t launch_safety_monitor(const void* depth, DepthDType dtype, float depth_scale,
                                  const uint8_t* mask, DistanceMode mode, CameraIntrinsics intr,
                                  SafetyZones zones, int* d_stop_count, int* d_warn_count,
                                  int* d_min_bits, int width, int height, cudaStream_t stream) {
  cudaMemsetAsync(d_stop_count, 0, sizeof(int), stream);
  cudaMemsetAsync(d_warn_count, 0, sizeof(int), stream);
  cudaMemsetAsync(d_min_bits, 0x7f, sizeof(int), stream);  // -> 0x7f7f7f7f

  const int n = width * height;
  if (n <= 0) { return cudaPeekAtLastError(); }

  constexpr int kBlock = 256;
  int grid = (n + kBlock - 1) / kBlock;
  grid = grid < 1024 ? grid : 1024;  // grid-stride loop absorbs the remainder

  switch (dtype) {
    case DepthDType::kUint16:
      safety_monitor_kernel<uint16_t><<<grid, kBlock, 0, stream>>>(
          static_cast<const uint16_t*>(depth), depth_scale, mask, mode, intr, zones, d_stop_count,
          d_warn_count, d_min_bits, width, height);
      break;
    case DepthDType::kFloat32:
      safety_monitor_kernel<float><<<grid, kBlock, 0, stream>>>(
          static_cast<const float*>(depth), depth_scale, mask, mode, intr, zones, d_stop_count,
          d_warn_count, d_min_bits, width, height);
      break;
  }

  return cudaPeekAtLastError();
}

__global__ void finalize_kernel(const int* stop, const int* warn, const int* min_bits,
                                int min_trigger, int* status, float* min_distance) {
  *status = static_cast<int>(decide_status(*stop, *warn, min_trigger));
  *min_distance = __int_as_float(*min_bits);
}

cudaError_t launch_finalize(const int* d_stop_count, const int* d_warn_count,
                            const int* d_min_bits, int min_trigger_pixels, int* d_status,
                            float* d_min_distance, cudaStream_t stream) {
  finalize_kernel<<<1, 1, 0, stream>>>(d_stop_count, d_warn_count, d_min_bits, min_trigger_pixels,
                                       d_status, d_min_distance);
  return cudaPeekAtLastError();
}

}  // namespace holoscan::ops
