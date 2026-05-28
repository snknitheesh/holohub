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

#ifndef HOLOSCAN_OPERATORS_PROXIMITY_SAFETY_MONITOR_SAFETY_MONITOR_HPP
#define HOLOSCAN_OPERATORS_PROXIMITY_SAFETY_MONITOR_SAFETY_MONITOR_HPP

#include <cuda_runtime.h>

#include <cstdint>

// Holoscan-free so the reduction kernel can be unit-tested with nvcc alone
// (see test/test_safety_monitor.cu).

namespace holoscan::ops {

enum class DepthDType : int { kUint16 = 0, kFloat32 = 1 };
enum class DistanceMode : int { kZ = 0, kRadial = 1 };
enum class SafetyStatus : int { kClear = 0, kWarn = 1, kStop = 2 };

struct CameraIntrinsics {
  float fx;
  float fy;
  float cx;
  float cy;
};

struct SafetyZones {
  float stop_distance;  // meters; pixels closer than this count toward STOP
  float warn_distance;  // meters; pixels closer than this count toward WARN (inclusive of STOP band)
  float depth_min;      // meters; metric depths below this are treated as invalid
  float depth_max;      // meters; metric depths above this are treated as invalid
};

// Usable from both host (operator/tests) and device (finalize kernel) so the decision rule has a
// single source of truth. __host__/__device__ only exist under nvcc.
#ifdef __CUDACC__
#define SAFETY_MONITOR_HD __host__ __device__
#else
#define SAFETY_MONITOR_HD
#endif

/// Pure decision rule from the reduced counts. A zone trips only when at least
/// min_trigger_pixels violate it (noise floor). STOP takes precedence over WARN.
SAFETY_MONITOR_HD inline SafetyStatus decide_status(int stop_count, int warn_count,
                                                    int min_trigger_pixels) {
  if (stop_count >= min_trigger_pixels) { return SafetyStatus::kStop; }
  if (warn_count >= min_trigger_pixels) { return SafetyStatus::kWarn; }
  return SafetyStatus::kClear;
}

/**
 * @brief Reduce a depth image to proximity-safety statistics.
 *
 * For each valid pixel (raw != 0, finite, metric depth in [depth_min, depth_max], and, if a mask
 * is supplied, mask != 0) computes a distance — either the metric depth Z (kZ) or the Euclidean
 * distance of the back-projected point (kRadial, needs intrinsics) — and accumulates:
 *   - stop_count: number of pixels with distance < stop_distance
 *   - warn_count: number of pixels with distance < warn_distance (includes the stop band)
 *   - min_bits:   bit pattern of the minimum distance (use the host helper to read it back)
 *
 * Outputs are device int pointers; the wrapper initializes them (counts to 0, min_bits to a large
 * sentinel). Reduction is block-local then one atomic per block per accumulator.
 *
 * @return cudaPeekAtLastError() after the launch.
 */
cudaError_t launch_safety_monitor(const void* depth, DepthDType dtype, float depth_scale,
                                  const uint8_t* mask, DistanceMode mode, CameraIntrinsics intr,
                                  SafetyZones zones, int* d_stop_count, int* d_warn_count,
                                  int* d_min_bits, int width, int height, cudaStream_t stream);

/// Device finalize: turn the reduced counts + min-distance bits into a SafetyStatus int and a
/// metric min distance, all on the GPU (no host round-trip). Reuses decide_status().
cudaError_t launch_finalize(const int* d_stop_count, const int* d_warn_count,
                            const int* d_min_bits, int min_trigger_pixels, int* d_status,
                            float* d_min_distance, cudaStream_t stream);

/// Convert the reduced min-distance bit pattern back to meters; returns FLT_MAX when no valid pixel.
inline float min_distance_from_bits(int bits) {
  float v;
  static_assert(sizeof(float) == sizeof(int), "float/int size mismatch");
  __builtin_memcpy(&v, &bits, sizeof(v));
  return v;
}

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_PROXIMITY_SAFETY_MONITOR_SAFETY_MONITOR_HPP */
