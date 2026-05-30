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

// Standalone golden-reference unit test for the safety-monitor kernel + decision rule.
// Build: nvcc -O2 -arch=sm_<cc> -o test_safety_monitor test/test_safety_monitor.cu safety_monitor.cu

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../safety_monitor.hpp"

using namespace holoscan::ops;

static int g_failures = 0;
#define CHECK(cond, msg)            \
  do {                              \
    if (!(cond)) {                  \
      printf("FAIL: %s\n", (msg));  \
      ++g_failures;                 \
    }                               \
  } while (0)

struct Reduced {
  int stop_count;
  int warn_count;
  float min_distance;
};

static Reduced run(const std::vector<float>* depth_f, const std::vector<uint16_t>* depth_u,
                   float depth_scale, const std::vector<uint8_t>* mask, DistanceMode mode,
                   CameraIntrinsics k, SafetyZones z, int W, int H) {
  const int N = W * H;
  void* d_depth = nullptr;
  uint8_t* d_mask = nullptr;
  int *d_stop, *d_warn, *d_min;
  cudaMalloc(&d_stop, sizeof(int));
  cudaMalloc(&d_warn, sizeof(int));
  cudaMalloc(&d_min, sizeof(int));
  DepthDType dtype;
  if (depth_f) {
    cudaMalloc(&d_depth, N * sizeof(float));
    cudaMemcpy(d_depth, depth_f->data(), N * sizeof(float), cudaMemcpyHostToDevice);
    dtype = DepthDType::kFloat32;
  } else {
    cudaMalloc(&d_depth, N * sizeof(uint16_t));
    cudaMemcpy(d_depth, depth_u->data(), N * sizeof(uint16_t), cudaMemcpyHostToDevice);
    dtype = DepthDType::kUint16;
  }
  if (mask) {
    cudaMalloc(&d_mask, N * sizeof(uint8_t));
    cudaMemcpy(d_mask, mask->data(), N * sizeof(uint8_t), cudaMemcpyHostToDevice);
  }

  launch_safety_monitor(d_depth, dtype, depth_scale, d_mask, mode, k, z, d_stop, d_warn, d_min, W, H,
                        0);
  cudaDeviceSynchronize();

  Reduced r{};
  int min_bits = 0;
  cudaMemcpy(&r.stop_count, d_stop, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&r.warn_count, d_warn, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&min_bits, d_min, sizeof(int), cudaMemcpyDeviceToHost);
  r.min_distance = min_distance_from_bits(min_bits);

  cudaFree(d_depth);
  cudaFree(d_stop);
  cudaFree(d_warn);
  cudaFree(d_min);
  if (d_mask) cudaFree(d_mask);
  return r;
}

int main() {
  const int W = 64, H = 48, N = W * H;
  const CameraIntrinsics k{50.f, 50.f, 31.5f, 23.5f};
  const SafetyZones z{0.5f, 1.0f, 0.1f, 10.0f};

  // ---- Case 1: Z mode, float32. 16 px @0.3 (stop), 36 px @0.8 (warn), hole + far ignored ----
  {
    std::vector<float> depth(N, 5.0f);
    int stop_px = 0, warn_only_px = 0;
    for (int v = 0; v < 4; ++v)
      for (int u = 0; u < 4; ++u) { depth[v * W + u] = 0.3f; ++stop_px; }            // 16
    for (int v = 10; v < 16; ++v)
      for (int u = 10; u < 16; ++u) { depth[v * W + u] = 0.8f; ++warn_only_px; }     // 36
    depth[20 * W + 20] = 0.0f;    // hole -> ignored
    depth[21 * W + 21] = 50.0f;   // out of range -> ignored
    Reduced r = run(&depth, nullptr, 1.0f, nullptr, DistanceMode::kZ, k, z, W, H);
    CHECK(r.stop_count == 16, "Case1 stop_count");
    CHECK(r.warn_count == 16 + 36, "Case1 warn_count (inclusive)");
    CHECK(fabsf(r.min_distance - 0.3f) < 1e-4f, "Case1 min_distance");
    CHECK(decide_status(r.stop_count, r.warn_count, 10) == SafetyStatus::kStop, "Case1 status STOP");
  }

  // ---- Case 2: mask gating. Only 5 of the close pixels are in-mask ----
  {
    std::vector<float> depth(N, 5.0f);
    std::vector<uint8_t> mask(N, 0);
    for (int u = 0; u < 16; ++u) depth[u] = 0.3f;  // 16 close pixels in row 0
    for (int u = 0; u < 5; ++u) mask[u] = 1;       // only first 5 in mask
    Reduced r = run(&depth, nullptr, 1.0f, &mask, DistanceMode::kZ, k, z, W, H);
    CHECK(r.stop_count == 5, "Case2 masked stop_count");
    CHECK(r.warn_count == 5, "Case2 masked warn_count");
    CHECK(decide_status(r.stop_count, r.warn_count, 10) == SafetyStatus::kClear,
          "Case2 status CLEAR (below trigger)");
  }

  // ---- Case 3: uint16 millimeters + depth_scale ----
  {
    std::vector<uint16_t> depth(N, 5000);
    for (int u = 0; u < 20; ++u) depth[u] = 300;  // 0.3 m
    Reduced r = run(nullptr, &depth, 0.001f, nullptr, DistanceMode::kZ, k, z, W, H);
    CHECK(r.stop_count == 20, "Case3 uint16 stop_count");
    CHECK(fabsf(r.min_distance - 0.3f) < 1e-4f, "Case3 uint16 min_distance");
  }

  // ---- Case 4: radial mode differs from Z. Single corner pixel Z=0.45 -> radial ~0.572 ----
  {
    std::vector<float> depth(N, 5.0f);
    depth[0] = 0.45f;  // (u=0, v=0)
    Reduced r = run(&depth, nullptr, 1.0f, nullptr, DistanceMode::kRadial, k, z, W, H);
    float X = (0 - k.cx) * 0.45f / k.fx, Y = (0 - k.cy) * 0.45f / k.fy, Z = 0.45f;
    float radial = sqrtf(X * X + Y * Y + Z * Z);  // ~0.572 m
    CHECK(r.stop_count == 0, "Case4 radial stop_count (0.45 Z but radial > stop)");
    CHECK(r.warn_count == 1, "Case4 radial warn_count");
    CHECK(fabsf(r.min_distance - radial) < 1e-3f, "Case4 radial min_distance");
  }

  // ---- Case 5: decision rule (pure) ----
  {
    CHECK(decide_status(12, 20, 10) == SafetyStatus::kStop, "Case5 STOP");
    CHECK(decide_status(0, 15, 10) == SafetyStatus::kWarn, "Case5 WARN");
    CHECK(decide_status(0, 3, 10) == SafetyStatus::kClear, "Case5 CLEAR");
    CHECK(decide_status(0, 0, 1) == SafetyStatus::kClear, "Case5 CLEAR (none)");
  }

  if (g_failures == 0) {
    printf("ALL TESTS PASSED\n");
    return 0;
  }
  printf("%d FAILURE(S)\n", g_failures);
  return 1;
}
