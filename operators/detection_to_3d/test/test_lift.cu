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

// Standalone golden-reference unit test for the lift kernel. Compiles with nvcc
// alone (no Holoscan SDK):
//   nvcc -O2 -o test_lift test/test_lift.cu lift.cu && ./test_lift

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../lift.hpp"

using namespace holoscan::ops;

static int g_failures = 0;
#define CHECK(cond, msg)           \
  do {                             \
    if (!(cond)) {                 \
      printf("FAIL: %s\n", (msg)); \
      ++g_failures;                \
    }                              \
  } while (0)

// CPU golden: deproject the box center pixel (px, py) at metric depth z.
static float3 golden(float z, int px, int py, CameraIntrinsics k) {
  return make_float3((px - k.cx) * z / k.fx, (py - k.cy) * z / k.fy, z);
}

static bool approx(float3 a, float3 b, float eps = 1e-4f) {
  return fabsf(a.x - b.x) < eps && fabsf(a.y - b.y) < eps && fabsf(a.z - b.z) < eps;
}

// Run the kernel over a depth image + boxes, return host outputs.
template <typename T>
static std::vector<float3> run(const std::vector<T>& depth, int W, int H,
                               const std::vector<float>& boxes, DepthDType dtype, float scale,
                               bool normalized, CameraIntrinsics k, float dmin, float dmax,
                               int patch, float invalid) {
  const int N = static_cast<int>(boxes.size()) / 4;
  T* d_depth = nullptr;
  float* d_boxes = nullptr;
  float3* d_out = nullptr;
  cudaMalloc(&d_depth, depth.size() * sizeof(T));
  cudaMalloc(&d_boxes, boxes.size() * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float3));
  cudaMemcpy(d_depth, depth.data(), depth.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_boxes, boxes.data(), boxes.size() * sizeof(float), cudaMemcpyHostToDevice);
  launch_lift(d_boxes, N, d_depth, dtype, scale, normalized, k, dmin, dmax, patch, invalid, d_out,
              W, H, 0);
  cudaDeviceSynchronize();
  std::vector<float3> out(N);
  cudaMemcpy(out.data(), d_out, N * sizeof(float3), cudaMemcpyDeviceToHost);
  cudaFree(d_depth);
  cudaFree(d_boxes);
  cudaFree(d_out);
  return out;
}

int main() {
  const int W = 64, H = 48;
  const CameraIntrinsics k{50.f, 50.f, 31.5f, 23.5f};
  const float dmin = 0.1f, dmax = 10.f, invalid = NAN;

  // ---- Case 1: float32 constant plane Z=2.0, pixel boxes, patch=1 (center pixel) ----
  {
    std::vector<float> depth(W * H, 2.0f);
    // two boxes with centers (40,30) and (10,10)
    std::vector<float> boxes{38, 28, 42, 32, 8, 8, 12, 12};
    auto out =
        run(depth, W, H, boxes, DepthDType::kFloat32, 1.0f, false, k, dmin, dmax, 1, invalid);
    CHECK(approx(out[0], golden(2.0f, 40, 30, k)), "Case1 box0 position wrong");
    CHECK(approx(out[1], golden(2.0f, 10, 10, k)), "Case1 box1 position wrong");
  }

  // ---- Case 2: normalized boxes describe the same centers -> same positions ----
  {
    std::vector<float> depth(W * H, 2.0f);
    std::vector<float> boxes{38.f / W, 28.f / H, 42.f / W, 32.f / H,
                             8.f / W,  8.f / H,  12.f / W, 12.f / H};
    auto out = run(depth, W, H, boxes, DepthDType::kFloat32, 1.0f, true, k, dmin, dmax, 1, invalid);
    CHECK(approx(out[0], golden(2.0f, 40, 30, k)), "Case2 normalized box0 wrong");
    CHECK(approx(out[1], golden(2.0f, 10, 10, k)), "Case2 normalized box1 wrong");
  }

  // ---- Case 3: depth beyond depth_max -> invalid (NaN) ----
  {
    std::vector<float> depth(W * H, 50.0f);  // > dmax
    std::vector<float> boxes{38, 28, 42, 32};
    auto out =
        run(depth, W, H, boxes, DepthDType::kFloat32, 1.0f, false, k, dmin, dmax, 1, invalid);
    CHECK(std::isnan(out[0].x) && std::isnan(out[0].z), "Case3 out-of-range not NaN");
  }

  // ---- Case 4: uint16 millimeters with depth_scale 0.001 ----
  {
    std::vector<uint16_t> depth(W * H, 2000);  // 2.0 m
    std::vector<float> boxes{38, 28, 42, 32};
    auto out =
        run(depth, W, H, boxes, DepthDType::kUint16, 0.001f, false, k, dmin, dmax, 1, invalid);
    CHECK(approx(out[0], golden(2.0f, 40, 30, k)), "Case4 uint16 depth_scale wrong");
  }

  // ---- Case 5: patch median robust to invalid + outlier pixels in the window ----
  {
    std::vector<float> depth(W * H, 2.0f);
    const int cx = 40, cy = 30;
    depth[cy * W + cx] = 0.0f;            // center is a hole (invalid)
    depth[cy * W + (cx + 1)] = 9.0f;      // one in-range outlier
    // 3x3 window: 9 pixels, 1 invalid (skipped), 8 valid {2,2,2,2,2,2,2,9} -> median 2.0
    std::vector<float> boxes{38, 28, 42, 32};
    auto out =
        run(depth, W, H, boxes, DepthDType::kFloat32, 1.0f, false, k, dmin, dmax, 3, invalid);
    CHECK(!std::isnan(out[0].z), "Case5 patch median should recover from a hole center");
    CHECK(approx(out[0], golden(2.0f, cx, cy, k)), "Case5 patch median should be 2.0");
  }

  if (g_failures == 0) {
    printf("ALL TESTS PASSED\n");
    return 0;
  }
  printf("%d FAILURE(S)\n", g_failures);
  return 1;
}
