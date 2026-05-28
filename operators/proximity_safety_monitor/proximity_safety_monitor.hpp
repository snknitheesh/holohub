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

#ifndef HOLOSCAN_OPERATORS_PROXIMITY_SAFETY_MONITOR_PROXIMITY_SAFETY_MONITOR_HPP
#define HOLOSCAN_OPERATORS_PROXIMITY_SAFETY_MONITOR_PROXIMITY_SAFETY_MONITOR_HPP

#include <memory>
#include <string>

#include "holoscan/core/operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

namespace holoscan::ops {

/**
 * @brief Perception-based proximity safety monitor (a "virtual safety curtain") on the GPU.
 *
 * WARNING: This is NOT a certified functional-safety device (no SIL/PL rating, no redundancy,
 * non-deterministic latency). It must not replace hardware safety (safety-rated scanners, light
 * curtains, e-stop, safety PLC). Use it as a soft proximity warning/assist only.
 *
 * Reduces an incoming depth image to a safety decision: classifies each valid (optionally masked)
 * pixel's distance into STOP / WARN bands and emits the overall status, the minimum distance, and
 * per-zone violating-pixel counts. A zone only trips when at least `min_trigger_pixels` pixels
 * violate it (noise floor).
 *
 * ==Named Inputs==
 * - **depth** : `Entity` with a 2D depth `Tensor` (`uint16` scaled by `depth_scale`, or `float32`
 *   meters at `depth_scale == 1.0`), shape `[H, W]` or `[H, W, 1]`, device memory.
 * - **mask** *(optional)* : `Entity` with an `H x W` `uint8` mask; when connected, only mask != 0
 *   pixels are considered (e.g. a human-detection mask).
 *
 * ==Named Outputs==
 * - **status** : `Entity` with device tensors: `status` (`int32`, 0=clear/1=warn/2=stop),
 *   `min_distance` (`float32`, meters; `FLT_MAX` if no valid pixel), `stop_count` (`int32`),
 *   `warn_count` (`int32`, includes the stop band).
 *
 * ==Parameters==
 * - **stop_distance**, **warn_distance**: zone thresholds in meters.
 * - **depth_scale**: raw depth -> meters (default `0.001`, i.e. uint16 mm).
 * - **depth_min**, **depth_max**: valid metric depth range (defaults `0.0` / `100.0`).
 * - **min_trigger_pixels**: pixels needed to trip a zone (noise floor, default `50`).
 * - **distance_mode**: `"z"` (axis depth, default) or `"radial"` (Euclidean, uses fx/fy/cx/cy).
 * - **fx**, **fy**, **cx**, **cy**: pinhole intrinsics (only used in `radial` mode).
 * - **depth_tensor_name**, **mask_tensor_name**: input tensor names (empty = first tensor).
 * - **allocator**: device allocator for the output tensors.
 * - **cuda_stream_pool**: optional `CudaStreamPool`.
 */
class ProximitySafetyMonitorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ProximitySafetyMonitorOp)

  ProximitySafetyMonitorOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<float> stop_distance_;
  Parameter<float> warn_distance_;
  Parameter<float> depth_scale_;
  Parameter<float> depth_min_;
  Parameter<float> depth_max_;
  Parameter<int> min_trigger_pixels_;
  Parameter<std::string> distance_mode_;
  Parameter<float> fx_;
  Parameter<float> fy_;
  Parameter<float> cx_;
  Parameter<float> cy_;
  Parameter<std::string> depth_tensor_name_;
  Parameter<std::string> mask_tensor_name_;
  Parameter<std::shared_ptr<Allocator>> allocator_;

  CudaStreamHandler cuda_stream_handler_;

  int* min_bits_scratch_ = nullptr;  // device scratch for the reduced min-distance bit pattern
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_PROXIMITY_SAFETY_MONITOR_PROXIMITY_SAFETY_MONITOR_HPP */
