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

#ifndef HOLOSCAN_OPERATORS_DETECTION_TO_3D_DETECTION_TO_3D_HPP
#define HOLOSCAN_OPERATORS_DETECTION_TO_3D_DETECTION_TO_3D_HPP

#include <memory>
#include <string>

#include "holoscan/core/operator.hpp"

namespace holoscan::ops {

/**
 * @brief Lift 2D detection boxes into 3D points using a depth image (GPU).
 *
 * For each detection box, samples a depth at the box center and deprojects it into a 3D point in
 * the camera optical frame. Detector-agnostic: consumes a generic `boxes` tensor.
 *
 * ==Named Inputs==
 *
 * - **detections** : `nvidia::gxf::Entity` containing:
 *   - `boxes` (`boxes_tensor_name`, default `"boxes"`): `float32` tensor `[N, 4]` in `xyxy`
 *     layout (absolute pixels, or `[0, 1]` when `normalized` is true);
 *   - `scores` (`scores_tensor_name`, default `"scores"`) *(optional)*: `[N]` tensor, passed
 *     through unchanged;
 *   - `labels` (`labels_tensor_name`, default `"labels"`) *(optional)*: `[N]` tensor, passed
 *     through unchanged.
 * - **depth** : `nvidia::gxf::Entity` containing a 2D depth `nvidia::gxf::Tensor`
 *   (`uint16` scaled by `depth_scale`, or `float32` meters when `depth_scale == 1.0`),
 *   shape `[H, W]` or `[H, W, 1]`, device memory.
 * - **intrinsics** *(optional)* : `nvidia::gxf::Entity` containing a `float32` tensor of 4 values
 *   `[fx, fy, cx, cy]`. Overrides the `fx/fy/cx/cy` parameters for the current frame.
 *
 * ==Named Outputs==
 *
 * - **detections_3d** : `nvidia::gxf::Entity` containing:
 *   - a `float32` tensor (`output_positions_name`, default `"positions"`) of shape `[N, 3]`
 *     (x, y, z) in the camera optical frame; boxes with no valid depth are set to `invalid_value`;
 *   - the passed-through `scores`/`labels` tensors (when present on the input).
 *
 * ==Parameters==
 *
 * - **fx**, **fy**, **cx**, **cy**: pinhole intrinsics in pixels (used when `intrinsics`
 *   port is not connected).
 * - **depth_scale**: multiply raw depth by this to get meters (default `0.001`, i.e. uint16 mm).
 * - **depth_min**, **depth_max**: valid metric depth range in meters (defaults `0.0` / `100.0`).
 * - **normalized**: if true, boxes are in `[0, 1]` and scaled by `(W, H)` (default `false`).
 * - **patch_size**: odd median-window size, clamped to `[1, 15]` (default `5`; `1` = center pixel).
 * - **invalid_value**: value written to X/Y/Z for boxes with no valid depth (default `NaN`).
 * - **boxes_tensor_name**, **scores_tensor_name**, **labels_tensor_name**: input tensor names.
 * - **output_positions_name**: name of the emitted positions tensor (default `"positions"`).
 * - **allocator**: device `holoscan::Allocator` for the output tensors (e.g. BlockMemoryPool).
 */
class DetectionTo3DOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DetectionTo3DOp)

  DetectionTo3DOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<float> fx_;
  Parameter<float> fy_;
  Parameter<float> cx_;
  Parameter<float> cy_;
  Parameter<float> depth_scale_;
  Parameter<float> depth_min_;
  Parameter<float> depth_max_;
  Parameter<bool> normalized_;
  Parameter<int> patch_size_;
  Parameter<float> invalid_value_;
  Parameter<std::string> boxes_tensor_name_;
  Parameter<std::string> scores_tensor_name_;
  Parameter<std::string> labels_tensor_name_;
  Parameter<std::string> depth_tensor_name_;
  Parameter<std::string> output_positions_name_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_DETECTION_TO_3D_DETECTION_TO_3D_HPP */
