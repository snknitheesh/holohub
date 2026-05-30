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

#include "detection_to_3d.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>

#include <gxf/std/tensor.hpp>

#include "lift.hpp"

#define CUDA_TRY(stmt)                                                                        \
  {                                                                                           \
    cudaError_t cuda_status = stmt;                                                           \
    if (cudaSuccess != cuda_status) {                                                         \
      HOLOSCAN_LOG_ERROR("CUDA runtime call {} in line {} of file {} failed with '{}' ({}).", \
                         #stmt,                                                                \
                         __LINE__,                                                             \
                         __FILE__,                                                             \
                         cudaGetErrorString(cuda_status),                                      \
                         static_cast<int>(cuda_status));                                       \
      throw std::runtime_error("CUDA runtime call failed");                                    \
    }                                                                                         \
  }

namespace holoscan::ops {

namespace {

// Map a DLPack dtype to the kernel's depth element type.
DepthDType to_depth_dtype(const DLDataType& dtype) {
  if (dtype.code == kDLFloat && dtype.bits == 32) { return DepthDType::kFloat32; }
  if (dtype.code == kDLUInt && dtype.bits == 16) { return DepthDType::kUint16; }
  throw std::runtime_error(
      "DetectionTo3DOp: unsupported depth dtype (expected float32 or uint16)");
}

// Fetch a required tensor from an entity, by name if given, otherwise the first tensor.
std::shared_ptr<Tensor> get_tensor(const holoscan::gxf::Entity& message, const std::string& name) {
  auto maybe = name.empty() ? message.get<Tensor>() : message.get<Tensor>(name.c_str());
  if (!maybe) {
    throw std::runtime_error("DetectionTo3DOp: input tensor '" + name + "' not found");
  }
  return maybe;
}

// Forward an optional input tensor (e.g. scores/labels) into the output message unchanged,
// preserving dtype. No-op when the named tensor is absent. Normalizes the shape to [N].
void forward_optional(nvidia::gxf::Entity& out_msg, const holoscan::gxf::Entity& in_msg,
                      const std::string& name, int num_boxes,
                      nvidia::gxf::Handle<nvidia::gxf::Allocator>& alloc, cudaStream_t stream) {
  if (name.empty()) { return; }
  auto in = in_msg.get<Tensor>(name.c_str());
  if (!in) { return; }  // optional input, not present

  if (in->shape().empty() || static_cast<int>(in->shape()[0]) != num_boxes) {
    throw std::runtime_error("DetectionTo3DOp: '" + name + "' length must equal number of boxes");
  }

  auto out_t = out_msg.add<nvidia::gxf::Tensor>(name.c_str());
  const nvidia::gxf::Shape shape{num_boxes};
  const DLDataType dt = in->dtype();
  if (dt.code == kDLFloat && dt.bits == 32) {
    out_t.value()->reshape<float>(shape, nvidia::gxf::MemoryStorageType::kDevice, alloc);
  } else if (dt.code == kDLInt && dt.bits == 32) {
    out_t.value()->reshape<int32_t>(shape, nvidia::gxf::MemoryStorageType::kDevice, alloc);
  } else if (dt.code == kDLInt && dt.bits == 64) {
    out_t.value()->reshape<int64_t>(shape, nvidia::gxf::MemoryStorageType::kDevice, alloc);
  } else if (dt.code == kDLUInt && dt.bits == 8) {
    out_t.value()->reshape<uint8_t>(shape, nvidia::gxf::MemoryStorageType::kDevice, alloc);
  } else {
    throw std::runtime_error("DetectionTo3DOp: unsupported dtype for '" + name + "' passthrough");
  }
  if (num_boxes > 0) {
    CUDA_TRY(cudaMemcpyAsync(out_t.value()->pointer(), in->data(), in->nbytes(),
                             cudaMemcpyDeviceToDevice, stream));
  }
}

}  // namespace

void DetectionTo3DOp::setup(OperatorSpec& spec) {
  auto& detections_in = spec.input<gxf::Entity>("detections");
  auto& depth_in = spec.input<gxf::Entity>("depth");
  // Optional input must not block execution when unconnected.
  auto& intrinsics_in = spec.input<gxf::Entity>("intrinsics").condition(ConditionType::kNone);
  auto& out = spec.output<gxf::Entity>("detections_3d");
  (void)detections_in;
  (void)depth_in;
  (void)intrinsics_in;
  (void)out;

  spec.param(fx_, "fx", "Focal length x", "Focal length in pixels (x).", 0.0f);
  spec.param(fy_, "fy", "Focal length y", "Focal length in pixels (y).", 0.0f);
  spec.param(cx_, "cx", "Principal point x", "Principal point in pixels (x).", 0.0f);
  spec.param(cy_, "cy", "Principal point y", "Principal point in pixels (y).", 0.0f);
  spec.param(depth_scale_, "depth_scale", "Depth scale",
             "Multiplier converting raw depth to meters (e.g. 0.001 for uint16 mm).", 0.001f);
  spec.param(depth_min_, "depth_min", "Min depth", "Minimum valid depth in meters.", 0.0f);
  spec.param(depth_max_, "depth_max", "Max depth", "Maximum valid depth in meters.", 100.0f);
  spec.param(normalized_, "normalized", "Normalized boxes",
             "If true, boxes are in [0, 1] and scaled by (W, H).", false);
  spec.param(patch_size_, "patch_size", "Patch size",
             "Odd median-window size at each box center (clamped to [1, 15]).", 5);
  spec.param(invalid_value_, "invalid_value", "Invalid value",
             "Value written to X/Y/Z for boxes with no valid depth.",
             std::numeric_limits<float>::quiet_NaN());
  spec.param(boxes_tensor_name_, "boxes_tensor_name", "Boxes tensor name",
             "Name of the boxes tensor in the detections message.", std::string("boxes"));
  spec.param(scores_tensor_name_, "scores_tensor_name", "Scores tensor name",
             "Name of the optional scores tensor (passed through).", std::string("scores"));
  spec.param(labels_tensor_name_, "labels_tensor_name", "Labels tensor name",
             "Name of the optional labels tensor (passed through).", std::string("labels"));
  spec.param(depth_tensor_name_, "depth_tensor_name", "Depth tensor name",
             "Name of the depth tensor in the depth message (empty = first tensor).",
             std::string(""));
  spec.param(output_positions_name_, "output_positions_name", "Output positions name",
             "Name of the emitted positions tensor.", std::string("positions"));
  spec.param(allocator_, "allocator", "Allocator", "Device allocator for output tensors.");
}

void DetectionTo3DOp::compute(InputContext& op_input, OutputContext& op_output,
                              ExecutionContext& context) {
  auto detections_message = op_input.receive<gxf::Entity>("detections").value();
  auto depth_message = op_input.receive<gxf::Entity>("depth").value();

  // Acquire this operator's CUDA stream and synchronize BOTH input producers onto it
  // (via events). `detections` and `depth` may come from different upstream operators on
  // different streams, so taking the stream from only one would race the other's writes.
  const cudaStream_t stream = op_input.receive_cuda_stream("depth");
  op_input.receive_cuda_stream("detections");

  // --- Boxes tensor: float32 [N, 4] (xyxy) ---
  auto boxes_tensor = get_tensor(detections_message, boxes_tensor_name_.get());
  const DLDataType bdt = boxes_tensor->dtype();
  if (bdt.code != kDLFloat || bdt.bits != 32) {
    throw std::runtime_error("DetectionTo3DOp: boxes tensor must be float32");
  }
  const auto& bshape = boxes_tensor->shape();
  if (bshape.size() != 2 || static_cast<int>(bshape[1]) != 4) {
    throw std::runtime_error("DetectionTo3DOp: boxes tensor must be [N, 4] (xyxy)");
  }
  const int num_boxes = static_cast<int>(bshape[0]);

  // --- Depth tensor: dtype, dimensions, device pointer ---
  auto depth_tensor = get_tensor(depth_message, depth_tensor_name_.get());
  const DepthDType depth_dtype = to_depth_dtype(depth_tensor->dtype());
  const auto& dshape = depth_tensor->shape();
  if (dshape.size() < 2) {
    throw std::runtime_error("DetectionTo3DOp: depth tensor must be at least 2D [H, W]");
  }
  const int height = static_cast<int>(dshape[0]);
  const int width = static_cast<int>(dshape[1]);

  // --- Optional per-frame intrinsics override ---
  CameraIntrinsics intr{fx_.get(), fy_.get(), cx_.get(), cy_.get()};
  if (auto maybe_intr = op_input.receive<gxf::Entity>("intrinsics")) {
    op_input.receive_cuda_stream("intrinsics");  // sync the intrinsics producer onto `stream`
    auto intr_tensor = get_tensor(maybe_intr.value(), std::string(""));
    const DLDataType idt = intr_tensor->dtype();
    if (intr_tensor->size() < 4 || idt.code != kDLFloat || idt.bits != 32) {
      throw std::runtime_error(
          "DetectionTo3DOp: intrinsics tensor must be float32 [fx, fy, cx, cy]");
    }
    float host[4];
    // Tiny (16 B) config read; pixel data stays GPU-resident. Ordered on `stream`.
    CUDA_TRY(cudaMemcpyAsync(host, intr_tensor->data(), sizeof(host), cudaMemcpyDefault, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));
    intr = CameraIntrinsics{host[0], host[1], host[2], host[3]};
  }

  // Clamp and snap the median window to an odd size so it is symmetric about the center.
  int patch_size = std::clamp(patch_size_.get(), 1, kMaxPatchSize);
  patch_size |= 1;

  // --- Allocate output positions [N, 3] ---
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), allocator_.get()->gxf_cid());
  auto out_message = nvidia::gxf::Entity::New(context.context());

  auto pos_tensor =
      out_message.value().add<nvidia::gxf::Tensor>(output_positions_name_.get().c_str());
  pos_tensor.value()->reshape<float>(nvidia::gxf::Shape{num_boxes, 3},
                                     nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
  if (num_boxes > 0 && !pos_tensor.value()->pointer()) {
    throw std::runtime_error("DetectionTo3DOp: failed to allocate positions tensor");
  }

  CUDA_TRY(launch_lift(static_cast<const float*>(boxes_tensor->data()),
                       num_boxes,
                       depth_tensor->data(),
                       depth_dtype,
                       depth_scale_.get(),
                       normalized_.get(),
                       intr,
                       depth_min_.get(),
                       depth_max_.get(),
                       patch_size,
                       invalid_value_.get(),
                       reinterpret_cast<float3*>(pos_tensor.value()->pointer()),
                       width,
                       height,
                       stream));

  // --- Pass through scores / labels unchanged (alignment preserved with positions[i]) ---
  forward_optional(out_message.value(), detections_message, scores_tensor_name_.get(), num_boxes,
                   allocator.value(), stream);
  forward_optional(out_message.value(), detections_message, labels_tensor_name_.get(), num_boxes,
                   allocator.value(), stream);

  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result, "detections_3d");
}

}  // namespace holoscan::ops
