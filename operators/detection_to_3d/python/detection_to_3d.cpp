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

#include "../../operator_util.hpp"

#include <detection_to_3d.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class providing a Pythonic kwarg-based constructor matching the C++ defaults. */
class PyDetectionTo3DOp : public DetectionTo3DOp {
 public:
  using DetectionTo3DOp::DetectionTo3DOp;

  PyDetectionTo3DOp(Fragment* fragment, const py::args& args,
                    std::shared_ptr<Allocator> allocator, float fx = 0.0f, float fy = 0.0f,
                    float cx = 0.0f, float cy = 0.0f, float depth_scale = 0.001f,
                    float depth_min = 0.0f, float depth_max = 100.0f, bool normalized = false,
                    int patch_size = 5,
                    float invalid_value = std::numeric_limits<float>::quiet_NaN(),
                    const std::string& boxes_tensor_name = "boxes",
                    const std::string& scores_tensor_name = "scores",
                    const std::string& labels_tensor_name = "labels",
                    const std::string& depth_tensor_name = "",
                    const std::string& output_positions_name = "positions",
                    const std::string& name = "detection_to_3d")
      : DetectionTo3DOp(ArgList{Arg{"allocator", allocator},
                                Arg{"fx", fx},
                                Arg{"fy", fy},
                                Arg{"cx", cx},
                                Arg{"cy", cy},
                                Arg{"depth_scale", depth_scale},
                                Arg{"depth_min", depth_min},
                                Arg{"depth_max", depth_max},
                                Arg{"normalized", normalized},
                                Arg{"patch_size", patch_size},
                                Arg{"invalid_value", invalid_value},
                                Arg{"boxes_tensor_name", boxes_tensor_name},
                                Arg{"scores_tensor_name", scores_tensor_name},
                                Arg{"labels_tensor_name", labels_tensor_name},
                                Arg{"depth_tensor_name", depth_tensor_name},
                                Arg{"output_positions_name", output_positions_name}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_detection_to_3d, m) {
  m.doc() = R"pbdoc(
        DetectionTo3DOp Python Bindings
        -------------------------------
        .. currentmodule:: _detection_to_3d
    )pbdoc";

  py::class_<DetectionTo3DOp,
             PyDetectionTo3DOp,
             Operator,
             std::shared_ptr<DetectionTo3DOp>>(
      m, "DetectionTo3DOp",
      "Lift 2D detection boxes into 3D points using a depth image (GPU).")
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<Allocator>,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    bool,
                    int,
                    float,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "fx"_a = 0.0f,
           "fy"_a = 0.0f,
           "cx"_a = 0.0f,
           "cy"_a = 0.0f,
           "depth_scale"_a = 0.001f,
           "depth_min"_a = 0.0f,
           "depth_max"_a = 100.0f,
           "normalized"_a = false,
           "patch_size"_a = 5,
           "invalid_value"_a = std::numeric_limits<float>::quiet_NaN(),
           "boxes_tensor_name"_a = "boxes"s,
           "scores_tensor_name"_a = "scores"s,
           "labels_tensor_name"_a = "labels"s,
           "depth_tensor_name"_a = ""s,
           "output_positions_name"_a = "positions"s,
           "name"_a = "detection_to_3d"s)
      .def("setup", &DetectionTo3DOp::setup, "spec"_a);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
