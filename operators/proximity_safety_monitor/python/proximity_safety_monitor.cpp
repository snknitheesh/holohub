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

#include <proximity_safety_monitor.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

class PyProximitySafetyMonitorOp : public ProximitySafetyMonitorOp {
 public:
  using ProximitySafetyMonitorOp::ProximitySafetyMonitorOp;

  PyProximitySafetyMonitorOp(
      Fragment* fragment, const py::args& args, std::shared_ptr<Allocator> allocator,
      float stop_distance = 0.3f, float warn_distance = 0.6f, float depth_scale = 0.001f,
      float depth_min = 0.0f, float depth_max = 100.0f, int min_trigger_pixels = 50,
      const std::string& distance_mode = "z", float fx = 0.0f, float fy = 0.0f, float cx = 0.0f,
      float cy = 0.0f, const std::string& depth_tensor_name = "",
      const std::string& mask_tensor_name = "",
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
      const std::string& name = "proximity_safety_monitor")
      : ProximitySafetyMonitorOp(ArgList{Arg{"allocator", allocator},
                                         Arg{"stop_distance", stop_distance},
                                         Arg{"warn_distance", warn_distance},
                                         Arg{"depth_scale", depth_scale},
                                         Arg{"depth_min", depth_min},
                                         Arg{"depth_max", depth_max},
                                         Arg{"min_trigger_pixels", min_trigger_pixels},
                                         Arg{"distance_mode", distance_mode},
                                         Arg{"fx", fx},
                                         Arg{"fy", fy},
                                         Arg{"cx", cx},
                                         Arg{"cy", cy},
                                         Arg{"depth_tensor_name", depth_tensor_name},
                                         Arg{"mask_tensor_name", mask_tensor_name}}) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_proximity_safety_monitor, m) {
  m.doc() = R"pbdoc(
        ProximitySafetyMonitorOp Python Bindings
        ----------------------------------------
        .. currentmodule:: _proximity_safety_monitor
    )pbdoc";

  py::class_<ProximitySafetyMonitorOp,
             PyProximitySafetyMonitorOp,
             Operator,
             std::shared_ptr<ProximitySafetyMonitorOp>>(
      m, "ProximitySafetyMonitorOp",
      "Perception-based proximity safety monitor (NOT a certified safety device).")
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<Allocator>,
                    float,
                    float,
                    float,
                    float,
                    float,
                    int,
                    const std::string&,
                    float,
                    float,
                    float,
                    float,
                    const std::string&,
                    const std::string&,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "stop_distance"_a = 0.3f,
           "warn_distance"_a = 0.6f,
           "depth_scale"_a = 0.001f,
           "depth_min"_a = 0.0f,
           "depth_max"_a = 100.0f,
           "min_trigger_pixels"_a = 50,
           "distance_mode"_a = "z"s,
           "fx"_a = 0.0f,
           "fy"_a = 0.0f,
           "cx"_a = 0.0f,
           "cy"_a = 0.0f,
           "depth_tensor_name"_a = ""s,
           "mask_tensor_name"_a = ""s,
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "proximity_safety_monitor"s)
      .def("setup", &ProximitySafetyMonitorOp::setup, "spec"_a);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
