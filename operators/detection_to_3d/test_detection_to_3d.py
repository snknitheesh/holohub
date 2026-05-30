# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the DetectionTo3DOp Python bindings.

Covers construction, port wiring, parameter handling, and error handling across the
pybind11 boundary. Numerical correctness of the lift kernel is covered separately by
the standalone CUDA golden-reference test (``test/test_lift.cu``) and the
``detection_to_3d_demo`` application, since the compute path requires real GXF
entities, an allocator, and a CUDA stream that the lightweight fixtures do not provide.
"""

import pytest
from holoscan.core import Operator
from holoscan.resources import UnboundedAllocator

from holohub.detection_to_3d import DetectionTo3DOp

try:
    from holoscan.core import BaseOperator
except ImportError:
    from holoscan.core import _Operator as BaseOperator


def _make_op(fragment, **overrides):
    """Construct a DetectionTo3DOp with sane defaults, overridable per test."""
    params = dict(
        name="detection_to_3d",
        allocator=UnboundedAllocator(fragment, name="alloc"),
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )
    params.update(overrides)
    return DetectionTo3DOp(fragment, **params)


def test_init(fragment):
    """Operator constructs and exposes the expected Holoscan properties."""
    name = "d23_op"
    op = _make_op(fragment, name=name)
    assert isinstance(op, BaseOperator), "DetectionTo3DOp should be a Holoscan operator"
    assert op.operator_type == Operator.OperatorType.NATIVE, "Operator type should be NATIVE"
    assert f"name: {name}" in repr(op), "Operator name should appear in repr()"


def test_ports(fragment):
    """setup() wires the required inputs, the optional intrinsics input, and the output."""
    spec = _make_op(fragment).spec
    assert "detections" in spec.inputs, 'required input "detections" missing'
    assert "depth" in spec.inputs, 'required input "depth" missing'
    assert "intrinsics" in spec.inputs, 'optional "intrinsics" input missing'
    assert "detections_3d" in spec.outputs, 'output "detections_3d" missing'


def test_requires_allocator(fragment):
    """allocator is a required argument; omitting it is a TypeError."""
    with pytest.raises(TypeError):
        DetectionTo3DOp(fragment, name="no_alloc", fx=500.0, fy=500.0, cx=320.0, cy=240.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"depth_scale": 0.001, "depth_min": 0.1, "depth_max": 10.0},
        {"normalized": True, "patch_size": 7},
        {"patch_size": 1, "invalid_value": 0.0},
        {
            "boxes_tensor_name": "boxes",
            "scores_tensor_name": "scores",
            "labels_tensor_name": "labels",
        },
        {"output_positions_name": "xyz"},
    ],
)
def test_param_acceptance(fragment, overrides):
    """The operator accepts its documented parameters without error."""
    op = _make_op(fragment, **overrides)
    assert isinstance(op, BaseOperator)
