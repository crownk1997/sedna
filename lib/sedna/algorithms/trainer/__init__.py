# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Aggregation algorithms"""

# TODO: merge with aggregation

import abc
from copy import deepcopy

import numpy as np

from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('BasicTrainer')

from plato.trainers import basic

class BasicTrainer(basic.Trainer):
    def __init__(self):
        super().__init__()
