# python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Sweep definition for catch_noise experiment."""

from bsuite.experiments.catch import sweep as catch_sweep

NUM_EPISODES = catch_sweep.NUM_EPISODES

_settings = []
for scale in [0.1, 0.3, 1.0, 3., 10.]:
  for seed_id in range(4):
    _settings.append({'noise_scale': 0.2, 'seed': seed_id})

SETTINGS = tuple(_settings)
TAGS = ('noise', 'credit_assignment')
