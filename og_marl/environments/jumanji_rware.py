# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base wrapper for Jumanji environments."""
from typing import Any, Dict

import numpy as np
import jax

from og_marl.environments.base import BaseEnvironment, ResetReturn, StepReturn


class JumanjiBase(BaseEnvironment):

    """Environment wrapper for Jumanji environments."""

    def __init__(self) -> None:
        """Constructor."""
        self._environment = ...
        self._num_agents = self._environment.num_agents
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self._state = ...  # Jumanji environment state

        self.info_spec: Dict[str, Any] = {}

    def reset(self) -> ResetReturn:
        """Resets the env."""
        # Reset the environment
        self.key, sub_key = jax.random.split(self.key)
        self._state, timestep = self._environment.reset(sub_key)

        # Infos
        info = {"state": env_state}

        # Convert observations to OLT format
        observations = self._convert_observations(observations, False)

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Steps in env."""
        actions = ...  # convert actions
        # Step the environment
        self._state, timestep = self._env.step(self._state, actions)

        # Global state
        env_state = self._create_state_representation(observations)

        # Extra infos
        info = {"state": env_state}

        return observations, rewards, terminals, truncations, info
