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
"""Base wrapper for Cooperative Pettingzoo environments."""
from typing import Any, Dict

import numpy as np

from og_marl.environments.base import BaseEnvironment, Observations, ResetReturn, StepReturn


class PettingZooBase(BaseEnvironment):

    """Environment wrapper for PettingZoo environments."""

    def __init__(self) -> None:
        """Constructor."""
        self.info_spec: Dict[str, Any] = {}

    def reset(self) -> ResetReturn:
        """Resets the environment.

        Returns:
            ResetReturn: the initial observations and info.
        """
        # Reset the underlying environment
        observations = self._environment.reset()  # type: ignore

        # Global state
        env_state = self._create_state_representation(observations)

        # Infos
        info = {"state": env_state}

        # Convert observations to OLT format
        observations = self._convert_observations(observations, False)

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        """Steps the environment.

        Args:
            actions (Dict[str, np.ndarray]): Actions taken by the agents.

        Returns:
            StepReturn: the next observations, rewards, terminals, truncations and info.
        """
        # Step the environment
        observations, rewards, terminals, truncations, _ = self._environment.step(actions)

        # Global state
        env_state = self._create_state_representation(observations)

        # Extra infos
        info = {"state": env_state}

        return observations, rewards, terminals, truncations, info

    def _add_zero_obs_for_missing_agent(self, observations: Observations) -> Observations:
        """TODO

        Args:
            observations (Observations): _description_

        Returns:
            Observations: _description_
        """
        for agent in self._agents:
            if agent not in observations:
                observations[agent] = np.zeros(
                    self.observation_spaces[agent].shape,  # type: ignore
                    self.observation_spaces[agent].dtype,  # type: ignore
                )
        return observations

    def _convert_observations(
        self, observations: Dict[str, np.ndarray], done: bool
    ) -> Dict[str, np.ndarray]:
        """TODO

        Args:
            observations (Dict[str, np.ndarray]): _description_
            done (bool): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Dict[str, np.ndarray]: _description_
        """
        raise NotImplementedError

    def _create_state_representation(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Create global state representation from agent observations.

        Args:
            observations (Dict[str, np.ndarray]): Observations from the agents.

        Raises:
            NotImplementedError: Abstract class.

        Returns:
            np.ndarray: Global state representation.
        """
        raise NotImplementedError
