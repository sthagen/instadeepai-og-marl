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

from typing import Any, Dict

import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import tree
from chex import Array
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flashbax.vault import Vault

Experience = Dict[str, Array]


class FlashbaxReplayBuffer:
    def __init__(
        self,
        sequence_length: int,
        max_size: int = 50_000,
        batch_size: int = 32,
        sample_period: int = 1,
        seed: int = 42,
    ):
        """_summary_

        Args:
            sequence_length (int): _description_
            max_size (int, optional): _description_. Defaults to 50_000.
            batch_size (int, optional): _description_. Defaults to 32.
            sample_period (int, optional): _description_. Defaults to 1.
            seed (int, optional): _description_. Defaults to 42.
        """
        self._sequence_length = sequence_length
        self._max_size = max_size
        self._batch_size = batch_size

        # Flashbax buffer
        self._replay_buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=batch_size,
            sample_sequence_length=sequence_length,
            period=sample_period,
            min_length_time_axis=1,
            max_size=max_size,
        )

        self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)
        self._buffer_add_fn = jax.jit(self._replay_buffer.add)

        self._buffer_state: TrajectoryBufferState = None
        self._rng_key = jax.random.PRNGKey(seed)

    def add(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, np.ndarray],
        terminals: Dict[str, np.ndarray],
        truncations: Dict[str, np.ndarray],
        infos: Dict[str, Any],
    ) -> None:
        """_summary_

        Args:
            observations (Dict[str, np.ndarray]): _description_
            actions (Dict[str, np.ndarray]): _description_
            rewards (Dict[str, np.ndarray]): _description_
            terminals (Dict[str, np.ndarray]): _description_
            truncations (Dict[str, np.ndarray]): _description_
            infos (Dict[str, Any]): _description_
        """
        timestep = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncations,
            "infos": infos,
        }

        if self._buffer_state is None:
            self._buffer_state = self._replay_buffer.init(timestep)

        timestep = tree.map_structure(
            lambda x: jnp.expand_dims(jnp.expand_dims(jnp.array(x), 0), 0), timestep
        )  # add batch & time dims
        self._buffer_state = self._buffer_add_fn(self._buffer_state, timestep)

    def sample(self) -> Experience:
        """_summary_

        Returns:
            Experience: _description_
        """
        self._rng_key, sample_key = jax.random.split(self._rng_key, 2)
        batch = self._buffer_sample_fn(self._buffer_state, sample_key)
        return batch.experience  # type: ignore

    def populate_from_vault(
        self, env_name: str, scenario_name: str, dataset_name: str, rel_dir: str = "vaults"
    ) -> bool:
        """_summary_

        Args:
            env_name (str): _description_
            scenario_name (str): _description_
            dataset_name (str): _description_
            rel_dir (str, optional): _description_. Defaults to "vaults".

        Returns:
            bool: _description_
        """
        try:
            self._buffer_state = Vault(
                vault_name=f"{env_name}/{scenario_name}.vlt",
                vault_uid=dataset_name,
                rel_dir=rel_dir,
            ).read()

            # Recreate the buffer and associated pure functions
            self._replay_buffer = fbx.make_trajectory_buffer(
                add_batch_size=1,
                sample_batch_size=self._batch_size,
                sample_sequence_length=self._sequence_length,
                period=1,
                min_length_time_axis=1,
                max_size=self._sequence_length,
            )
            self._buffer_sample_fn = jax.jit(self._replay_buffer.sample)
            self._buffer_add_fn = jax.jit(self._replay_buffer.add)

            return True

        except ValueError:
            return False
