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

import time
from typing import Dict

import numpy as np
from chex import Numeric

from og_marl.wrapped_environments.base import BaseEnvironment
from og_marl.loggers import BaseLogger
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer


class BaseOfflineSystem:
    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
    ):
        self.environment = environment  # only used for evaluation
        self.logger = logger

        self.agents = environment.agents

        self.training_step_ctr = 0

    def evaluate(self, num_eval_episodes: int = 32) -> Dict[str, Numeric]:
        """Method to evaluate the system in the environment."""
        episode_returns = []
        for _ in range(num_eval_episodes):
            self.reset()
            observations, infos = self.environment.reset()

            done = False
            episode_return = 0.0
            while not done:
                if "legals" in infos:
                    legal_actions = infos["legals"]
                else:
                    legal_actions = None

                actions = self.select_actions(observations, legal_actions)

                observations, rewards, terminal, truncation, infos = self.environment.step(actions)

                episode_return += np.mean(list(rewards.values()), dtype="float")

                done = all(terminal.values()) or all(truncation.values())

            episode_returns.append(episode_return)

        logs = {
            "evaluation/mean_episode_return": np.mean(episode_returns),
            "evaluation/max_episode_return": np.max(episode_returns),
            "evaluation/min_episode_return": np.min(episode_returns),
        }
        return logs

    def train(
        self,
        replay_buffer: FlashbaxReplayBuffer,
        training_steps: int = int(1e5),
        evaluation_every: int = 5000,
        num_eval_episodes: int = 32,
    ) -> None:
        """Method to train the system offline."""
        for _ in range(training_steps):
            if evaluation_every is not None and self.training_step_ctr % evaluation_every == 0:
                print("EVALUATION")
                eval_logs = self.evaluate(num_eval_episodes)
                self.logger.write(
                    eval_logs | {"evaluation/training_steps": self.training_step_ctr}, force=True
                )

            start_time = time.time()
            experience = replay_buffer.sample()
            end_time = time.time()
            time_to_sample = end_time - start_time

            start_time = time.time()
            train_logs = self.train_step(experience)
            end_time = time.time()
            time_train_step = end_time - start_time

            train_steps_per_second = 1 / (time_train_step + time_to_sample)

            logs = {
                **train_logs,
                "training_steps": self.training_step_ctr,
                "time_for_sampling": time_to_sample,
                "time_for_training_step": time_train_step,
                "training_sps": train_steps_per_second,
            }

            self.logger.write(logs)

            self.training_step_ctr += 1

        print("FINAL EVALUATION")
        eval_logs = self.evaluate(10 * num_eval_episodes)
        self.logger.write(
            eval_logs | {"evaluation/training_steps": self.training_step_ctr}, force=True
        )

    def reset(self) -> None:
        """Called at the start of each new episode during evaluation."""
        return

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        raise NotImplementedError
