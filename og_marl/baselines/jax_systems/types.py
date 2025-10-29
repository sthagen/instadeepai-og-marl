# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from typing import Any, Callable, Dict, Generic, Optional, Protocol, Tuple, TypeVar, Union

import chex
# import jumanji.specs as specs
from flax.core.frozen_dict import FrozenDict
# from jumanji.types import TimeStep
from typing_extensions import NamedTuple, TypeAlias

Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array
# Can't know the exact type of State.
State: TypeAlias = Any
Metrics: TypeAlias = Dict[str, chex.Array]

class Observation(NamedTuple):
    """The observation that the agent sees.

    agents_view: the agent's view of the environment.
    action_mask: boolean array specifying, for each agent, which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    step_count: Optional[chex.Array] = None  # (num_agents, )


class ObservationGlobalState(NamedTuple):
    """The observation seen by agents in centralised systems.

    Extends `Observation` by adding a `global_state` attribute for centralised training.
    global_state: The global state of the environment, often a concatenation of agents' views.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    global_state: chex.Array  # (num_agents, num_agents * num_obs_features)
    step_count: Optional[chex.Array] = None  # (num_agents, )


RNNObservation: TypeAlias = Tuple[Observation, Done]
RNNGlobalObservation: TypeAlias = Tuple[ObservationGlobalState, Done]
MavaObservation: TypeAlias = Union[Observation, ObservationGlobalState]

# `MavaState` is the main type passed around in our systems. It is often used as a scan carry.
# Types like: `LearnerState` (mava/systems/<system_name>/types.py) are `MavaState`s.
MavaState = TypeVar("MavaState")
MavaTransition = TypeVar("MavaTransition")


class ExperimentOutput(NamedTuple, Generic[MavaState]):
    """Experiment output."""

    learner_state: MavaState
    episode_metrics: Metrics
    train_metrics: Metrics


LearnerFn = Callable[[MavaState], ExperimentOutput[MavaState]]
SebulbaLearnerFn = Callable[[MavaState, MavaTransition], ExperimentOutput[MavaState]]
ActorApply = Callable[[FrozenDict, Observation], Any]
CriticApply = Callable[[FrozenDict, Observation], Value]
RecActorApply = Callable[
    [FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, Any]
]
RecCriticApply = Callable[[FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, Value]]
