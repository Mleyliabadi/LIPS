# copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

from typing import Union

import grid2op
import re

from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Runner import Runner
from grid2op.PlotGrid import PlotMatplot

from lips.physical_simulator.physicalSimulator import PhysicalSimulator


class Grid2opSimulator(PhysicalSimulator):
    """
    This simulator uses the `grid2op` package to implement a physical simulator.

    It accepts both grid2op BaseAction and grid2op BaseAgent to modify the internal state of the simulator
    """
    def __init__(self,
                 env_kwargs: dict,
                 initial_chronics_id: Union[int, None] = None,  # the initial chronic id
                 chronics_selected_regex: str = None,  # the chronics to keep for this simulator
                 ):
        PhysicalSimulator.__init__(self, actor_types=(BaseAction, BaseAgent))
        self._simulator = grid2op.make(**env_kwargs)
        self._simulator.deactivate_forecast()
        if chronics_selected_regex is not None:
            # special case of the grid2Op environment: data are read from chronics that should be part of the dataset
            # here i keep only certain chronics for the training, and the other for the test
            chronics_selected_regex = re.compile(chronics_selected_regex)
            self._simulator.chronics_handler.set_filter(lambda path:
                                                        re.match(chronics_selected_regex, path) is not None)
            self._simulator.chronics_handler.real_data.reset()

        if initial_chronics_id is not None:
            self._simulator.set_id(initial_chronics_id)

        self._obs = None
        self._reward = None
        self._info = None
        self._reset_simulator()
        self._plot_helper = None

        self._nb_divergence = 0  # number of failures of modify_state
        self._nb_output = 0  # number of time get_state is called

    def seed(self, seed: int):
        """
        It seeds the environment, for reproducible experiments.
        Parameters
        ----------
        seed:
            An integer representing the seed.

        """
        seeds = self._simulator.seed(seed)
        self._reset_simulator()
        return seeds

    def get_state(self):
        """
        The state of the powergrid is, for this class, represented by a tuple:
        - grid2op observation.
        - extra information (can be empty)

        """
        self._nb_output += 1
        return self._obs, self._info

    def modify_state(self, actor):
        """
        It calls `env.step` until a convergence is obtained.
        """
        super().modify_state(actor)  # perform the check that the actor is legit
        done = True
        while done:
            # simulate data (resimulate in case of divergence of the simulator)
            act = actor.act(self._obs, self._reward, done)
            self._obs, self._reward, done, self._info = self._simulator.step(act)
            if self._info["is_illegal"]:
                raise RuntimeError("Your `actor` should not take illegal action. Please modify the environment "
                                   "or your actor.")
            if done:
                self._nb_divergence += 1
                self._reset_simulator()

    def visualize_network(self):
        """
        This functions shows the network state evolution over time for a given dataset
        """
        if self._plot_helper is None:
            self._plot_helper = PlotMatplot(self._simulator.observation_space)
        return self._plot_helper.plot_layout()

    def visualize_network_reference_topology(self,
                                             action: Union[BaseAction, None] = None,
                                             **plot_kwargs):
        """
        visualize the power network's reference topology
        """
        if self._plot_helper is None:
            self._plot_helper = PlotMatplot(self._simulator.observation_space)
        env = self._simulator.copy()

        obs = env.reset()
        if action is not None:
            obs, _, _, _ = env.step(action)

        fig = self._plot_helper.plot_obs(obs, **plot_kwargs)
        return fig

    def _reset_simulator(self):
        self._obs = self._simulator.reset()
        self._reward = self._simulator.reward_range[0]
        self._info = {}


