# Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LIPS, LIPS is a python platform for power networks benchmarking

"""
Set of utility function to generate the data for NeuripsBenchmark1
"""
import warnings
import grid2op

from lips.neurips_benchmark.actor_utils import ChangeTopoRefN1, ChangeTopoRefN2, ChangeTopoRefN1Ref

# for reference
REF_ACTION = [# sub_5_id1
              {"set_bus": {"loads_id": [(4, 2)], "generators_id": [(2, 2)], "lines_or_id": [(9, 2), (8, 2)]}},
              # sub_4_id1
              {"set_bus": {"loads_id": [(3, 2)], "lines_or_id": [(17, 2)], "lines_ex_id": [(6, 2)]}},
              {"set_bus": {"lines_or_id": [(17, 2)], "lines_ex_id": [(4, 2)]}},  # sub_4_id2
              {"set_bus": {"lines_or_id": [(15, 2)], "lines_ex_id": [(3, 2)]}},  # "sub3_id1"
              {"set_bus": {"lines_or_id": [(16, 2)], "lines_ex_id": [(3, 2)]}},  # "sub3_id2"
              {"set_bus": {"lines_or_id": [(6, 2), (16, 2)]}},  # "sub3_id3"
              {"set_bus": {"lines_or_id": [(15, 2)], "lines_ex_id": [(3, 2)], "loads_id": [(2, 2)]}},  # "sub3_id4"
              # "sub8_id1"
              {"set_bus": {"lines_or_id": [(10, 2), (19, 2)], "loads_id": [(5, 2)]}},
              # "sub8_id2"
              {"set_bus": {"lines_or_id": [(10, 2), (19, 2)]}},
              # "sub8_id3"
              {"set_bus": {"lines_or_id": [(10, 2)], "loads_id": [(5, 2)]}},
              {"set_bus": {"lines_or_id": [(4, 2), (2, 2)]}},  # "sub1_id1" \n",
              {"set_bus": {"lines_or_id": [(3, 2)], "generators_id": [(0, 2)]}},  # "sub1_id2"
              {"set_line_status": [(14, -1)]},  # "powerline_9_10"
              {"set_line_status": [(12, -1)]},  # "powerline_12_13"
              {"set_line_status": [(3, -1)]},  # "powerline_1_3"
              # composed action sub_5_id1 + sub_4_id1
              {"set_bus": {"loads_id": [(4, 2), (3, 2)],
                           "generators_id": [(2, 2)],
                           "lines_or_id": [(9, 2), (8, 2), (17, 2)],
                           "lines_ex_id": [(6, 2)]}},
              # composed action sub_5_id1 + sub8_id1
              {"set_bus": {"loads_id": [(4, 2), (5, 2)],
                           "generators_id": [(2, 2)],
                           "lines_or_id": [(9, 2), (8, 2), (10, 2), (19, 2)],
                           "lines_ex_id": []}},
              # composed action sub_4_id1 + sub8_id1
              {"set_bus": {"loads_id": [(3, 2), (5, 2)],
                           "lines_or_id": [(17, 2), (10, 2), (19, 2)],
                           "lines_ex_id": [(6, 2)]}},
              # composed action sub_1_id1 + sub_3_id4
              {"set_bus": {"loads_id": [(2, 2)],
                           "lines_or_id": [(15, 2), (4, 2), (2, 2)],
                           "lines_ex_id": [(3, 2)]}},
              ]


def get_kwargs_simulator_scenario():
    """
    This function return the
    Returns
    -------

    """
    try:
        from lightsim2grid import LightSimBackend
        BkCls = LightSimBackend
    except ImportError as exc_:
        from grid2op.Backend import PandaPowerBackend
        BkCls = PandaPowerBackend

    env_name = "l2rpn_case14_sandbox"
    # create a temporary environment to retrieve the default parameters of this specific environment
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env_tmp = grid2op.make(env_name)
    param = env_tmp.parameters
    param.NO_OVERFLOW_DISCONNECTION = True
    # i can act on all powerline / substation at once
    param.MAX_LINE_STATUS_CHANGED = 999999
    param.MAX_SUB_CHANGED = 999999
    # i can act every step on every line / substation (no cooldown)
    param.NB_TIMESTEP_COOLDOWN_LINE = 0
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    return {"dataset": env_name,
            "param": param,
            "backend": BkCls()}


def _aux_act_scenario(env):
    """
    It returns singular topo actions (acting on one sub)
    """
    li_ref_topo = [{"set_line_status": [(l_id, -1)]} for l_id in range(env.n_line)]
    li_act_n1 = [REF_ACTION[0], REF_ACTION[1], REF_ACTION[6], REF_ACTION[7], REF_ACTION[10]]
    li_ref_topo = [env.action_space(el) for el in li_ref_topo]
    li_act_n1 = [env.action_space(el) for el in li_act_n1]
    return li_ref_topo, li_act_n1

def _aux_act_scenario_ood(env):
    """
    It returns composed topo actions (acting on two subs) to test the out-of-distribution
    """
    li_ref_topo = [{"set_line_status": [(l_id, -1)]} for l_id in range(env.n_line)]
    li_act_n1 = [REF_ACTION[15], REF_ACTION[16], REF_ACTION[17], REF_ACTION[18]]
    li_ref_topo = [env.action_space(el) for el in li_ref_topo]
    li_act_n1 = [env.action_space(el) for el in li_act_n1]
    return li_ref_topo, li_act_n1

def get_actor_training_scenario(simulator):
    env = simulator._simulator
    li_ref_topo, li_act_n1 = _aux_act_scenario(env)
    agent = ChangeTopoRefN1Ref(env.action_space,
                               p=0.5,
                               ref_topo=li_ref_topo,
                               list_act=li_act_n1)
    return agent


def get_actor_test_scenario(simulator):
    env = simulator._simulator
    li_ref_topo, li_act_n1 = _aux_act_scenario(env)
    agent = ChangeTopoRefN1(env.action_space,
                            ref_topo=li_ref_topo,
                            list_act=li_act_n1)
    return agent


def get_actor_test_ood_topo_scenario(simulator):
    env = simulator._simulator
    li_ref_topo, li_act_n1 = _aux_act_scenario_ood(env)
    agent = ChangeTopoRefN1(env.action_space,
                            ref_topo=li_ref_topo,
                            list_act=li_act_n1)
    return agent
