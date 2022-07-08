import os
import sys
sys.path.insert(0, "../")

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)

import pandas as pd
import warnings

import numpy as np
import grid2op
import copy

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import pathlib
from lips.augmented_simulators.tensorflow_models import TfFullyConnected#, TfFullyConnectedTopoEncoding
from lips.dataset.scaler import StandardScaler

from pprint import pprint
from matplotlib import pyplot as plt
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.utils import get_path
from lips.augmented_simulators.tensorflow_models import LeapNet
from lips.dataset.scaler import PowerGridScaler, StandardScaler
from sklearn.preprocessing import MinMaxScaler

from lips.config import ConfigManager

#from execution_jobs.utils import init_df_bench1, append_metrics_to_df_bench1, init_df_bench2, append_metrics_to_df_bench2, init_df_bench3, append_metrics_to_df_bench3, filter_bench1, filter_bench2_3

# indicate required paths
# indicate required paths
LIPS_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()
DATA_PATH = LIPS_PATH / "lips" / "tests" / "data" / "powergrid" / "l2rpn_case14_sandbox"
BENCH_CONFIG_PATH = LIPS_PATH / "lips" / "tests" / "configs" / "powergrid" / "benchmarks" / "l2rpn_case14_sandbox.ini"
SIM_CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "simulators"
BASELINES_PATH = LIPS_PATH / "trained_baselines" / "powergrid"
TRAINED_MODEL_PATH = LIPS_PATH / "trained_models" / "powergrid"
EVALUATION_PATH = LIPS_PATH / "evaluation_results" / "PowerGrid"
LOG_PATH = LIPS_PATH / "lips_logs.log"

benchmark1 = PowerGridBenchmark(benchmark_name="Benchmark2",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=BENCH_CONFIG_PATH
                               )

from lips.augmented_simulators.tensorflow_models import LeapNet

from lips.augmented_simulators.tensorflow_models import LeapNet

def test_fast_transform_tau():
    """
    In this test, we check that we are able to consistently get the same encoding as the one in leap_net package but for a faster vectorized method
    """
    bench_config = ConfigManager(section_name="Benchmark1", path=BENCH_CONFIG_PATH)
    topo_actions = bench_config.get_option("dataset_create_params")["reference_args"]["topo_actions"]

    kwargs_tau = []
    for el in topo_actions:
         kwargs_tau.append(el["set_bus"]["substations_id"][0])

    leap_net1 = LeapNet(name="tf_leapnet",

                        bench_config_path=BENCH_CONFIG_PATH,
                        bench_config_name="Benchmark2",
                        sim_config_path=SIM_CONFIG_PATH / "tf_leapnet.ini",
                        sim_config_name="DEFAULT",
                        log_path=LOG_PATH,

                        loss={"name": "mse"},
                        lr=1e-4,
                        activation=tf.keras.layers.LeakyReLU(alpha=0.01),

                        sizes_enc=(),
                        sizes_main=(150, 150),
                        sizes_out=(),
                        topo_vect_to_tau="given_list",
                        is_topo_vect_input=False,
                        kwargs_tau=kwargs_tau,
                        layer="resnet",
                        attr_tau=("line_status","topo_vect"),
                        scale_main_layer=150,
                        # scale_input_enc_layer = 40,
                        #scale_input_dec_layer=200,
                        # topo_vect_as_input = True,
                        mult_by_zero_lines_pred=False,
                        topo_vect_as_input=True,
                        scaler=PowerGridScaler,

                        )

    ## add topo_vect (temporary ) in attr_x in benchmark config file


    ## add topo_vect (temporary ) in attr_x in benchmark config file
    ##############
    nb_timesteps_to_test=1000
    indices=[i for i in range(nb_timesteps_to_test)]
    for key in benchmark1.train_dataset.data.keys():
        benchmark1.train_dataset.data[key]=benchmark1.train_dataset.data[key][indices]

    benchmark1.train_dataset.size=len(indices)

    leap_net1._leap_net_model.max_row_training_set=len(indices)
    dataset=benchmark1.train_dataset
    obss = leap_net1._make_fake_obs(dataset)
    leap_net1._leap_net_model.init(obss)


    (extract_x, extract_tau), extract_y = leap_net1.scaler.fit_transform(dataset)

    #####
    #Launch two different mathods for transformation and check that they match
    tau = copy.deepcopy(extract_tau)
    import time
    start = time.time()
    extract_tau_1 = leap_net1._transform_tau(dataset, extract_tau)
    end = time.time()
    print(end - start) #3.9s pour 10 000 / Pour 100 000 35.63s

    start = time.time()
    extract_tau_bis = leap_net1._transform_tau_given_list(tau)
    end = time.time()
    print(end - start) #0.026s pour 10 000 => 150 fois plus rapide! Pour 100 000, 1,7s
    #0.021s with int32, could we only work with boolean ? 0.32s for 100 000 (or 0.82s with numpy matmult, a bit faster with tensorflow then)

    #check that there is no multiple topologies encoded for a given substation at the same time
    sub_encoding_pos = np.array([topo_action[0] for topo_action in kwargs_tau])
    for sub in set(sub_encoding_pos):
        indices = np.where(sub_encoding_pos == sub)[0]
        if (len(indices) >= 2):
            nb_topology_activated_per_timestep= extract_tau_1[1][indices, :].sum(axis=1)
            assert np.any(nb_topology_activated_per_timestep>=2)


    assert np.all(extract_tau_1[1].astype((np.bool_))==extract_tau_bis[1])

def test_fast_transform_tau_multiple_line_disconnect():
    """
    In this test, we check that we are able to always get at most one topology per substation activated in the encoding.
    If two or more such topologies at a given substation have been matched, only one should be chosen
    """
    bench_config = ConfigManager(section_name="Benchmark1", path=BENCH_CONFIG_PATH)
    topo_actions=[{'set_bus': {'substations_id': [(4, (2, 1, 2, 1, 2))]}},
                  {'set_bus': {'substations_id': [(1, (1, 2, 1, 2, 2, 2))]}},
                  {'set_bus': {'substations_id': [(5, (1, 1, 2, 2, 1, 2, 2))]}},
                  {'set_bus': {'substations_id': [(5, (1, 2, 1, 2, 1, 2, 2))]}}]#the last action is added to test that the two action on sub 5 do not appear simultaneously in tau encoding


    kwargs_tau = []
    for el in topo_actions:
         kwargs_tau.append(el["set_bus"]["substations_id"][0])
    tau=[[],[]]
    #here we have a match fro the first topology at substation 5
    tau[1].append(np.array([1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1]))

    #here the two topologies at substation 5 will be matched (given 2 disconnected lines that create that ambiguity,
    # but only one should be activated in the encoding
    tau[1].append(np.array([1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 2, 1, 2, 2, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1]))

    tau[1]=np.array(tau[1])

    leap_net1 = LeapNet(name="tf_leapnet",
                        bench_config_path=BENCH_CONFIG_PATH,
                        bench_config_name="Benchmark2",
                        sim_config_path=SIM_CONFIG_PATH / "tf_leapnet.ini",
                        sim_config_name="DEFAULT",
                        log_path=LOG_PATH,
                        loss={"name": "mse"},
                        lr=1e-4,
                        activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                        sizes_enc=(),
                        sizes_main=(150, 150),
                        sizes_out=(),
                        topo_vect_to_tau="given_list",
                        is_topo_vect_input=False,
                        kwargs_tau=kwargs_tau,
                        layer="resnet",
                        attr_tau=("line_status","topo_vect"),
                        scale_main_layer=150,
                        # scale_input_enc_layer = 40,
                        #scale_input_dec_layer=200,
                        # topo_vect_as_input = True,
                        mult_by_zero_lines_pred=False,
                        topo_vect_as_input=True,
                        scaler=PowerGridScaler,
                        )
    leap_net1._leap_net_model.subs_index=[(0, 3), (3, 9), (9, 13), (13, 19), (19, 24), (24, 31), (31, 34), (34, 36), (36, 41), (41, 44), (44, 47), (47, 50), (50, 54), (54, 57)]
    extract_tau = leap_net1._transform_tau_given_list(tau)

    #check that there is no multiple topologies encoded for a given substation at the same time
    sub_encoding_pos = np.array([topo_action[0] for topo_action in kwargs_tau])
    for sub in set(sub_encoding_pos):
        indices = np.where(sub_encoding_pos == sub)[0]
        if (len(indices) >= 2):
            nb_topology_activated_per_timestep= extract_tau[1][:,indices].sum(axis=1)
            assert np.all(nb_topology_activated_per_timestep<=1) #maximum 1 topology activated per substation
