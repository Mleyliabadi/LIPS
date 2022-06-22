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

from IPython.display import display
import pandas as pd
import warnings

import numpy as np
import grid2op

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.set_option('display.max_columns', None)
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
LIPS_PATH = pathlib.Path().resolve().parent # it is supposed that the notebook had run from getting_started folder
DATA_PATH = LIPS_PATH / "reference_data" / "powergrid" / "l2rpn_neurips_2020_track1_small"
BENCH_CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "benchmarks" / "l2rpn_neurips_2020_track1_small.ini"
SIM_CONFIG_PATH = LIPS_PATH / "configurations" / "powergrid" / "simulators"
BASELINES_PATH = LIPS_PATH / "trained_baselines" / "powergrid"
TRAINED_MODEL_PATH = LIPS_PATH / "trained_models" / "powergrid"
EVALUATION_PATH = LIPS_PATH / "evaluation_results" / "PowerGrid"
LOG_PATH = LIPS_PATH / "lips_logs.log"

benchmark1 = PowerGridBenchmark(benchmark_name="Benchmark1",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=BENCH_CONFIG_PATH
                               )

from lips.augmented_simulators.tensorflow_models import LeapNet

from lips.augmented_simulators.tensorflow_models import LeapNet

bench_config = ConfigManager(section_name="Benchmark1", path=BENCH_CONFIG_PATH)
topo_actions = bench_config.get_option("dataset_create_params")["reference_args"]["topo_actions"]

kwargs_tau = []
for el in topo_actions:
     kwargs_tau.append(el["set_bus"]["substations_id"][0])

leap_net1 = LeapNet(name="tf_leapnet",

                    bench_config_path=BENCH_CONFIG_PATH,
                    bench_config_name="Benchmark1",
                    sim_config_path=SIM_CONFIG_PATH / "tf_leapnet.ini",
                    sim_config_name="DEFAULT",
                    log_path=LOG_PATH,

                    loss={"name": "mse"},
                    lr=1e-4,
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01),

                    sizes_enc=(),
                    sizes_main=(400, 400),
                    sizes_out=(),
                    topo_vect_to_tau="given_list",
                    is_topo_vect_input=False,
                    kwargs_tau=kwargs_tau,
                    layer="resnet",
                    attr_tau=("line_status","topo_vect"),
                    scale_main_layer=400,
                    # scale_input_enc_layer = 40,
                    #scale_input_dec_layer=200,
                    # topo_vect_as_input = True,
                    mult_by_zero_lines_pred=False,
                    topo_vect_as_input=True,
                    scaler=PowerGridScaler,

                    )

## add topo_vect (temporary ) in attr_x in benchmark config file


## add topo_vect (temporary ) in attr_x in benchmark config file
indices=[i for i in range(100)]
for key in benchmark1.train_dataset.data.keys():
    benchmark1.train_dataset.data[key]=benchmark1.train_dataset.data[key][indices]

benchmark1.train_dataset.size=len(indices)

leap_net1._leap_net_model.max_row_training_set=len(indices)

leap_net1.train(train_dataset= benchmark1.train_dataset,
            max_row_training_set=benchmark1.train_dataset.size,
            val_dataset=benchmark1.val_dataset,
            batch_size = 128,
            epochs=15)

leap_net1.summary()


EVAL_SAVE_PATH = get_path(EVALUATION_PATH, benchmark1)
tf_fc_metrics1 = benchmark1.evaluate_simulator(augmented_simulator=leap_net1,
                                              eval_batch_size=128,
                                              dataset="all",
                                              shuffle=False,
                                              save_predictions=True,
                                              save_path=EVAL_SAVE_PATH,
                                             )

print(tf_fc_metrics1["test"]["ML"])
print(tf_fc_metrics1["test_ood_topo"]["ML"])
print("ok")

#leap_net1.visualize_convergence()
