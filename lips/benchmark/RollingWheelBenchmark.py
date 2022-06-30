#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lips.config.configmanager import ConfigManager
from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected
from lips.augmented_simulators.torch_simulator import TorchSimulator
from lips.dataset.scaler.standard_scaler import StandardScaler
from lips.dataset.scaler.rolling_scaler import RollingWheelScaler

from lips.benchmark.wheelBenchmark import WeightSustainingWheelBenchmark,DispRollingWheelBenchmark
from lips.dataset.pneumaticWheelDataSet import QuasiStaticWheelDataSet
from lips.physical_simulator.getfemSimulator import GetfemSimulator
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN

CONFIG_PATH_BENCHMARK="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/configurations/pneumatic/benchmarks/confWheel.ini"
CONFIG_PATH_AUGMENTED_SIMULATOR_FC="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/configurations/pneumatic/simulators/torch_fc.ini"

def Benchmark2FFNNDisp(data_path):
    LOG_PATH="RollingFFNNDisplacement.log"
    benchmark_quasistatic_dataSet = DispRollingWheelBenchmark(benchmark_name="RollingWheelBenchmarkDisplacement",
                                benchmark_path=data_path,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=CONFIG_PATH_BENCHMARK,
                                input_required_for_post_process=True
                               )

    wheel_config=ConfigManager(path=CONFIG_PATH_BENCHMARK,
                              section_name="RollingWheelBenchmarkDisplacement")

    benchmark_quasistatic_dataSet.split_train_test_valid()

    rolling_properties=wheel_config.get_option("env_params").get("physical_properties").get("rolling")[1]
    theta_Rolling = rolling_properties.get("theta_Rolling")
    verticalDisp = rolling_properties.get("d")
    wheel_origin_y = wheel_config.get_option("env_params").get("physical_domain").get("wheel_Dimensions")[-1]
    wheel_speed = theta_Rolling * (wheel_origin_y - verticalDisp/3)


    torch_sim = TorchSimulator(name="torch_ffnn",
                           model=TorchFullyConnected,
                           scaler=RollingWheelScaler,
                           log_path=LOG_PATH,
                           seed=42,
                           architecture_type="Classical",
                           scalerParams={"wheel_velocity":wheel_speed}
                          )

    SAVE_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel/RollingDispFFNN"
    torch_sim_config=ConfigManager(path=CONFIG_PATH_AUGMENTED_SIMULATOR_FC,
                              section_name="CONFIGROLLINGDISP")
    torch_sim_params=torch_sim_config.get_options_dict()

    torch_sim.train(train_dataset=benchmark_quasistatic_dataSet.train_dataset,
                    val_dataset=benchmark_quasistatic_dataSet._test_dataset,
                    save_path=SAVE_PATH, **torch_sim_params)

    torch_sim_metrics_val = benchmark_quasistatic_dataSet.evaluate_simulator(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="test",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )

def Benchmark2FFNNMult(data_path):
    DATA_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel"
    LOG_PATH="RollingFFNNMultipliers.log"
    benchmark_quasistatic_dataSet = DispRollingWheelBenchmark(benchmark_name="RollingWheelBenchmarkMultiplier",
                                benchmark_path=data_path,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=CONFIG_PATH_BENCHMARK,
                                input_required_for_post_process=False
                               )

    wheel_config=ConfigManager(path=CONFIG_PATH_BENCHMARK,
                              section_name="RollingWheelBenchmarkMultiplier")
    benchmark_quasistatic_dataSet.split_train_test_valid()


    torch_sim = TorchSimulator(name="torch_ffnn",
                           model=TorchFullyConnected,
                           scaler=StandardScaler,
                           log_path=LOG_PATH,
                           seed=42,
                           architecture_type="Classical",
                          )

    torch_sim_config=ConfigManager(path=CONFIG_PATH_AUGMENTED_SIMULATOR_FC,
                              section_name="CONFIGROLLINGMULTIPLIER")
    torch_sim_params=torch_sim_config.get_options_dict()

    SAVE_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel/RollingMultFFNN"
    torch_sim.train(benchmark_quasistatic_dataSet.train_dataset, benchmark_quasistatic_dataSet._test_dataset, save_path=SAVE_PATH, **torch_sim_params)

    torch_sim_metrics_val = benchmark_quasistatic_dataSet.evaluate_simulator(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="test",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )

def GenerateDataBaseBenchmark2():
    wheel_config=ConfigManager(path=CONFIG_PATH_BENCHMARK,
                              section_name="RollingWheelBenchmarkDisplacement")
    env_params=wheel_config.get_option("env_params")
    physical_domain=env_params.get("physical_domain")
    physical_properties=env_params.get("physical_properties")
    base_simulator=GetfemSimulator(physical_domain=physical_domain,physical_properties=physical_properties)
    
    attr_names=(PFN.displacement,PFN.contactMultiplier)
    attr_x = wheel_config.get_option("attr_x")
    quasiStaticWheelDataSet=QuasiStaticWheelDataSet("base",attr_names=attr_names,attr_x = attr_x,attr_y = attr_names)
    quasiStaticWheelDataSet.generate(simulator=base_simulator,
                                    path_out="RollingWheelBenchmarkBase")

    base_simulator._simulator.ExportSolutionInGmsh(filename="RollingSol.pos")


if __name__ == '__main__':
    # GenerateDataBaseBenchmark2()
    DATA_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/lips/benchmark/"

    #Benchmark2FFNNDisp(data_path=DATA_PATH)
    Benchmark2FFNNMult(data_path=DATA_PATH)
