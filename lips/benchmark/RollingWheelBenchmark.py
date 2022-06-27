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

def Benchmark2FFNNDisp():
    DATA_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel"
    LOG_PATH="RollingFFNNDisplacement.log"
    benchmarkQuasiStaticDataSet = DispRollingWheelBenchmark(benchmark_name="RollingWheelBenchmarkDisplacement",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=CONFIG_PATH_BENCHMARK,
                                input_required_for_post_process=True
                               )

    benchmarkQuasiStaticDataSet.split_train_test_valid(train_ratio=0.7,test_ratio=0.2,valid_ratio=0.1)

    wheelConfig=ConfigManager(path=CONFIG_PATH_BENCHMARK,
                              section_name="RollingWheelBenchmarkDisplacement")
    rollingProperties=wheelConfig.get_option("env_params").get("physicalProperties").get("rolling")[1]
    theta_Rolling = rollingProperties.get("theta_Rolling")
    verticalDisp = rollingProperties.get("d")
    wheel_origin_y = wheelConfig.get_option("env_params").get("physicalDomain").get("wheel_Dimensions")[-1]
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
    torch_sim.train(train_dataset=benchmarkQuasiStaticDataSet.train_dataset,
                    val_dataset=benchmarkQuasiStaticDataSet._test_dataset,
                    save_path=SAVE_PATH, epochs=10)

    torch_sim_metrics_val = benchmarkQuasiStaticDataSet.evaluate_simulator(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="test",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )

def Benchmark2FFNNMult():
    DATA_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel"
    LOG_PATH="RollingFFNNMultipliers.log"
    benchmarkQuasiStaticDataSet = DispRollingWheelBenchmark(benchmark_name="RollingWheelBenchmarkMultiplier",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=CONFIG_PATH_BENCHMARK,
                                input_required_for_post_process=False
                               )

    benchmarkQuasiStaticDataSet.split_train_test_valid(train_ratio=0.7,test_ratio=0.2,valid_ratio=0.1)


    torch_sim = TorchSimulator(name="torch_ffnn",
                           model=TorchFullyConnected,
                           scaler=StandardScaler,
                           log_path=LOG_PATH,
                           seed=42,
                           architecture_type="Classical",
                          )
    torch_sim_params={
            "layers" : (200,200,200),
            "metrics" : ("MAELoss",),
            "loss" : {"name": "MSELoss",
            "params": {"size_average": None,
                   "reduce": None,
                   "reduction": 'mean'}
                       },
            "optimizer" : {"name": "adam",
                           "params": {"lr": 5e-5}
                           },
            "train_batch_size" : 10,
            "epochs":200,
                }

    SAVE_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel/RollingMultFFNN"
    torch_sim.train(benchmarkQuasiStaticDataSet.train_dataset, benchmarkQuasiStaticDataSet._test_dataset, save_path=SAVE_PATH, **torch_sim_params)

    torch_sim_metrics_val = benchmarkQuasiStaticDataSet.evaluate_simulator(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="test",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )

def GenerateDataBaseBenchmark2():
    wheelConfig=ConfigManager(path=CONFIG_PATH_BENCHMARK,
                              section_name="RollingWheelBenchmarkDisplacement")
    envParams=wheelConfig.get_option("env_params")
    physicalDomain=envParams.get("physicalDomain")
    physicalProperties=envParams.get("physicalProperties")
    base_simulator=GetfemSimulator(physicalDomain=physicalDomain,physicalProperties=physicalProperties)
    
    attr_names=(PFN.displacement,PFN.contactMultiplier)
    quasiStaticWheelDataSet=QuasiStaticWheelDataSet("base",attr_names=attr_names,attr_x = ("timeSteps",),attr_y = attr_names)
    quasiStaticWheelDataSet.generate(simulator=base_simulator,
                                    path_out="RollingWheelBenchmarkBase")

    base_simulator._simulator.ExportSolutionInGmsh(filename="RollingSol.pos")


if __name__ == '__main__':
    #GenerateDataBaseBenchmark2()
    Benchmark2FFNNDisp()
    #Benchmark2FFNNMult()
