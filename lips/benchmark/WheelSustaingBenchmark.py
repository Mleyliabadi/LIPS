#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lips.config.configmanager import ConfigManager
from lips.augmented_simulators.torch_models.u_net import TorchUnet
from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected
from lips.augmented_simulators.torch_simulator import TorchSimulator
from lips.dataset.sampler import LHSSampler
from lips.dataset.scaler.standard_scaler_per_channel import StandardScalerPerChannel
from lips.dataset.scaler.standard_scaler import StandardScaler

from lips.physical_simulator.getfemSimulator import GetfemSimulator
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN

from lips.benchmark.wheelBenchmark import WeightSustainingWheelBenchmark
from lips.dataset.pneumaticWheelDataSet import SamplerStaticWheelDataSet,DataSetInterpolatorOnGrid,DataSetInterpolatorOnMesh

CONFIG_PATH_BENCHMARK="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/configurations/pneumatic/benchmarks/confWheel.ini"
CONFIG_PATH_AUGMENTED_SIMULATOR_FC="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/configurations/pneumatic/simulators/torch_fc.ini"
CONFIG_PATH_AUGMENTED_SIMULATOR_UNET="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/configurations/pneumatic/simulators/torch_unet.ini"

def GenerateDataSets(simulator,config,path_out=None):
    space_params=config.get_option("samplerParams")
    training_actor=LHSSampler(space_params=space_params)

    attr_names=(PFN.displacement,)

    pneumaticWheelDataSetTrain=SamplerStaticWheelDataSet("train",attr_names=attr_names,config=config)
    pneumaticWheelDataSetTrain.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=21,
                                    actor_seed=42,
                                    path_out=path_out
                                    )


    pneumaticWheelDataSetVal=SamplerStaticWheelDataSet("val",attr_names=attr_names,config=config)
    pneumaticWheelDataSetVal.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=6,
                                    actor_seed=42,
                                    path_out=path_out
                                    )

    pneumaticWheelDataSetTest=SamplerStaticWheelDataSet("test",attr_names=attr_names,config=config)
    pneumaticWheelDataSetTest.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=3,
                                    actor_seed=42,
                                    path_out=path_out
                                    )

    return pneumaticWheelDataSetTrain,pneumaticWheelDataSetVal,pneumaticWheelDataSetTest

def LoadDataSets(path_in,attr_names,attr_x,attr_y):
    pneumaticWheelDataSet=dict()
    for dataset_types in ["train","val","test"]:
        pneumaticWheelDataSet[dataset_types]=SamplerStaticWheelDataSet(dataset_types,attr_names=attr_names,attr_x=attr_x,attr_y=attr_y)
        pneumaticWheelDataSet[dataset_types].load(path=path_in)

    return pneumaticWheelDataSet


def GenerateInterpolatedDataSetsOnGrid(simulator,datasets,grid_support,dofnum_by_field,path_out):
    dataset_by_type=dict()
    for name,dataset in datasets.items():
        myTransformer=DataSetInterpolatorOnGrid(name=name,simulator=simulator,
                                                dataset=dataset,
                                                grid_support=grid_support)
        myTransformer.generate(dofnum_by_field=dofnum_by_field,path_out=path_out)
        dataset_by_type[dataset.name]=myTransformer
    return dataset_by_type

def GenerateInterpolatedDataSetsOnMesh(simulator,datasets,path_out):
    dataset_by_type=dict()
    for dataset in datasets:
        myTransformer=DataSetInterpolatorOnMesh(name=dataset.name,simulator=simulator,
                                                dataset=dataset,
                                                grid_support=grid_support)
        myTransformer.generate(dofnum_by_field=dofnum_by_field,path_out=path_out)
        dataset_by_type[dataset.name]=myTransformer
    return dataset_by_type

def ComputeMeshInterpolatedPrediction(name,dataset_grid,prediction_on_grid,field_name):
    prediction=prediction_on_grid[name]
    interpolated_field_name=field_name+"Interpolated"
    prediction[field_name] = prediction.pop(interpolated_field_name)
    simulator=dataset_grid[name].simulator
    
    interpolatedDatasetGrid=DataSetInterpolatorOnGrid(name=name,
                                                      simulator=simulator,
                                                      dataset=dataset_grid[name],
                                                      grid_support=dataset_grid[name].grid_support)

    interpolatedDatasetGrid.load_from_data(grid_support_points=dataset_grid[name].grid_support_points,
                                           interpolated_dataset=prediction,
                                           distributed_inputs_on_grid=dataset_grid[name].distributed_inputs_on_grid)

    interpolatedDatasOnMesh=DataSetInterpolatorOnMesh(name=name,
                                                      simulator=simulator,
                                                      dataset=interpolatedDatasetGrid)
    interpolatedDatasOnMesh.generate(field_names=[field_name])
    prediction_on_mesh={name: interpolatedDatasOnMesh.interpolated_dataset}
    return prediction_on_mesh

def Benchmark1CNN():
    wheelConfig=ConfigManager(path=CONFIG_PATH_BENCHMARK,
                              section_name="WeightSustainingWheelBenchmarkInterpolated")
    envParams=wheelConfig.get_option("env_params")
    physicalDomain=envParams.get("physicalDomain")
    physicalProperties=envParams.get("physicalProperties")
    simulator=GetfemSimulator(physicalDomain=physicalDomain,physicalProperties=physicalProperties)

    attr_x= wheelConfig.get_option("attr_x")
    attr_y= ("disp",)
    attr_names=attr_x+attr_y


    pneumaticWheelDataSets=LoadDataSets(path_in="WeightSustainingWheelBenchmarkRegular",
                                        attr_names=attr_names,
                                        attr_x=attr_x,
                                        attr_y=attr_y)

    interpolation_info=wheelConfig.get_option("interpolation_info")
    grid_support=interpolation_info.get("grid_support")
    dofnum_by_field=interpolation_info.get("dofnum_by_field")
    datasetOnGrid_by_type=GenerateInterpolatedDataSetsOnGrid(simulator=simulator,
                                                             datasets=pneumaticWheelDataSets,
                                                             grid_support=grid_support,
                                                             dofnum_by_field=dofnum_by_field,
                                                             path_out="WeightSustainingWheelBenchmarkInterpolated")

    LOG_PATH="WeightSustainingCNN.log"
    DATA_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel"
    benchmarkCNN = WeightSustainingWheelBenchmark(benchmark_name="WeightSustainingWheelBenchmarkInterpolated",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=CONFIG_PATH_BENCHMARK
                               )

    torch_sim = TorchSimulator(name="torch_unet",
                           model=TorchUnet,
                           scaler=StandardScalerPerChannel,
                           log_path=LOG_PATH,
                           seed=42,
                           architecture_type="Classical",
                          )


    SAVE_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel/CNNResults"
    torch_sim_config=ConfigManager(path=CONFIG_PATH_AUGMENTED_SIMULATOR_UNET,
                              section_name="DEFAULT")
    torch_sim_params=torch_sim_config.get_options_dict()

    torch_sim.train(train_dataset=benchmarkCNN.train_dataset,
                    val_dataset=benchmarkCNN.val_dataset,
                    save_path=SAVE_PATH,
                    **torch_sim_params)

    predictor_val = benchmarkCNN.evaluate_predictor(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="val",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )
    

    predictor_val_on_mesh=ComputeMeshInterpolatedPrediction(name="val",
                                                            dataset_grid=datasetOnGrid_by_type,
                                                            prediction_on_grid=predictor_val,
                                                            field_name=PFN.displacement)
    observation_val_on_mesh={"val":pneumaticWheelDataSets["val"].data}

    torch_sim_metrics_val = benchmarkCNN.evaluate_simulator_from_predictions(predictions=predictor_val_on_mesh,
                                                                             observations=observation_val_on_mesh,
                                                                             eval_batch_size=128,
                                                                             dataset="val",
                                                                             shuffle=False,
                                                                             save_path=None,
                                                                             save_predictions=False
                                                                            )

def Benchmark1FFNN():
    DATA_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel"
    LOG_PATH="WeightSustainingFFNN.log"

    benchmarkFFNN = WeightSustainingWheelBenchmark(benchmark_name="WeightSustainingWheelBenchmarkRegular",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=CONFIG_PATH_BENCHMARK
                               )

    torch_sim = TorchSimulator(name="torch_ffnn",
                           model=TorchFullyConnected,
                           scaler=StandardScaler,
                           log_path=LOG_PATH,
                           seed=42,
                           architecture_type="Classical",
                          )

    SAVE_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel/FFNNResults"
    torch_sim_config=ConfigManager(path=CONFIG_PATH_AUGMENTED_SIMULATOR_FC,
                              section_name="CONFIGWHEELSUSTAIN")
    torch_sim_params=torch_sim_config.get_options_dict()

    print("Training model")
    torch_sim.train(train_dataset=benchmarkFFNN.train_dataset,
                    val_dataset=benchmarkFFNN.val_dataset,
                    save_path=SAVE_PATH,
                    **torch_sim_params)

    print("Evaluation on val dataset")
    torch_sim_metrics_val = benchmarkFFNN.evaluate_simulator(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="val",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )

    print("Evaluation on test dataset")
    torch_sim_metrics_test = benchmarkFFNN.evaluate_simulator(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="test",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )

def GenerateDataBaseBenchmark1():
    wheelConfig=ConfigManager(path=CONFIG_PATH_BENCHMARK,
                              section_name="WeightSustainingWheelBenchmarkRegular")
    envParams=wheelConfig.get_option("env_params")
    physicalDomain=envParams.get("physicalDomain")
    physicalProperties=envParams.get("physicalProperties")
    simulator=GetfemSimulator(physicalDomain=physicalDomain,physicalProperties=physicalProperties)

    pneumaticWheelDataSetTrain,pneumaticWheelDataSetVal,pneumaticWheelDataSetTest=GenerateDataSets(simulator=simulator,
                                                                                                   config=wheelConfig,
                                                                                                   path_out="WeightSustainingWheelBenchmarkRegular")


if __name__ == '__main__':
    #GenerateDataBaseBenchmark1()
    #Benchmark1FFNN()
    Benchmark1CNN()
