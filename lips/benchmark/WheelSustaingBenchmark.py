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
    sampler=config.get_option("sampler")
    sampler_input_params=sampler.get("sampler_input_params")
    sampler_seeds=sampler.get("seeds")
    sampler_nb_samples=sampler.get("nb_samples")

    training_actor=LHSSampler(space_params=sampler_input_params)

    attr_names=(PFN.displacement,)

    pneumatic_wheel_dataset_train=SamplerStaticWheelDataSet("train",attr_names=attr_names,config=config)
    pneumatic_wheel_dataset_train.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=sampler_nb_samples.get("train"),
                                    actor_seed=sampler_seeds.get("train"),
                                    path_out=path_out
                                    )


    pneumatic_wheel_dataset_val=SamplerStaticWheelDataSet("val",attr_names=attr_names,config=config)
    pneumatic_wheel_dataset_val.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=sampler_nb_samples.get("val"),
                                    actor_seed=sampler_seeds.get("val"),
                                    path_out=path_out
                                    )

    pneumatic_wheel_dataset_test=SamplerStaticWheelDataSet("test",attr_names=attr_names,config=config)
    pneumatic_wheel_dataset_test.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=sampler_nb_samples.get("test"),
                                    actor_seed=sampler_seeds.get("test"),
                                    path_out=path_out
                                    )

    return pneumatic_wheel_dataset_train,pneumatic_wheel_dataset_val,pneumatic_wheel_dataset_test

def LoadDataSets(path_in,attr_names,attr_x,attr_y):
    pneumatic_wheel_dataset=dict()
    for dataset_types in ["train","val","test"]:
        pneumatic_wheel_dataset[dataset_types]=SamplerStaticWheelDataSet(dataset_types,attr_names=attr_names,attr_x=attr_x,attr_y=attr_y)
        pneumatic_wheel_dataset[dataset_types].load(path=path_in)

    return pneumatic_wheel_dataset


def GenerateInterpolatedDataSetsOnGrid(simulator,datasets,grid_support,dofnum_by_field,path_out):
    dataset_by_type=dict()
    for name,dataset in datasets.items():
        myTransformer=DataSetInterpolatorOnGrid(name=name,
                                                simulator=simulator,
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
    
    interpolated_dataset_grid=DataSetInterpolatorOnGrid(name=name,
                                                      simulator=simulator,
                                                      dataset=dataset_grid[name],
                                                      grid_support=dataset_grid[name].grid_support)

    interpolated_dataset_grid.load_from_data(grid_support_points=dataset_grid[name].grid_support_points,
                                           interpolated_dataset=prediction,
                                           distributed_inputs_on_grid=dataset_grid[name].distributed_inputs_on_grid)

    interpolated_datas_on_mesh=DataSetInterpolatorOnMesh(name=name,
                                                      simulator=simulator,
                                                      dataset=interpolated_dataset_grid)
    interpolated_datas_on_mesh.generate(field_names=[field_name])
    prediction_on_mesh={name: interpolated_datas_on_mesh.interpolated_dataset}
    return prediction_on_mesh

def Benchmark1CNN(data_path):
    wheel_config=ConfigManager(path=CONFIG_PATH_BENCHMARK,
                              section_name="WeightSustainingWheelBenchmarkInterpolated")
    env_params=wheel_config.get_option("env_params")
    physical_domain=env_params.get("physical_domain")
    physical_properties=env_params.get("physical_properties")
    simulator=GetfemSimulator(physical_domain=physical_domain,physical_properties=physical_properties)

    attr_x= wheel_config.get_option("attr_x")
    attr_y= ("disp",)
    attr_names=attr_x+attr_y


    pneumatic_wheel_datasets=LoadDataSets(path_in="WeightSustainingWheelBenchmarkRegular",
                                        attr_names=attr_names,
                                        attr_x=attr_x,
                                        attr_y=attr_y)

    interpolation_info=wheel_config.get_option("interpolation_info")
    grid_support=interpolation_info.get("grid_support")
    dofnum_by_field=interpolation_info.get("dofnum_by_field")
    dataset_on_grid_by_type=GenerateInterpolatedDataSetsOnGrid(simulator=simulator,
                                                             datasets=pneumatic_wheel_datasets,
                                                             grid_support=grid_support,
                                                             dofnum_by_field=dofnum_by_field,
                                                             path_out="WeightSustainingWheelBenchmarkInterpolated")

    LOG_PATH="WeightSustainingCNN.log"
    benchmark_cnn = WeightSustainingWheelBenchmark(benchmark_name="WeightSustainingWheelBenchmarkInterpolated",
                                benchmark_path=data_path,
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

    torch_sim.train(train_dataset=benchmark_cnn.train_dataset,
                    val_dataset=benchmark_cnn.val_dataset,
                    save_path=SAVE_PATH,
                    **torch_sim_params)

    predictor_val = benchmark_cnn.evaluate_predictor(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="val",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )
    

    predictor_val_on_mesh=ComputeMeshInterpolatedPrediction(name="val",
                                                            dataset_grid=dataset_on_grid_by_type,
                                                            prediction_on_grid=predictor_val,
                                                            field_name=PFN.displacement)
    observation_val_on_mesh={"val":pneumatic_wheel_datasets["val"].data}

    torch_sim_metrics_val = benchmark_cnn.evaluate_simulator_from_predictions(predictions=predictor_val_on_mesh,
                                                                             observations=observation_val_on_mesh,
                                                                             eval_batch_size=128,
                                                                             dataset="val",
                                                                             shuffle=False,
                                                                             save_path=None,
                                                                             save_predictions=False
                                                                            )
    print(torch_sim_metrics_val)

def Benchmark1FFNN(data_path):
    LOG_PATH="WeightSustainingFFNN.log"

    benchmark_ffnn = WeightSustainingWheelBenchmark(benchmark_name="WeightSustainingWheelBenchmarkRegular",
                                benchmark_path=data_path,
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
    torch_sim.train(train_dataset=benchmark_ffnn.train_dataset,
                    val_dataset=benchmark_ffnn.val_dataset,
                    save_path=SAVE_PATH,
                    **torch_sim_params)

    print("Evaluation on val dataset")
    torch_sim_metrics_val = benchmark_ffnn.evaluate_simulator(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="val",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )

    print("Evaluation on test dataset")
    torch_sim_metrics_test = benchmark_ffnn.evaluate_simulator(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="test",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )
    print(torch_sim_metrics_val)

def GenerateDataBaseBenchmark1():
    wheel_config=ConfigManager(path=CONFIG_PATH_BENCHMARK,
                              section_name="WeightSustainingWheelBenchmarkRegular")

    print(wheel_config)

    env_params=wheel_config.get_option("env_params")
    physical_domain=env_params.get("physical_domain")
    physical_properties=env_params.get("physical_properties")
    simulator=GetfemSimulator(physical_domain=physical_domain,physical_properties=physical_properties)

    pneumatic_wheel_dataset_train,pneumatic_wheel_dataset_val,pneumatic_wheel_dataset_test=GenerateDataSets(simulator=simulator,
                                                                                                   config=wheel_config,
                                                                                                   path_out="WeightSustainingWheelBenchmarkRegular")


if __name__ == '__main__':
    #GenerateDataBaseBenchmark1()
    DATA_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/lips/benchmark"
    #Benchmark1FFNN(data_path=DATA_PATH)
    Benchmark1CNN(data_path=DATA_PATH)
