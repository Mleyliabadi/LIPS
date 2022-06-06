#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math


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

def CreateSimulatorBenchmark1():
    physicalDomain={
        "Mesher":"Getfem",
        "refNumByRegion":{"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3},
        "wheelDimensions":(8.0,15.0),
        "meshSize":1
    }

    physicalProperties={
        "ProblemType":"StaticMechanicalStandard",
        "materials":[["ALL", {"law":"LinearElasticity","young":5.98e6,"poisson":0.495} ]],
        "neumann":[["HOLE_BOUND", {"type": "RimRigidityNeumann", "Force": 1.0e7}]],
        "contact":[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.0}] ]
    }
    simulator=GetfemSimulator(physicalDomain=physicalDomain,physicalProperties=physicalProperties)
    return simulator


def GenerateDataSets(simulator,config,path_out=None):
    trainingInput={
              "Force":(1.0e4,1.0e7),
              }

    training_actor=LHSSampler(space_params=trainingInput)

    attr_names=(PFN.displacement,PFN.contactMultiplier)

    pneumaticWheelDataSetTrain=SamplerStaticWheelDataSet("train",attr_names=attr_names,config=config)
    pneumaticWheelDataSetTrain.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=2100,
                                    actor_seed=42,
                                    path_out=path_out
                                    )


    pneumaticWheelDataSetVal=SamplerStaticWheelDataSet("val",attr_names=attr_names,config=config)
    pneumaticWheelDataSetVal.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=600,
                                    actor_seed=42,
                                    path_out=path_out
                                    )

    pneumaticWheelDataSetTest=SamplerStaticWheelDataSet("test",attr_names=attr_names,config=config)
    pneumaticWheelDataSetTest.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=300,
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
    simulator=CreateSimulatorBenchmark1()
    attr_names=(PFN.displacement,PFN.contactMultiplier,"Force")
    attr_x= ("Force",)
    attr_y= ("disp",)
    pneumaticWheelDataSets=LoadDataSets(path_in="WeightSustainingWheelBenchmarkRegular",
                                        attr_names=attr_names,
                                        attr_x=attr_x,
                                        attr_y=attr_y)

    grid_support={"origin":(-16.0,0.0),"lenghts":(32.0,32.0),"sizes":(128,128)}
    dofnum_by_field={PFN.displacement:2}
    datasetOnGrid_by_type=GenerateInterpolatedDataSetsOnGrid(simulator=simulator,
                                                             datasets=pneumaticWheelDataSets,
                                                             grid_support=grid_support,
                                                             dofnum_by_field=dofnum_by_field,
                                                             path_out="WeightSustainingWheelBenchmarkInterpolated")


    LOG_PATH="WeightSustainingCNN.log"
    CONFIG_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/lips/config/confWheel.ini"
    DATA_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel"
    benchmarkCNN = WeightSustainingWheelBenchmark(benchmark_name="WeightSustainingWheelBenchmarkInterpolated",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=CONFIG_PATH
                               )

    # print(benchmarkCNN.config.get_options_dict())

    torch_sim = TorchSimulator(name="torch_unet",
                           model=TorchUnet,
                           scaler=StandardScalerPerChannel,
                           log_path=LOG_PATH,
                           seed=42,
                           architecture_type="Classical",
                          )


    # print(torch_sim.params)
    SAVE_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel/CNNResults"
    torch_sim_params={
            "metrics" : ("MAELoss",),
            "loss" : {"name": "MSELoss",
            "params": {"size_average": None,
                   "reduce": None,
                   "reduction": 'mean'}
                       },
            "optimizer" : {"name": "adam",
                           "params": {"lr": 2e-4}
                           },
            "train_batch_size" : 50,
            "epochs":200,
                }
    torch_sim.train(benchmarkCNN.train_dataset, benchmarkCNN.val_dataset, save_path=SAVE_PATH,**torch_sim_params)

    predictor_val = benchmarkCNN.evaluate_predictor(augmented_simulator=torch_sim,
                                                  eval_batch_size=128,
                                                  dataset="val",
                                                  shuffle=False,
                                                  save_path=None,
                                                  save_predictions=False
                                                 )
    

    predictor_val_on_mesh=ComputeMeshInterpolatedPrediction(name="val",dataset_grid=datasetOnGrid_by_type,prediction_on_grid=predictor_val,field_name=PFN.displacement)
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

    CONFIG_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/lips/config/confWheel.ini"
    benchmarkFFNN = WeightSustainingWheelBenchmark(benchmark_name="WeightSustainingWheelBenchmarkRegular",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=LOG_PATH,
                                config_path=CONFIG_PATH
                               )

    # #print(benchmarkFFNN.config.get_options_dict())
    torch_sim = TorchSimulator(name="torch_ffnn",
                           model=TorchFullyConnected,
                           scaler=StandardScaler,
                           log_path=LOG_PATH,
                           seed=42,
                           architecture_type="Classical",
                          )

    # #print(torch_sim.params)
    SAVE_PATH="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/getting_started/TestBenchmarkWheel/FFNNResults"
    torch_sim_params={
            "layers" : (300, 300, 300, 300),
            "metrics" : ("MAELoss",),
            "loss" : {"name": "MSELoss",
            "params": {"size_average": None,
                   "reduce": None,
                   "reduction": 'mean'}
                       },
            "optimizer" : {"name": "adam",
                           "params": {"lr": 2e-4}
                           },
            "train_batch_size" : 50,
            "epochs":200,
                }

    print("Training model")
    torch_sim.train(benchmarkFFNN.train_dataset, benchmarkFFNN.val_dataset, save_path=SAVE_PATH,**torch_sim_params)

    # torch_sim.summary()

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
    simulator=CreateSimulatorBenchmark1()
    wheelConfig=ConfigManager(path="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/lips/config/confWheel.ini",
                              section_name="WeightSustainingWheelBenchmarkRegular")
    pneumaticWheelDataSetTrain,pneumaticWheelDataSetVal,pneumaticWheelDataSetTest=GenerateDataSets(simulator=simulator,
                                                                                                   config=wheelConfig,
                                                                                                   path_out="WeightSustainingWheelBenchmarkRegular")


if __name__ == '__main__':
    #GenerateDataBaseBenchmark1()
    Benchmark1FFNN()
    #Benchmark1CNN()
