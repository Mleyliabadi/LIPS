#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import shutil
import csv

import copy
from typing import Union
from tqdm import tqdm  # TODO remove for final push

from lips.dataset.dataSet import DataSet
from lips.logger import CustomLogger
from lips.physical_simulator.GetfemSimulator.GetfemSimulatorBridge import GetfemInterpolationOnSupport,InterpolationOnCloudPoints
from lips.config.configmanager import ConfigManager


def Domain2DGridGenerator(origin,lenghts,sizes):
    origin_x,origin_y=origin
    lenght_x,lenght_y=lenghts
    nb_line,nb_column=sizes
    coordX,coordY=np.meshgrid(np.arange(origin_x,origin_x+lenght_x,lenght_x/nb_line),np.arange(origin_y,origin_y+lenght_y,lenght_y/nb_column))
    grid_support_points = np.vstack(list(zip(coordX.ravel(), coordY.ravel()))).transpose()
    return grid_support_points

class DataSetInterpolatorOnGrid():
    def __init__(self,simulator,dataset,grid_support):
        self.simulator=simulator
        self.dataset=dataset
        self.grid_support=grid_support
        self.grid_support_points=np.array([])
        self.interpolated_dataset = dict()
        self.distributed_inputs_on_grid = dict()

    def generate(self,dofnum_by_field,path_out=None):
        self.generate_interpolation_fields(dofnum_by_field)
        self.distribute_data_on_grid()

        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)
    
    def generate_interpolation_fields(self,dofnum_by_field):
        self._init_interpolation_fields(dofnum_by_field)

        grid_shape=self.grid_support["sizes"]
        self.grid_support_points=Domain2DGridGenerator(origin=self.grid_support["origin"],
                                               lenghts=self.grid_support["lenghts"],
                                               sizes=grid_shape)

        for dataIndex in range(len(self.dataset)):
            data_solver_obs=self.dataset.get_data(dataIndex)
            for field_name,dofnum in dofnum_by_field.items():
                single_field=np.zeros((dofnum,grid_shape[0],grid_shape[1]))
                original_field=data_solver_obs[field_name]
                interpolated_field=GetfemInterpolationOnSupport(self.simulator,original_field,self.grid_support_points)
                for dof_index in range(dofnum):
                    intermediate_field=interpolated_field[dof_index::dofnum]
                    single_field[dof_index]=intermediate_field.reshape((grid_shape[0],grid_shape[1]))
                self.interpolated_dataset[field_name][dataIndex]=single_field


    def distribute_data_on_grid(self):
        samples=self.dataset._inputs
        fieldNum=[len(samples[0].keys()) for sample in samples]
        if fieldNum.count(fieldNum[0]) != len(fieldNum):
            raise RuntimeError("Samples do not have the same input parameters")
        value_by_input_attrib = {attribName: np.array([sample[attribName] for sample in samples]) for attribName in samples[0]}

        nx,ny=self.grid_support["sizes"]
        for attrib_name,data in value_by_input_attrib.items():
            data = np.repeat(data[:, np.newaxis], nx*ny, axis=1)
            data = np.reshape(data,(data.shape[0],nx,ny))
            self.distributed_inputs_on_grid[attrib_name]=data

    def _init_interpolation_fields(self,dofnum_by_field):
        grid_shape=self.grid_support["sizes"]
        for field_name,dof_per_nodes in dofnum_by_field.items():
            self.interpolated_dataset[field_name]=np.zeros((len(self.dataset),dof_per_nodes,grid_shape[0],grid_shape[1]))

    def _save_internal_data(self, path_out):
        """save the self.data in a proper format"""
        full_path_out = os.path.join(os.path.abspath(path_out), self.dataset.name)

        if not os.path.exists(os.path.abspath(path_out)):
            os.mkdir(os.path.abspath(path_out))

        if os.path.exists(full_path_out):
            shutil.rmtree(full_path_out)

        os.mkdir(full_path_out)

        field_name="GridPoints"
        np.savez_compressed(f"{os.path.join(full_path_out, field_name)}.npz", data=self.grid_support_points)

        for field_name,data in self.interpolated_dataset.items():
            np.savez_compressed(f"{os.path.join(full_path_out, field_name)}Interpolated.npz", data=data)

        for attrib_name,data in self.distributed_inputs_on_grid.items():
            np.savez_compressed(f"{os.path.join(full_path_out, attrib_name)}.npz", data=data)

class DataSetInterpolatorOnMesh():
    def __init__(self,simulator,dataset):
        self.simulator=simulator
        self.dataset=dataset
        self.interpolated_dataset = dict()

    def generate(self,field_names,path_out=None):
        self.generate_projection_fields_on_mesh(field_names)
        self.accumulate_data_from_grid()

        if path_out is not None:
            #I should save the data
            self._save_internal_data(path_out)

    def generate_projection_fields_on_mesh(self,field_names):
        self._init_projection_fields(field_names)

        for dataIndex in range(self.dataset.interpolated_dataset[field_names[0]].shape[0]):
            for field_name in field_names:
                grid_field=self.dataset.interpolated_dataset[field_name][dataIndex]
                grid_field=np.transpose(grid_field.reshape(grid_field.shape[0],-1))
                fieldSupport=np.transpose(self.dataset.grid_support_points)

                #Clean true zeros
                exteriorPointsRows = np.where(grid_field[:,0] == 0.0) and np.where(grid_field[:,1] == 0.0)
                interpolatedInteriorSol = np.delete(grid_field, exteriorPointsRows, axis=0)
                interpolatedInteriorCoords=np.delete(fieldSupport, exteriorPointsRows, axis=0)
                
                interpolated_field=InterpolationOnCloudPoints(fieldSupport=interpolatedInteriorCoords,fieldValue=interpolatedInteriorSol,phyProblem=self.simulator)
                interleave_interpolated_field=np.empty((interpolated_field.shape[0]*interpolated_field.shape[1],))
                for dof in range(interpolated_field.shape[1]):
                    interleave_interpolated_field[dof::interpolated_field.shape[1]]=interpolated_field[:,dof]
                self.interpolated_dataset[field_name][dataIndex]=interleave_interpolated_field

    def accumulate_data_from_grid(self):
        grid_inputs=self.dataset.distributed_inputs_on_grid
        inputs_separated = [dict(zip(grid_inputs,t)) for t in zip(*grid_inputs.values())]
        accumulated_data_from_grid=[None]*len(inputs_separated)
        for obs_id,obs_input in enumerate(inputs_separated):
            obs_input={key:np.mean(value) for key,value in obs_input.items()}
            accumulated_data_from_grid[obs_id]=obs_input
        self.accumulated_data_from_grid={key: [single_data[key] for single_data in accumulated_data_from_grid] for key in accumulated_data_from_grid[0]}

    def _init_projection_fields(self,field_names):
        nb_samples=self.dataset.interpolated_dataset[field_names[0]].shape[0]
        for field_name in field_names:
            array_ = self.simulator.get_variable_value(field_name=field_name)
            self.interpolated_dataset[field_name] = np.zeros((nb_samples, array_.shape[0]), dtype=array_.dtype)


class WheelDataSet(DataSet):
    def __init__(self,
                 name="train",
                 attr_names=("disp",),
                 config: Union[ConfigManager, None]=None,
                 log_path: Union[str, None]=None,
                 **kwargs):
        super(WheelDataSet,self).__init__(name=name)
        self._attr_names = copy.deepcopy(attr_names)
        self.size = 0
        self._inputs = []

        # logger
        self.logger = CustomLogger(__class__.__name__, log_path).logger

        if config is not None:
            self.config = config
        else:
            self.config = ConfigManager()

        # number of dimension of x and y (number of columns)
        self._size_x = None
        self._size_y = None
        self._sizes_x = None  # dimension of each variable
        self._sizes_y = None  # dimension of each variable
        self._attr_x = kwargs["attr_x"] if "attr_x" in kwargs.keys() else self.config.get_option("attr_x")
        self._attr_y = kwargs["attr_y"] if "attr_y" in kwargs.keys() else self.config.get_option("attr_y")

    def _infer_sizes(self):
        data = copy.deepcopy(self.data)
        attrs_x=np.array([np.expand_dims(data[el], axis=1) for el in self._attr_x], dtype=int)
        self._sizes_x = np.array([attr_x.shape[1] for attr_x in attrs_x], dtype=int)
        self._size_x = np.sum(self._sizes_x)

        self._sizes_y = np.array([data[el].shape[1] for el in self._attr_y], dtype=int)
        self._size_y = np.sum(self._sizes_y)

    def _init_store_data(self,simulator,nb_samples):
        self.data=dict()
        for attr_nm in self._attr_names:
            array_ = simulator.get_variable_value(field_name=attr_nm)
            self.data[attr_nm] = np.zeros((nb_samples, array_.shape[0]), dtype=array_.dtype)

    def get_sizes(self):
        """Get the sizes of the dataset

        Returns
        -------
        tuple
            A tuple of size (nb_sample, size_x, size_y)

        """
        return self._size_x, self._size_y


    def get_data(self, index):
        """
        This function returns the data in the data that match the index `index`

        Parameters
        ----------
        index:
            A list of integer

        Returns
        -------

        """
        super().get_data(index)  # check that everything is legit

        # make sure the index are numpy array
        if isinstance(index, list):
            index = np.array(index, dtype=int)
        elif isinstance(index, int):
            index = np.array([index], dtype=int)

        # init the results
        res = {}
        nb_sample = index.size
        #for el in (*self._attr_names, *self._theta_attr_names):
        for el in self._attr_names:
            res[el] = np.zeros((nb_sample, self.data[el].shape[1]), dtype=self.data[el].dtype)

        #for el in (*self._attr_names, *self._theta_attr_names):
        for el in self._attr_names:
            res[el][:] = self.data[el][index, :]

        return res

    def _save_internal_data(self, path_out):
        """save the self.data in a proper format"""
        full_path_out = os.path.join(os.path.abspath(path_out), self.name)

        if not os.path.exists(os.path.abspath(path_out)):
            os.mkdir(os.path.abspath(path_out))
            # TODO logger
            #print(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")
            self.logger.info(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")

        if os.path.exists(full_path_out):
            # deleting previous saved data
            # TODO logger
            #print(f"Deleting previous run at {full_path_out}")
            self.logger.warning(f"Deleting previous run at {full_path_out}")
            shutil.rmtree(full_path_out)

        os.mkdir(full_path_out)
        # TODO logger
        #print(f"Creating the path {full_path_out} to store the dataset name {self.name}")
        self.logger.info(f"Creating the path {full_path_out} to store the dataset name {self.name}")

        for attr_nm in self._attr_names:
            np.savez_compressed(f"{os.path.join(full_path_out, attr_nm)}.npz", data=self.data[attr_nm])


    def reconstruct_output(self, data: "np.ndarray") -> dict:
        """It reconstruct the data from the extracted data

        Parameters
        ----------
        data : ``np.ndarray``
            the array that should be reconstruted

        Returns
        -------
        dict
            the reconstructed data with variable names in a dictionary
        """
        predictions = {}
        prev_ = 0
        for var_id, this_var_size in enumerate(self._sizes_y):
            attr_nm = self._attr_y[var_id]
            predictions[attr_nm] = data[:, prev_:(prev_ + this_var_size)]
            prev_ += this_var_size
        return predictions


class QuasiStaticWheelDataSet(WheelDataSet):
    """
    This specific DataSet uses Getfem framework to simulate data arising from a rolling wheel problem.
    """

    def __init__(self,
                 name="train",
                 attr_names=("disp",),
                 config: Union[ConfigManager, None]=None,
                 log_path: Union[str, None]=None
                 ):
        super(QuasiStaticWheelDataSet,self).__init__(name=name,attr_names=attr_names,config=config,log_path=log_path)

    def generate(self,
                 simulator: "GetfemSimulator",
                 path_out: Union[str, None]= None):
        try:
            import getfem
        except ImportError as exc_:
            raise RuntimeError("Impossible to `generate` a wheel dateset if you don't have "
                               "the getfem package installed") from exc_

        transientParams=getattr(simulator._simulator,"transientParams")
        nb_samples=int(transientParams["time"]//transientParams["timeStep"]) + 1
        self._init_store_data(simulator=simulator,nb_samples=nb_samples)
        simulator.build_model()
        solverState=simulator.run_problem()
            
        self._store_obs(obs=simulator)

        timesteps=getattr(simulator._simulator,"timeSteps")
        self.data["time"]=timesteps
        self._infer_sizes()

        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)
            attrib_name="timesteps"
            full_path_out = os.path.join(os.path.abspath(path_out), self.name)
            np.savez_compressed(f"{os.path.join(full_path_out, attrib_name)}.npz", data=timesteps)


    def _store_obs(self, obs):
        for attr_nm in self._attr_names:
            array_ = obs.get_solution(field_name=attr_nm)
            self.data[attr_nm] = array_


class SamplerStaticWheelDataSet(WheelDataSet):
    """
    This specific DataSet uses Getfem framework to simulate data arising from a rolling wheel problem.
    """

    def __init__(self,
                 name="train",
                 attr_names=("disp",),
                 config: Union[ConfigManager, None]=None,
                 log_path: Union[str, None]=None,
                 **kwargs
                 ):
        super(SamplerStaticWheelDataSet,self).__init__(name=name,attr_names=attr_names,config=config,log_path=log_path,**kwargs)

    def generate(self,
                 simulator: "GetfemSimulator",
                 actor,
                 nb_samples: int,
                 path_out: Union[str, None]= None,
                 simulator_seed: Union[None, int] = None,
                 actor_seed: Union[None, int] = None):

        try:
            import getfem
        except ImportError as exc_:
            raise RuntimeError("Impossible to `generate` a wheel dateset  if you don't have "
                               "the getfem package installed") from exc_
        if nb_samples <= 0:
            raise RuntimeError("Impossible to generate a negative number of data.")

        self._inputs=actor.generate_samples(nb_samples=nb_samples,sampler_seed=actor_seed)
        self._init_store_data(simulator=simulator,nb_samples=nb_samples)

        for current_size,sample in enumerate(tqdm(self._inputs, desc=self.name)):
            simulator=type(simulator)(simulatorInstance=simulator)
            simulator.modify_state(actor=sample)
            simulator.build_model()
            solverState=simulator.run_problem()
            
            self._store_obs(current_size=current_size,obs=simulator)
        self.size = nb_samples
        actor_attrib=actor.get_attributes_as_data()
        self.data={**self.data, **actor_attrib}
        self._infer_sizes()

        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)
            full_path_out = os.path.join(os.path.abspath(path_out), self.name)
            actor.save(path_out=full_path_out)


    def _store_obs(self, current_size, obs):
        for attr_nm in self._attr_names:
            array_ = obs.get_solution(field_name=attr_nm)
            self.data[attr_nm][current_size, :] = array_

    def load(self, path):
        if not os.path.exists(path):
            raise RuntimeError(f"{path} cannot be found on your computer")
        if not os.path.isdir(path):
            raise RuntimeError(f"{path} is not a valid directory")
        full_path = os.path.join(path, self.name)
        if not os.path.exists(full_path):
            raise RuntimeError(f"There is no data saved in {full_path}. Have you called `dataset.generate()` with "
                               f"a given `path_out` ?")
        #for attr_nm in (*self._attr_names, *self._theta_attr_names):
        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            if not os.path.exists(path_this_array):
                raise RuntimeError(f"Impossible to load data {attr_nm}. Have you called `dataset.generate()` with "
                                   f"a given `path_out` and such that `dataset` is built with the right `attr_names` ?")

        if self.data is not None:
            warnings.warn(f"Deleting previous run in attempting to load the new one located at {path}")
        self.data = {}
        self.size = None
        #for attr_nm in (*self._attr_names, *self._theta_attr_names):
        for attr_nm in self._attr_names:
            path_this_array = f"{os.path.join(full_path, attr_nm)}.npz"
            self.data[attr_nm] = np.load(path_this_array)["data"]
            self.size = self.data[attr_nm].shape[0]

        self._infer_sizes()

    def extract_data(self, concat: bool=True) -> tuple:
        """extract the x and y data from the dataset

        Parameters
        ----------
        concat : ``bool``
            If True, the data will be concatenated in a single array.
        Returns
        -------
        tuple
            extracted inputs and outputs
        """
        # init the sizes and everything
        data = copy.deepcopy(self.data)
        extract_x = [data[el].astype(np.float32) for el in self._attr_x]
        extract_y = [data[el].astype(np.float32) for el in self._attr_y]

        if concat:
            extract_x = [single_x.reshape((single_x.shape[0],1)) for single_x in extract_x]
            extract_x = np.concatenate(extract_x, axis=1)
            extract_y = np.concatenate(extract_y, axis=1)
        return extract_x, extract_y


#Check integrities

import math
from lips.physical_simulator.getfemSimulator import GetfemSimulator
import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
from lips.dataset.sampler import LHSSampler

def check_static_samples_generation():
    physicalDomain={
        "Mesher":"Getfem",
        "refNumByRegion":{"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3},
        "wheelDimensions":(8.,15.),
        "meshSize":1
    }

    physicalProperties={
        "ProblemType":"StaticMechanicalStandard",
        "materials":[["ALL", {"law":"LinearElasticity","young":21E6,"poisson":0.3} ]],
        "sources":[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0}] ],
        "dirichlet":[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":6, "Disp_Angle":-math.pi/2}] ],
        "contact":[ ["CONTACT_BOUND",{"type" : "Plane","gap":2.0,"fricCoeff":0.9}] ]
    }
    training_simulator=GetfemSimulator(physicalDomain=physicalDomain,physicalProperties=physicalProperties)

    trainingInput={
              "young":(75.0,85.0),
              "poisson":(0.38,0.44),
              "fricCoeff":(0.5,0.8)
              }

    training_actor=LHSSampler(space_params=trainingInput)

    attr_names=(PFN.displacement,PFN.contactMultiplier)

    wheelConfig=ConfigManager(path="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/lips/config/confWheel.ini",
                              section_name="BenchmarkWheelFFNN")

    staticWheelDataSet=SamplerStaticWheelDataSet("train",attr_names=attr_names,config=wheelConfig)
    staticWheelDataSet.generate(simulator=training_simulator,
                                    actor=training_actor,
                                    path_out="WheelDir",
                                    nb_samples=5,
                                    actor_seed=42
                                    )
    # print(staticWheelDataSet.get_data(index=0))
    # print(staticWheelDataSet.data)

    #Interpolation on grid
    grid_support={"origin":(-16.0,0.0),"lenghts":(32.0,32.0),"sizes":(16,16)}
    myTransformer=DataSetInterpolatorOnGrid(simulator=training_simulator,
                                            dataset=staticWheelDataSet,
                                            grid_support=grid_support)
    dofnum_by_field={PFN.displacement:2}
    myTransformer.generate(dofnum_by_field=dofnum_by_field,path_out="wheel_interpolated")

def check_quasi_static_generation():
    physicalDomain={
        "Mesher":"Getfem",
        "refNumByRegion":{"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3},
        "wheelDimensions":(8.,15.),
        "meshSize":1
    }

    dt = 10e-4
    physicalProperties={
        "ProblemType":"QuasiStaticMechanicalRolling",
        "materials":[["ALL", {"law": "IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2": 1} ]],
        "sources":[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ],
        "rolling":["HOLE_BOUND",{"type" : "DIS_Rolling", "theta_Rolling":150., 'd': 1.}],
        "contact":[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.6}] ],
        "transientParams":{"time": 3*dt, "timeStep": dt}
    }

    training_simulator=GetfemSimulator(physicalDomain=physicalDomain,physicalProperties=physicalProperties)
    attr_names=(PFN.displacement,PFN.contactMultiplier)
    wheelConfig=ConfigManager(path="/home/ddanan/HSAProject/LIPSPlatform/LIPS_Github/LIPS/lips/config/confWheel.ini",
                              section_name="RollingWheel")

    quasiStaticWheelDataSet=QuasiStaticWheelDataSet("train",attr_names=attr_names,config=wheelConfig)
    quasiStaticWheelDataSet.generate(simulator=training_simulator,
                                    path_out="WheelRolDir",
                                    )

def check_interpolation_back_and_forth():
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

    trainingInput={
              "Force":(1.0e4,1.0e7),
              }

    training_actor=LHSSampler(space_params=trainingInput)

    attr_names=(PFN.displacement,PFN.contactMultiplier)
    pneumaticWheelDataSetTrain=SamplerStaticWheelDataSet("train",attr_names=attr_names,attr_x= ("Force",),attr_y= ("disp",))
    path_out="WeightRegular"
    pneumaticWheelDataSetTrain.generate(simulator=simulator,
                                    actor=training_actor,
                                    nb_samples=3,
                                    actor_seed=42,
                                    path_out=path_out
                                    )

    charac_sizes=[32,48,64,96,128]
    abs_error=[None]*len(charac_sizes)
    for charac_id,charac_size in enumerate(charac_sizes):
        print("Interpolation for charac_size=",charac_size)
        grid_support={"origin":(-16.0,0.0),"lenghts":(32.0,32.0),"sizes":(charac_size,charac_size)}
        interpolatedDatasetGrid=DataSetInterpolatorOnGrid(simulator=simulator,
                                                    dataset=pneumaticWheelDataSetTrain,
                                                    grid_support=grid_support)
        dofnum_by_field={PFN.displacement:2}
        path_out="WeightInterpolated"
        interpolatedDatasetGrid.generate(dofnum_by_field=dofnum_by_field,path_out=path_out)

        interpolatedDatasetMesh=DataSetInterpolatorOnMesh(simulator=simulator,
                                                    dataset=interpolatedDatasetGrid)

        interpolatedDatasetMesh.generate(field_names=[PFN.displacement])

        original_data=pneumaticWheelDataSetTrain.data["disp"]
        reinterpolated_data=interpolatedDatasetMesh.interpolated_dataset["disp"]
        abs_error[charac_id]=np.linalg.norm(original_data-reinterpolated_data)

    #Check error is decreasing
    print(abs_error)
    np.testing.assert_equal(abs_error[::-1],np.sort(abs_error))


if __name__ == '__main__':
    check_interpolation_back_and_forth()
