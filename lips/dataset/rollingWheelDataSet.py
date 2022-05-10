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
from lips.physical_simulator.GetfemSimulator.GetfemSimulatorBridge import GetfemInterpolationOnSupport

def Domain2DGridGenerator(origin,lenghts,sizes):
    origin_x,origin_y=origin
    lenght_x,lenght_y=lenghts
    nb_line,nb_column=sizes
    coordX,coordY=np.meshgrid(np.arange(origin_x,origin_x+lenght_x,lenght_x/nb_line),np.arange(origin_y,origin_y+lenght_y,lenght_y/nb_column))
    gridSupport = np.vstack(list(zip(coordX.ravel(), coordY.ravel()))).transpose()
    return gridSupport

class DataSetInterpolatorOnGrid():
    """
    This specific DataSet uses Getfem framework to simulate data arising from a rolling wheel problem.
    """

    def __init__(self,simulator,dataset,gridSupport):
        self.simulator=simulator
        self.dataset=dataset
        self.gridSupport=gridSupport
        self.interpolated_dataset=dict()

    def generate_interpolation_fields(self,dofnum_by_field,path_out=None):
        self._init_interpolation_fields(dofnum_by_field)

        for dataIndex in range(len(self.dataset)):
            data_solver_obs=self.dataset.get_data(dataIndex)
            for field_name in dofnum_by_field.keys():
                field=data_solver_obs[field_name]
                self.interpolated_dataset[field_name][dataIndex]=GetfemInterpolationOnSupport(self.simulator,field,self.gridSupport)

        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)

    def _init_interpolation_fields(self,dofnum_by_field):
        for field_name,dof_per_nodes in dofnum_by_field.items():
            self.interpolated_dataset[field_name]=np.zeros((len(self.dataset),dof_per_nodes*self.gridSupport.shape[1]))

    def _save_internal_data(self, path_out):
        """save the self.data in a proper format"""
        full_path_out = os.path.join(os.path.abspath(path_out), self.dataset.name)

        if not os.path.exists(os.path.abspath(path_out)):
            os.mkdir(os.path.abspath(path_out))

        if os.path.exists(full_path_out):
            shutil.rmtree(full_path_out)

        os.mkdir(full_path_out)

        for field_name in self.interpolated_dataset.keys():
            np.savez_compressed(f"{os.path.join(full_path_out, field_name)}Interpolated.npz", data=self.interpolated_dataset[field_name])

        samples=self.dataset._inputs
        fieldNum=[len(samples[0].keys()) for sample in samples]
        if fieldNum.count(fieldNum[0]) != len(fieldNum):
            raise RuntimeError("Samples do not have the same input parameters")

        full_path_samples_out=os.path.join(full_path_out, "inputs.data")

        with open(full_path_samples_out, mode='w') as csv_file:
            fieldnames = list(samples[0].keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for paramsSet in samples:
                writer.writerow(paramsSet)

class RollingWheelDataSet(DataSet):
    """
    This specific DataSet uses Getfem framework to simulate data arising from a rolling wheel problem.
    """

    def __init__(self,
                 name="train",
                 attr_names=("disp",),
                 log_path: Union[str, None]=None
                 ):
        DataSet.__init__(self, name=name)
        self._attr_names = copy.deepcopy(attr_names)
        self.size = 0
        self._inputs = []

        # logger
        self.logger = CustomLogger(__class__.__name__, log_path).logger

    def generate(self,
                 simulator: "GetfemSimulator",
                 actor,
                 path_out,
                 nb_samples,
                 simulator_seed: Union[None, int] = None,
                 actor_seed: Union[None, int] = None):
        """
        For this dataset, we use a GetfemSimulator and a Sampler to generate data from a rolling wheel.

        Parameters
        ----------
        simulator:
           In this case, this should be a getfem-based instance

        actor:
           In this case, it is the sampler used for the input parameters space discretization

        path_out:
            The path where the data will be saved

        nb_samples:
            Number of rows (examples) in the final dataset

        simulator_seed:
            Seed used to set the simulator for reproducible experiments

        actor_seed:
            Seed used to set the actor for reproducible experiments

        Returns
        -------

        """
        try:
            import getfem
        except ImportError as exc_:
            raise RuntimeError("Impossible to `generate` rolling wheel datet if you don't have "
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

        if path_out is not None:
            # I should save the data
            self._save_internal_data(path_out)
            full_path_out = os.path.join(os.path.abspath(path_out), self.name)
            full_path_samples_out=os.path.join(full_path_out, "inputs.data")
            actor.save(path_out=full_path_samples_out)


    def _init_store_data(self,simulator,nb_samples):
        self.data=dict()
        for attr_nm in self._attr_names:
            array_ = simulator.get_variable_value(field_name=attr_nm)
            self.data[attr_nm] = np.zeros((nb_samples, array_.shape[0]), dtype=array_.dtype)

    def _store_obs(self, current_size, obs):
        for attr_nm in self._attr_names:
            array_ = obs.get_solution(field_name=attr_nm)
            self.data[attr_nm][current_size, :] = array_

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

if __name__ == '__main__':
    import math
    from lips.physical_simulator.getfemSimulator import GetfemSimulator
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

    from lips.dataset.sampler import LHSSampler
    trainingInput={
              "young":(75.0,85.0),
              "poisson":(0.38,0.44),
              "fricCoeff":(0.5,0.8)
              }

    training_actor=LHSSampler(space_params=trainingInput)

    import lips.physical_simulator.GetfemSimulator.PhysicalFieldNames as PFN
    attr_names=(PFN.displacement,PFN.contactMultiplier)

    rollingWheelDataSet=RollingWheelDataSet("train",attr_names=attr_names)
    rollingWheelDataSet.generate(simulator=training_simulator,
                                    actor=training_actor,
                                    path_out="WheelDir",
                                    nb_samples=3,
                                    actor_seed=42
                                    )
    # print(rollingWheelDataSet.get_data(index=0))
    # print(rollingWheelDataSet.data)

    #Interpolation on grid
    gridSupport=Domain2DGridGenerator(origin=(-16.0,0.0),lenghts=(32.0,32.0),sizes=(64,64))
    myTransformer=DataSetInterpolatorOnGrid(simulator=training_simulator,
                                            dataset=rollingWheelDataSet,
                                            gridSupport=gridSupport)
    dofnum_by_field={PFN.displacement:2}
    myTransformer.generate_interpolation_fields(dofnum_by_field=dofnum_by_field,path_out="wheel_interpolated")