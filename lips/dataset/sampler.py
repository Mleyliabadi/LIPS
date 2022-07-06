"""
Usage:
    Introduce the sampling methods used to generate a space of parameters
Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""
 
import abc
import numpy as np
import pyDOE2 as doe
from typing import Union
import os

class Sampler(metaclass=abc.ABCMeta):
    def __init__(self,space_params):
        self.space_params=space_params
        self.sampling_output=[]
        self.sampling_name=""
    
    def generate_samples(self,nb_samples,sampler_seed=None):
        self.sampling_output=self._define_sampling_method(nb_samples=nb_samples,sampler_seed=sampler_seed)
        return self.sampling_output

    @abc.abstractmethod
    def _define_sampling_method(self,nb_samples,sampler_seed=None):
        pass

    def get_attributes_as_data(self,samples=None):
        if samples is None:
            samples=self.sampling_output

        fieldNum=[len(samples[0].keys()) for sample in samples]
        if fieldNum.count(fieldNum[0]) != len(fieldNum):
            raise RuntimeError("Samples do not have the same input parameters")

        value_by_input_attrib = {attribName: np.array([sample[attribName] for sample in samples]) for attribName in samples[0]}
        return value_by_input_attrib

    def save(self,path_out,samples=None):
        value_by_input_attrib = self.get_attributes_as_data(samples=samples)
        for attrib_name,data in value_by_input_attrib.items():
            np.savez_compressed(f"{os.path.join(path_out, attrib_name)}.npz", data=data)

    def __str__(self): 
        sInfo="Type of sampling: "+self.sampling_name+"\n"
        sInfo+="Parameters\n"
        for paramName,paramVal in self.space_params.items():
            sInfo+="\t"+str(paramName)+": "+str(paramVal)+"\n"
        return sInfo 

    def __len__(self):
        return len(self.sampling_output)


class LHSSampler(Sampler):
    def __init__(self, space_params):
        super(LHSSampler,self).__init__(space_params=space_params) 
        self.sampling_name="LHSSampler"

    def _define_sampling_method(self,nb_samples,sampler_seed=None):
        space_params=self.space_params
        nfactor = len(space_params)
        self.vals =doe.lhs(nfactor, samples=nb_samples, random_state=sampler_seed, criterion="maximin")
        
        vals=np.transpose(self.vals)
        paramsVectByName = {}
        for i,paramName in enumerate(space_params.keys()):
            minVal,maxVal=space_params[paramName]
            paramsVectByName[paramName] = minVal + vals[i]*(maxVal - minVal)
        return list(map(dict, zip(*[[(k, v) for v in value] for k, value in paramsVectByName.items()])))

if __name__ =="__main__":
    params={"params1":(21E6,22E6),"params2":(0.2,0.3)}
    sample_params=LHSSampler(space_params=params)
    test_params1=sample_params.generate_samples(nb_samples=2,sampler_seed=42)
    test_params2=sample_params.generate_samples(nb_samples=2,sampler_seed=42)
    assert test_params1==test_params2

    params={"params1":(21E6,22E6),"params2":(0.2,0.3)}
    sample_params=LHSSampler(space_params=params)
    nb_samples=5
    test_params=sample_params.generate_samples(nb_samples=nb_samples)
    assert len(test_params)==nb_samples
    print(sample_params)

