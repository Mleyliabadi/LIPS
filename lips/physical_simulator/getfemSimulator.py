#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lips.physical_simulator.physicalSimulator import PhysicalSimulator
from lips.physical_simulator.GetfemSimulator.GetfemSimulatorBridge import SimulatorGeneration

def lipsToGetfemBridge(physicalDomain,physicalProperties):
    simulator=SimulatorGeneration(physicalDomain=physicalDomain,physicalProperties=physicalProperties)
    return simulator


class GetfemSimulator(PhysicalSimulator):
    """
    This simulator uses the `Getfem` library to implement a physical simulator.
    """
    def __init__(self, physicalDomain=None,physicalProperties=None,simulatorInstance=None):
        if simulatorInstance is None:
            self._simulator = lipsToGetfemBridge(physicalDomain,physicalProperties)
            self._simulator.Preprocessing()
        else:
            self._simulator=type(simulatorInstance._simulator)(simulatorInstance._simulator)

    def build_model(self):
        self._simulator.BuildModel()

    def run_problem(self):
        self._simulator.RunProblem()

    def get_solution(self,field_name):
        return self._simulator.GetSolution(field_name)

    def get_variable_value(self,field_name):
        return self._simulator.GetVariableValue(field_name)

    def get_solverOrder_positions(self):
        return self._simulator.GetSolverOrderPosition()

    def get_state(self):
        """
        TODO
        """
        return self._simulator.internalInitParams

    def modify_state(self, actor):
        """
        TODO
        """
        self._simulator.SetPhyParams(actor)

    def __str__(self):
        return str(self._simulator)

import math


def check_static_benchmark():
    physicalDomain={
        "Mesher":"Getfem",
        "refNumByRegion":{"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3},
        "wheelDimensions":(8.0,15.0),#(13.5,18.0),
        "meshSize":1.5
    }


    physicalProperties={
        "ProblemType":"StaticMechanicalStandard",
        "materials":[["ALL", {"law":"LinearElasticity","young":5.98e6,"poisson":0.495} ]],#[["ALL", {"law":"IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2":1} ]],
        #"neumann":[["HOLE_BOUND",{"type" : "StandardNeumann", "fx":0.0, "fy":-1e2}] ],
        "neumann":[["HOLE_BOUND", {"type": "RimRigidityNeumann", "Force": 1.0e7}]],
        "incompressibility":True,
        "contact":[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.5}] ]
    }


    # physicalProperties={
    #     "ProblemType":"StaticMechanicalStandard",
    #     "materials":[["ALL", {"law":"SaintVenantKirchhoff","young":5.98e6,"poisson":0.495} ]],
    #     "incompressibility":True,
    #     "sources":[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0}] ],
    #     #"neumann":[["HOLE_BOUND",{"type" : "StandardNeumann", "fx":0.0, "fy":-5}] ],
    #     #"neumann":[["HOLE_BOUND",{"type" : "RimRigidityNeumann", "Force":5}] ],
    #     "dirichlet":[["HOLE_BOUND",{"type" : "scalar", "Disp_Amplitude":3, "Disp_Angle":-math.pi/2}] ],
    #     "contact":[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.0}] ]
    # }
    mySimulator=GetfemSimulator(physicalDomain=physicalDomain,physicalProperties=physicalProperties)
    mySimulator.build_model()
    mySimulator.run_problem()
    mySimulator._simulator.ExportSolutionInGmsh(filename="StaticBenchmark.pos")


def check_static():
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

    mySimulator = GetfemSimulator(physicalDomain=physicalDomain,physicalProperties=physicalProperties)
    mySimulator.build_model()
    mySimulator.run_problem()

def check_quasi_static_rolling():
    physicalDomain={
        "Mesher":"Gmsh",
        "subcategory":"DentedWheelGenerator",
        "refNumByRegion":{"HOLE_BOUND": 1,"CONTACT_BOUND": 2, "EXTERIOR_BOUND": 3},
        "wheel_Dimensions":(30.,36.,40.),
        "tread_Angle_deg":5.0,
        "teeth_Size":(10/3.0,10/6.0),
        "mesh_size":2,
        "meshFilename":"DentedWheel"
    }

    dt = 10e-4
    physicalProperties={
        "ProblemType":"QuasiStaticMechanicalRolling",
        "materials":[["ALL", {"law": "IncompressibleMooneyRivlin", "MooneyRivlinC1": 1, "MooneyRivlinC2": 1} ]],
        "incompressibility":True,
        "sources":[["ALL",{"type" : "Uniform","source_x":0.0,"source_y":0.0}] ],
        "rolling":["HOLE_BOUND",{"type" : "DIS_Rolling", "theta_Rolling":150., 'd': 1.}],
        "contact":[ ["CONTACT_BOUND",{"type" : "Plane","gap":0.0,"fricCoeff":0.6}] ],
        "transientParams":{"time": 5*dt, "timeStep": dt}
    }
    mySimulator=GetfemSimulator(physicalDomain=physicalDomain,physicalProperties=physicalProperties)
    mySimulator.build_model()
    print(mySimulator)
    mySimulator.run_problem()
    mySimulator._simulator.ExportSolutionInGmsh(filename="RollingSol.pos")

if __name__ == '__main__':
    #check_static()
    #check_quasi_static_rolling()
    check_static_benchmark()
