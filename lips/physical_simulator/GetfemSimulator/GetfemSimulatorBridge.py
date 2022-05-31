#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lips.physical_simulator.GetfemSimulator.GetfemHSA as PhySolver
import lips.physical_simulator.GetfemSimulator.MeshGenerationTools as ExternalMesher
from lips.physical_simulator.GetfemSimulator.GetfemWheelProblem import GetfemMecaProblem,GetfemRollingWheelProblem
from lips.physical_simulator.GetfemSimulator.GetfemWheelProblemQuasiStatic import QuasiStaticRollingProblem,QuasiStaticMecanicalProblem
from lips.physical_simulator.GetfemSimulator.GetfemInterpolationTools import FEMInterpolationOnSupport
import lips.physical_simulator.GetfemSimulator.PhysicalCriteria as PhyCriteria 

def GetfemInterpolationOnSupport(simulator,field,gridSupport):
    physical_problem=simulator._simulator
    return FEMInterpolationOnSupport(phyProblem=physical_problem,originalField=field,targetSupport=gridSupport)

def MeshGeneration(physicalDomain):
    if physicalDomain["Mesher"]=="Getfem":
        return PhySolver.GenerateWheelMesh(wheelDimensions=physicalDomain["wheelDimensions"],\
                                    meshSize=physicalDomain["meshSize"],\
                                    RefNumByRegion=physicalDomain["refNumByRegion"])
    elif physicalDomain["Mesher"]=="Gmsh":
        if "subcategory" not in physicalDomain.keys():
            return ExternalMesher.GenerateCoincidentHFLFMeshes(wheelExtMeshFile="wheel_ext",\
                                                           wheelMeshFile=physicalDomain["meshFilename"],\
                                                           interRadius=physicalDomain["interRadius"],\
                                                           wheelDim=physicalDomain["wheelDimensions"],\
                                                           meshSize=physicalDomain["meshSize"],\
                                                           version=physicalDomain["version"])
        elif physicalDomain["subcategory"]=="DentedWheelGenerator":
            myDentedWheel = ExternalMesher.DentedWheelGenerator(wheel_Dimensions=physicalDomain["wheel_Dimensions"],
                                               teeth_Size=physicalDomain["teeth_Size"],
                                               tread_Angle_deg=physicalDomain["tread_Angle_deg"],
                                               mesh_size=physicalDomain["mesh_size"]
                                               )
            myDentedWheel.GenerateMesh(outputFile=physicalDomain["meshFilename"])
            mesh=PhySolver.ImportGmshMesh(physicalDomain["meshFilename"]+".msh")
            taggedMesh=PhySolver.TagWheelMesh(mesh=mesh,
                                              wheelDimensions=(min(physicalDomain["wheel_Dimensions"]),max(physicalDomain["wheel_Dimensions"])),
                                              center=(0.0,max(physicalDomain["wheel_Dimensions"])),
                                              refNumByRegion=physicalDomain["refNumByRegion"])
            return taggedMesh

    else:
        raise Exception("Mesher "+str(physicalDomain["Mesher"])+" not supported")

def SimulatorGeneration(physicalDomain,physicalProperties):
    problemType=physicalProperties["ProblemType"]

    classNameByProblemType = {
                               "StaticMechanicalStandard":"GetfemMecaProblem",
                               "StaticMechanicalRolling":"GetfemRollingWheelProblem",
                               "QuasiStaticMechanicalStandard":"QuasiStaticMecanicalProblem",
                               "QuasiStaticMechanicalRolling":"QuasiStaticRollingProblem"
                               }

    try:
        simulator = globals()[classNameByProblemType[problemType]]()
    except KeyError:
        raise(Exception("Unable to treat this kind of problem !"))

    simulator.mesh=MeshGeneration(physicalDomain)
    simulator.refNumByRegion=physicalDomain["refNumByRegion"]

    filterPhysicalProperties={k: v for k, v in physicalProperties.items() if k!="ProblemType"}
    for physicalProperty,physicalValue in filterPhysicalProperties.items():
        attribute=setattr(simulator,physicalProperty,physicalValue)

    return simulator

def PhysicalCriteriaComputation(criteriaType,physicalProblem,field,criteriaParams=None):

    classNameByCriteriaType = {
                               "DeformedVolume":"DeformedVolume",
                               "NormalContactForces":"UnilateralContactPressure",
                               "FrictionContactForces":"FrictionalContactPressure",
                               "StrainEnergy":"TotalElasticEnergy",
                               "MaxStress":"MaxVonMises",
                               "MaxDeflection":"MaxDisp",
                               }

    try:
        criteria = globals()[classNameByCriteriaType[criteriaType]]()
    except KeyError:
        raise(Exception("Unable to treat this kind of problem !"))

    criteria.SetExternalSolutions(extSol)
    if criteriaParams is not None:
        return criteria.ComputeValue(**criteriaParams)
    return criteria.ComputeValue()

