#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lips.physical_simulator.GetfemSimulator.GetfemHSA as PhySolver
import lips.physical_simulator.GetfemSimulator.MeshGenerationTools as ExternalMesher
from lips.physical_simulator.GetfemSimulator.GetfemWheelProblem import GetfemMecaProblem,GetfemRollingWheelProblem
from lips.physical_simulator.GetfemSimulator.GetfemWheelProblemQuasiStatic import QuasiStaticRollingProblem,QuasiStaticMecanicalProblem
from lips.physical_simulator.GetfemSimulator.GetfemInterpolationTools import FEMInterpolationOnSupport,InterpolateSolOnNodes
from lips.physical_simulator.GetfemSimulator.PhysicalCriteria import DeformedVolume,UnilateralContactPressure,FrictionalContactPressure,TotalElasticEnergy,MaxVonMises,MaxDisp

def GetfemInterpolationOnSupport(simulator,field,gridSupport):
    physical_problem=simulator._simulator
    return FEMInterpolationOnSupport(phyProblem=physical_problem,originalField=field,targetSupport=gridSupport)

def InterpolationOnCloudPoints(fieldSupport,fieldValue,phyProblem):
    targetSupport=phyProblem.get_solverOrder_positions()
    return InterpolateSolOnNodes(fieldSupport=fieldSupport,fieldValue=fieldValue,targetSupport=targetSupport)

def MeshGeneration(physical_domain):
    if physical_domain["Mesher"]=="Getfem":
        return PhySolver.GenerateWheelMesh(wheelDimensions=physical_domain["wheelDimensions"],\
                                    meshSize=physical_domain["meshSize"],\
                                    RefNumByRegion=physical_domain["refNumByRegion"])
    elif physical_domain["Mesher"]=="Gmsh":
        if "subcategory" not in physical_domain.keys():
            return ExternalMesher.GenerateCoincidentHFLFMeshes(wheelExtMeshFile="wheel_ext",\
                                                           wheelMeshFile=physical_domain["meshFilename"],\
                                                           interRadius=physical_domain["interRadius"],\
                                                           wheelDim=physical_domain["wheelDimensions"],\
                                                           meshSize=physical_domain["meshSize"],\
                                                           version=physical_domain["version"])
        elif physical_domain["subcategory"]=="DentedWheelGenerator":
            myDentedWheel = ExternalMesher.DentedWheelGenerator(wheel_Dimensions=physical_domain["wheel_Dimensions"],
                                               teeth_Size=physical_domain["teeth_Size"],
                                               tread_Angle_deg=physical_domain["tread_Angle_deg"],
                                               mesh_size=physical_domain["mesh_size"]
                                               )
            myDentedWheel.GenerateMesh(outputFile=physical_domain["meshFilename"])
            mesh=PhySolver.ImportGmshMesh(physical_domain["meshFilename"]+".msh")
            taggedMesh=PhySolver.TagWheelMesh(mesh=mesh,
                                              wheelDimensions=(min(physical_domain["wheel_Dimensions"]),max(physical_domain["wheel_Dimensions"])),
                                              center=(0.0,max(physical_domain["wheel_Dimensions"])),
                                              refNumByRegion=physical_domain["refNumByRegion"])
            return taggedMesh

    else:
        raise Exception("Mesher "+str(physical_domain["Mesher"])+" not supported")

def SimulatorGeneration(physical_domain,physicalProperties):
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

    simulator.mesh=MeshGeneration(physical_domain)
    simulator.refNumByRegion=physical_domain["refNumByRegion"]

    filterPhysicalProperties={k: v for k, v in physicalProperties.items() if k!="ProblemType"}
    for physicalProperty,physicalValue in filterPhysicalProperties.items():
        attribute=setattr(simulator,physicalProperty,physicalValue)

    return simulator

def PhysicalCriteriaComputation(criteriaType,simulator,field,criteriaParams=None):

    classNameByCriteriaType = {
                               "DeformedVolume":"DeformedVolume",
                               "NormalContactForces":"UnilateralContactPressure",
                               "FrictionContactForces":"FrictionalContactPressure",
                               "StrainEnergy":"TotalElasticEnergy",
                               "MaxStress":"MaxVonMises",
                               "MaxDeflection":"MaxDisp",
                               }

    physical_problem=simulator._simulator
    try:
        criteria = globals()[classNameByCriteriaType[criteriaType]](problem=physical_problem)
    except KeyError:
        raise(Exception("Unable to treat this kind of problem !"))

    criteria.SetExternalSolutions(field)
    if criteriaParams is not None:
        return criteria.ComputeValue(**criteriaParams)
    return criteria.ComputeValue()

