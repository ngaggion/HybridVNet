from functools import partial
from multiprocessing import Pool
import meshio
import os 
from utils.file_utils import load_folder
import numpy as np    
import pyvista as pv
import vtk

# as pv only provides a subset of vtk's functionality, we need to use vtk directly
def compute_cell_quality(mesh, metric):
    if metric == "MEAN_RATIO":
        alg = vtk.vtkCellQuality()
        alg.SetInputData(mesh)
        alg.SetQualityMeasure(33)
        alg.SetUndefinedQuality(-1)
        alg.Update()
        out = pv.wrap(alg.GetOutput())
        return out["CellQuality"]
    elif metric == "EQUIANGLE_SKEW":
        alg = vtk.vtkCellQuality()
        alg.SetInputData(mesh)
        alg.SetQualityMeasure(29)
        alg.SetUndefinedQuality(-1)
        alg.Update()
        out = pv.wrap(alg.GetOutput())
        return out["CellQuality"]
    elif metric == "DISTORTION":
        alg = vtk.vtkCellQuality()
        alg.SetInputData(mesh)
        alg.SetQualityMeasure(15)
        alg.SetUndefinedQuality(-1)
        alg.Update()
        out = pv.wrap(alg.GetOutput())
        return out["CellQuality"]
    else:
        raise ValueError(f"Metric {metric} not supported")
        

def process_subject(subject, tets, out_path):

    timesteps = load_folder(subject)
    for time in timesteps:
        print(time)

        savefolder = time.replace('Meshes', 'VTK_Meshes')
        if not os.path.exists(savefolder):
            os.makedirs(savefolder, exist_ok=True)
        
        savepath = os.path.join(savefolder, "mesh.vtk")
        
        if not os.path.exists(savepath):       
            nodes = np.load(os.path.join(time, "mesh.npy"))
            mesh = meshio.Mesh(nodes, cells = {'tetra': tets})
            
            mesh.write(savepath)
            
            del mesh, nodes
        
        mesh = pv.read(savepath)

        #qual1 = mesh.compute_cell_quality(quality_measure='scaled_jacobian')["CellQuality"]
        #qual2 = mesh.compute_cell_quality(quality_measure='aspect_ratio')["CellQuality"]
        #qual3 = mesh.compute_cell_quality(quality_measure='shape')["CellQuality"]
        qual4 = compute_cell_quality(mesh, "EQUIANGLE_SKEW")
        qual5 = compute_cell_quality(mesh, "DISTORTION")
        qual6 = compute_cell_quality(mesh, "MEAN_RATIO")
        
        #qual1 = pv.convert_array(qual1)
        #qual2 = pv.convert_array(qual2)
        #qual3 = pv.convert_array(qual3)
        qual4 = pv.convert_array(qual4)
        qual5 = pv.convert_array(qual5)
        qual6 = pv.convert_array(qual6)
        
        #np.save(os.path.join(time.replace('Meshes', 'VTK_Meshes'), "scaled_jacobian.npy"), qual1)
        #np.save(os.path.join(time.replace('Meshes', 'VTK_Meshes'), "aspect_ratio.npy"), qual2)
        #np.save(os.path.join(time.replace('Meshes', 'VTK_Meshes'), "shape.npy"), qual3)
        np.save(os.path.join(time.replace('Meshes', 'VTK_Meshes'), "skew.npy"), qual4)
        np.save(os.path.join(time.replace('Meshes', 'VTK_Meshes'), "distortion.npy"), qual5)
        np.save(os.path.join(time.replace('Meshes', 'VTK_Meshes'), "mean_ratio.npy"), qual6)
        
    return
        

if __name__ == "__main__":
    input_path = "../Predictions/Volumetric/"
    overwrite = True
    evaluate = True
    
    tets = np.load("../Dataset/VolumetricFiles/vol_tets.npy")

    models = load_folder(input_path)
    
    for model_path in models:
        i = 0
                
        out_path = os.path.join(model_path, "VTK_Meshes")

        if not os.path.exists(out_path) or overwrite:
            os.makedirs(out_path, exist_ok=True)
            
        mesh_path = os.path.join(model_path, "Meshes")
        subjects = load_folder(mesh_path)
        
        # Create a partial function with the common arguments
        func = partial(process_subject, tets=tets, out_path=out_path)
        
        with Pool(8) as p:
            p.map(func, subjects)
            
        print("")