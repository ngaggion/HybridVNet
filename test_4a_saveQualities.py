from functools import partial
from multiprocessing import Pool
import meshio
import os 
from utils.file_utils import load_folder
import numpy as np    
import pyvista

def process_subject(subject, tets, out_path):

    timesteps = load_folder(subject)
    for time in timesteps:
        print(time)
        nodes = np.load(os.path.join(time, "mesh.npy"))
        mesh = meshio.Mesh(nodes, cells = {'tetra': tets})
        
        savefolder = time.replace('Meshes', 'VTK_Meshes')
        if not os.path.exists(savefolder):
            os.makedirs(savefolder, exist_ok=True)
        
        savepath = os.path.join(savefolder, "mesh.vtk")
        mesh.write(savepath)
        
        del mesh, nodes
        
        mesh = pyvista.read(savepath)

        qual1 = mesh.compute_cell_quality(quality_measure='scaled_jacobian')["CellQuality"]
        qual2 = mesh.compute_cell_quality(quality_measure='aspect_ratio')["CellQuality"]
        
        qual1 = pyvista.convert_array(qual1)
        qual2 = pyvista.convert_array(qual2)
        
        np.save(os.path.join(time.replace('Meshes', 'VTK_Meshes'), "scaled_jacobian.npy"), qual1)
        np.save(os.path.join(time.replace('Meshes', 'VTK_Meshes'), "aspect_ratio.npy"), qual2)
        
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