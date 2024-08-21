import pandas as pd
import pathlib
import re
import numpy as np

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def load_folder(folder):
    paths = pathlib.Path(folder).glob('*')
    paths = [str(path) for path in paths]
    paths.sort(key = natural_key)

    return paths

def extract(id, subpartID):
    idxs = np.zeros(len(subpartID), dtype = bool)

    for i in range(0, len(subpartID)):
        idxs[i] = id in subpartID[i]
            
    return idxs

from utils.SaxImage_VTK import SAXImage

model = "/home/ngaggion/DATA/HybridGNet3D/Predictions/YanSurface/Meshes"
gtp = "/home/ngaggion/DATA/HybridGNet3D/Backup/Dataset/Meshes/DownsampledMeshes"
subjects = load_folder(gtp)

dataframe = pd.DataFrame(columns=["ID", "Model", "Subpart" ,"MAE", "MSE", "RMSE"])

subpart_list = np.loadtxt("/home/ngaggion/DATA/HybridGNet3D/Dataset/SurfaceFiles/subparts_fhm.txt", dtype=str)

subparts = ["LV", "RV", "LA", "RA", "aorta"]

indices = []
for subpart in subparts:
    indices.append(extract(subpart, subpart_list))

i = 0
for subject in subjects:
    times = load_folder(subject)
    for time in times:
        gt_path = time + '/fhm.npy'
        
        id = subject.split('/')[-1] + ' ' + time.split('/')[-1]

        mesh_path = time.replace(gtp, model) + '/mesh.npy'
        try:
            res_mesh = np.load(mesh_path)
        except:
            continue

        print(id)

        gt_mesh = np.load(gt_path)

        image = "/home/ngaggion/DATA/HybridGNet3D/Backup/Dataset/Images/SAX_VTK/" + subject.split('/')[-1] + "/image_SAX_" + time.split('/')[-1][-3:] + ".vtk"
        image = SAXImage(image)

        res_mesh = res_mesh - np.mean(res_mesh, axis=0)
        res_mesh = res_mesh + np.mean(gt_mesh, axis=0)

        mse = np.mean((res_mesh - gt_mesh)**2)
        mae = np.mean(np.abs(res_mesh - gt_mesh))
        rmse = np.sqrt(mse)
        
        dataframe.loc[i] = [id, model, "Full", mae, mse, rmse]

        i+=1

        for j in range(0,5):
            sub_mesh = res_mesh[indices[j]]
            sub_gt = gt_mesh[indices[j]]

            mse = np.mean((sub_mesh - sub_gt)**2)
            mae = np.mean(np.abs(sub_mesh - sub_gt))
            rmse = np.sqrt(mse)

            dataframe.loc[i] = [id, model, subparts[j], mae, mse, rmse]

            i+=1   

dataframe.to_csv("/home/ngaggion/DATA/HybridGNet3D/Predictions/YanSurface/Results.csv")   
