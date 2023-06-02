import json
import os
import pickle as pkl
import numpy as np
import torch
from models.hybridGNet_3D import HybridGNet3D
from models.hybridGNet_3D_noLAX import HybridGNet3D as HybridGNet3D_noLAX
from models.utils import scipy_to_torch_sparse
from torchvision import transforms
from utils.dataset import (NumpyCardiacImageMeshDataset, 
                           alignMeshWithImage, 
                           PadToShape, 
                           SampleRoIVal, 
                           ToTensorForTest, 
                           ToTensorForTestRoI)
from utils.file_utils import load_folder

import pandas as pd

def configure_model(config):
    matrix_path = config['matrix_path']
 
    # Load mesh matrices and set up device
    with open(matrix_path, "rb") as f:
        dic = pkl.load(f)

    gpu = "cuda:" + str(config["cuda_device"])
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    config['device'] = device

    M = dic["M"]
    A = dic["A"]
    D = dic["D"]
    U = dic["U"]

    D_t = [scipy_to_torch_sparse(d).to(device).float() for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device).float() for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device).float() for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]
    
    config['n_nodes'] = num_nodes

    # Set up skip connections
    skip_connections = [True] * config['n_skips'] + [False] * (4 - config['n_skips']) if config['do_skip'] else [False] * 4
    
    # Initialize and load model
    
    if config['noLAX']:
        model = HybridGNet3D_noLAX(config, D_t, U_t, A_t, skip_connections).float().to(device)
    else:    
        model = HybridGNet3D(config, D_t, U_t, A_t, skip_connections).float().to(device)
    return model


def go_back(config, image, mesh_v, x0=0, y0=0):
    # Scales and translates mesh vertices back to original image space
    def get_both_paddings(desired, actual):
        pad = desired - actual
        v1, v2 = pad // 2, pad // 2
        if v1 + v2 < pad:
            v2 += 1
        return v1, v2

    outh, outw = config['h'], config['w']
    z = image.num_slices
    dz = get_both_paddings(16, z)

    mesh_v[:, 0] = (mesh_v[:, 0] * outh + x0) * image.spacing[0] + image.origin[0]
    mesh_v[:, 1] = (mesh_v[:, 1] * outw + y0) * image.spacing[1] + image.origin[1]
    mesh_v[:, 2] = (mesh_v[:, 2] * 16 - dz[0]) * image.slice_gap + image.origin[2]

    return mesh_v

def evaluate(target, output):
    mse = np.mean((target - output) ** 2)
    mae = np.mean(np.abs(target - output))
    rmse = np.sqrt(mse)
    return mse, mae, rmse    

def extract(id, subpartID):
    idxs = np.zeros(len(subpartID), dtype = bool)

    for i in range(0, len(subpartID)):
        idxs[i] = id in subpartID[i]
            
    return idxs
 
def segmentDataset(config, model, test_dataset, meshes_path, model_out_path):
    model.eval()
    device = config['device']
    
    evalDF = pd.DataFrame(columns=['Subject','Time', 'Subpart', 'MSE', 'MAE', 'RMSE'])

    subpart_list = np.loadtxt("../Dataset/Meshes/DownsampledMeshes_files/subparts_fhm.txt", dtype=str)
    subparts = ["LV", "RV", "LA", "RA", "aorta"]
    subpart_indices = []
    for subpart in subparts:
        subpart_indices.append(extract(subpart, subpart_list))

    j = 0
    with torch.no_grad():
        for t in range(len(test_dataset)):
            print('\r', t + 1, 'of', len(test_dataset), end='')

            sample = test_dataset[t]
            vtk, image, target = sample['image'], sample['array'].to(device), sample['mesh']
            lax2ch, lax3ch, lax4ch = sample['LAX2CH'].to(device), sample['LAX3CH'].to(device), sample['LAX4CH'].to(device)
            
            x0, y0 = (sample[k] for k in ['x0', 'y0']) if not config['full'] else (0, 0)

            subject, time = test_dataset.dataframe.iloc[t][['subject', 'time']]
            
            subj_time_path = os.path.join(meshes_path, subject.astype('str'), time)
            os.makedirs(subj_time_path, exist_ok=True)

            if not config['noLAX']:
                output, _ = model(image.unsqueeze(0), lax2ch.unsqueeze(0), lax3ch.unsqueeze(0), lax4ch.unsqueeze(0))
            else:
                output, _ = model(image.unsqueeze(0))
                
            mesh = go_back(config, vtk, output.squeeze(0).cpu().numpy(), x0, y0)
            
            gt_path = "../Dataset/Meshes/DownsampledMeshes/" + subject.astype('str') + "/" + time + "/fhm.npy"
            target = np.load(gt_path)

            np.save(os.path.join(subj_time_path, "mesh.npy"), mesh)
            
            MSE, MAE, RMSE = evaluate(target, mesh)
            evalDF.loc[j] = [subject, time, 'Full', MSE, MAE, RMSE]
            j+=1
            
            for i in range(0, len(subparts)):
                sub_mesh = mesh[subpart_indices[i]]
                sub_gt = target[subpart_indices[i]]
                MSE, MAE, RMSE = evaluate(sub_gt, sub_mesh)
                evalDF.loc[j] = [subject, time, subparts[i], MSE, MAE, RMSE]
                j+=1

    evalDF.to_csv(os.path.join(model_out_path, 'eval.csv'))

if __name__ == "__main__":
    class Mesh:
        def __init__(self, v, f):
            self.v = v
            self.f = f

    input = "weights"
    output = "../Predictions"

    models = load_folder(input)

    for model_path in models:
        config = json.load(open(os.path.join(model_path, "config.json")))
        if config['finished'] and os.path.isfile(os.path.join(model_path, "segmented.txt")):
            continue
        
        out_path = os.path.join(output, "Surface" if config['surface'] else "Volumetric")
            
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
                
        model_out_path = os.path.join(out_path, config['name'])
        os.makedirs(model_out_path, exist_ok=True)

        meshes_path = os.path.join(model_out_path, "Meshes")
        os.makedirs(meshes_path, exist_ok=True)

        model = configure_model(config)
        model.load_state_dict(torch.load(os.path.join(model_path, "bestMSE.pt"), map_location=config['device']))
        faces = np.load(config['faces_path']).astype(np.int32)

        part_file = f"files/train_test_splits/test_{'surface' if config['surface'] else 'volumetric'}_splits.csv"

        transform = transforms.Compose([
            alignMeshWithImage(),
            (PadToShape([210, 210, 16]) if config['full'] else SampleRoIVal([100, 100, 16])),
            ToTensorForTest() if config['full'] else ToTensorForTestRoI()
        ])
        
        print("Segmenting model", config['name'])
        test_dataset = NumpyCardiacImageMeshDataset(file=part_file, faces=faces, transform=transform)

        segmentDataset(config, model, test_dataset, meshes_path, model_out_path)
        print("")
        
        if config['finished']:
            # create a segmented.txt file to indicate that the model has been segmented
            with open(os.path.join(model_path, "segmented.txt"), "w") as f:
                f.write("True")
    