import argparse
import json
import os
import pickle as pkl
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from models.hybridGNet_3D import HybridGNet3D

from models.utils import Pool, scipy_to_torch_sparse
from pytorch3d.loss import (mesh_edge_loss, mesh_laplacian_smoothing,
                            mesh_normal_consistency)

from pytorch3d.structures import Meshes
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.dataset import (CardiacImageMeshDataset, PadArraysToSquareShape, CropArraysToSquareShape, RandomCropBoth,
                           RandomScalingBoth, Rotate, ToTorchTensors, AugColor, AlignMeshWithSaxImage, CropSax)
                           

np.random.seed(12)

import gc

def well_shaped_loss(nodes, index_a, index_b):
    edges_a = nodes[:, index_a, :]
    edges_b = nodes[:, index_b, :]
    edges = edges_a - edges_b

    lengths = torch.norm(edges, dim=3)
    mean_length = torch.mean(lengths, axis=2).unsqueeze(2)
    a = (lengths - mean_length) / (mean_length + 1e-15)
    loss_value = torch.mean(a ** 2)

    return loss_value

def setup_tensorboard_log_directory(tensorboard, name, config):
    folder = os.path.join(tensorboard, name)

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, 'config.json'), 'w') as f:
        config['device'] = str(config['device'])
        json.dump(config, f)

    return folder

def trainer(train_dataset, val_dataset, models, names, configs, config):
    torch.manual_seed(0)
    device = config['device']
    print('Training on', device)

    for model in models:
        model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['val_batch_size'], num_workers=0, pin_memory=True)
    
    optimizers = []
    for model in models:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        optimizers.append(optimizer)
        
    tensorboard = "weights"
    
    writers = []
    folders = []
    
    for j in range(len(models)):
        name = names[j]
        model_config = configs[j]
        
        folder = setup_tensorboard_log_directory(tensorboard, name, model_config)
        writer = SummaryWriter(log_dir=folder)
        writers.append(writer)
        folders.append(folder)

    best_mse = [1e12] * len(models)

    print('Training ...')

    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])
    pool = Pool()

    faces = torch.from_numpy(np.load(config['faces_path']).astype('int')).float().to(device).unsqueeze(0)

    if not config['surface']:
        tets = np.load(config['faces_path'].replace('faces.npy', 'tets.npy')).astype('int')
        tets = torch.from_numpy(tets)
        index_a = torch.tensor([0,0,0,1,1,2])
        index_b = torch.tensor([1,2,3,2,3,3])

        index_a = tets[:,index_a]
        index_a.requires_grad = False
        index_b = tets[:,index_b]
        index_b.requires_grad = False

        index_a = index_a.to(config['device'])
        index_b = index_b.to(config['device'])
  
    for epoch in range(1, config['epochs'] + 1):
        for model in models:
            model.train()

        train_loss_avg = [0] * len(models)
        train_rec_loss_avg = [0] * len(models)
        train_kld_loss_avg = [0] * len(models)
        train_w_laplacian_loss = [0] * len(models)
        train_w_shape_loss = [0] * len(models)
        train_ds_loss = [0] * len(models)
        val_loss_avg = [0] * len(models)
        
        num_batches = 0
        
        print('Beggining epoch %s'%epoch)
        
        t0 = time.time()
        for sample_batched in train_loader:
            image, target, lax2ch, lax3ch, lax4ch = sample_batched['Sax_Array'].to(device), sample_batched['Mesh'].to(device), sample_batched['Lax2CH_Array'].to(device), sample_batched['Lax3CH_Array'].to(device), sample_batched['Lax4CH_Array'].to(device)

            target_d1 = pool(target, models[0].downsample_matrices[0])
            target_d2 = pool(target_d1, models[0].downsample_matrices[1])
            target_d3 = pool(target_d2, models[0].downsample_matrices[2])
            target_d4 = pool(target_d3, models[0].downsample_matrices[3])
                    
            for j in range(len(models)):
                # Weight for mesh laplacian smoothing
                w_laplacian = configs[j]['w_laplacian']
                # Weight for the deep supervision
                w_ds = configs[j]['w_ds']
                # Weight for volumetric shape loss
                w_shape = configs[j]['w_shape']

                out, ds = models[j](image, lax2ch, lax3ch, lax4ch)
                        
                ds4, ds3, ds2, ds1 = ds

                optimizers[j].zero_grad()
            
                # Mean squared error loss
                outloss = F.mse_loss(out, target) 

                if w_ds > 0:
                    ds4loss = F.mse_loss(ds4, target_d1)
                    ds3loss = F.mse_loss(ds3, target_d2)
                    ds2loss = F.mse_loss(ds2, target_d3)
                    ds1loss = F.mse_loss(ds1, target_d4)
                    ds_loss = w_ds * (ds4loss + ds3loss + ds2loss + ds1loss) 
                else:
                    ds_loss = 0
                
                f = [faces]*out.shape[0]
                f = torch.cat(f)
                meshes = Meshes(out, f)
                
                # mesh laplacian smoothing
                if w_laplacian > 0:
                    loss_laplacian = mesh_laplacian_smoothing(meshes, method="cot")
                    train_w_laplacian_loss[j] += w_laplacian * loss_laplacian.item()
                else:
                    loss_laplacian = 0

                if w_shape > 0 and not config['surface']:
                    well_shaped_loss = well_shaped_loss(out, index_a, index_b)
                    train_w_shape_loss[j] += well_shaped_loss.item()
                else:
                    well_shaped_loss = 0

                # Kullback-Leibler divergence term
                kld_loss = -0.5 * torch.mean(torch.mean(1 + models[j].log_var - models[j].mu ** 2 - models[j].log_var.exp(), dim=1), dim=0)
                
                loss = outloss + ds_loss + models[j].kld_weight * kld_loss + w_laplacian * loss_laplacian + w_shape * well_shaped_loss

                loss.backward()
                
                # one step of the optmizer (using the gradients from backpropagation)
                optimizers[j].step()
            
                train_kld_loss_avg[j] += models[j].kld_weight * kld_loss.item()
                train_rec_loss_avg[j] += outloss.item()
                train_loss_avg[j] += loss.item()
                
                if w_ds > 0:
                    train_ds_loss[j] += ds_loss.item()
                else:
                    train_ds_loss[j] += 0

            num_batches += 1

            if num_batches % 100 == 0:
                t2 = time.time()
                print('SubEpoch %s, Time %.6f' % (num_batches, t2-t0))
                t0 = time.time()
                
                
        gc.collect()            
        
        for j in range(len(models)):
            train_loss_avg[j] /= num_batches
            train_rec_loss_avg[j] /= num_batches
            train_kld_loss_avg[j] /= num_batches
            train_w_laplacian_loss[j] /= num_batches
            train_w_shape_loss[j] /= num_batches
            train_ds_loss[j] /= num_batches

            print('Model %s'%names[j])
            print('Epoch [%d / %d] train average reconstruction error: %f' % (epoch, config['epochs'], train_rec_loss_avg[j]))

            writer = writers[j]
            writer.add_scalar('Train/Loss', train_loss_avg[j], epoch)
            writer.add_scalar('Train/MSE', train_rec_loss_avg[j], epoch)        
            writer.add_scalar('Train_Regularization/Loss kld', train_kld_loss_avg[j], epoch)
            
            w_laplacian = configs[j]['w_laplacian']
            w_ds = configs[j]['w_ds']
            w_shape = configs[j]['w_shape']
                
            if w_laplacian > 0:
                writer.add_scalar('Train_Regularization/Loss laplacian', train_w_laplacian_loss[j], epoch)
            if w_shape > 0:
                writer.add_scalar('Train_Regularization/Loss shape', train_w_shape_loss[j], epoch)
            if ds_loss > 0:
                writer.add_scalar('Train_Regularization/Loss ds', train_ds_loss[j], epoch)
                
        num_batches = 0
        
        if epoch < 150:
            val_rate = 20
        elif epoch < 300:
            val_rate = 10
        elif epoch < 450:
            val_rate = 5
        else:
            val_rate = 5
            
        if epoch  % val_rate == 0:
            print("")
            
            for model in models:
                model.eval()

            with torch.no_grad():
                t0 = time.time()
                for sample_batched in val_loader:
                    image, target, lax2ch, lax3ch, lax4ch = sample_batched['Sax_Array'].to(device), sample_batched['Mesh'].to(device), sample_batched['Lax2CH_Array'].to(device), sample_batched['Lax3CH_Array'].to(device), sample_batched['Lax4CH_Array'].to(device)
                    
                    for j in range(len(models)):
                        out, _ = models[j](image, lax2ch, lax3ch, lax4ch)
                        loss_rec = F.mse_loss(out, target).item()
                        val_loss_avg[j] += loss_rec
                    
                    num_batches += 1

            t2 = time.time()
            
            for j in range(len(models)):
                
                val_loss_avg[j] /= num_batches
                print('Model %s'%names[j])
                print('Epoch [%d / %d] validation average reconstruction error: %.6f' % (epoch, config['epochs'], val_loss_avg[j]))
                
                if val_loss_avg[j] < best_mse[j]:
                    best_mse[j] = val_loss_avg[j]
                    print('Model Saved MSE')
                    torch.save(models[j].state_dict(), os.path.join(folders[j],  "bestMSE.pt"))
                    
                writer = writers[j]
                writer.add_scalar('Validation/MSE', val_loss_avg[j] , epoch)
            
        
        gc.collect()      
        
        scheduler.step()

        for j in range(len(models)):
            torch.save(models[j].state_dict(), os.path.join(folders[j], "final.pt"))
        
    
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        config['finished'] = True
        json.dump(config, f)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", default = 600, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 1, type = int)
    parser.add_argument("--gamma", default = 0.99, type = float)

    parser.add_argument("--batch_size", default = 4, type = int)
    parser.add_argument("--val_batch_size", default = 4, type = int)
    
    parser.add_argument("--K", default = 6, type = int)
    parser.add_argument("--w_edge", default = 0, type = float)
    parser.add_argument("--average_edge_length", default = 0.04, type = float)
    parser.add_argument("--w_normal", default = 0, type = float)
    parser.add_argument("--w_laplacian", default = 0, type = float)
    parser.add_argument("--w_shape", default = 0, type = float)
    parser.add_argument("--w_ds", default = 1, type = float)
    parser.add_argument("--kld_weight", default = 1e-5, type = float)
    parser.add_argument("--weight_decay", default = 1e-5, type = float)

    parser.add_argument("--fold", default = 0, type = int)
    parser.add_argument("--n_folds", default = 10, type = int)

    parser.add_argument("--do_skip", dest='do_skip', action='store_true')
    parser.set_defaults(do_skip=False)

    parser.add_argument("--grad_prob", default = 1.0, type = float)
    
    parser.add_argument("--n_skips", default = 0, type = int)    
    parser.add_argument("--cuda_device", default = 0, type = int)
    
    parser.add_argument("--load", default = "", type = str)
    parser.add_argument("--full", dest='full', action='store_true')
    parser.set_defaults(full=False)

    parser.add_argument("--volumetric", dest='surface', action='store_false')
    parser.set_defaults(surface=True)

    parser.add_argument("--latents3D", default = 64, type = int)
    parser.add_argument("--latents2D", default = 16, type = int)
    
    parser.add_argument("--rotate", default=10, type=int)

    config = parser.parse_args()
    config = vars(config)

    
    part_file = '../Dataset/train_split.csv'
        
    # Set surface/volumetric configs and paths
    if config['surface']:
        faces_path = "../Dataset/SurfaceFiles/faces_fhm_numpy.npy"
        matrix_path = "../Dataset/SurfaceFiles/Matrices_fhm.pkl"
    else:
        faces_path = "../Dataset/VolumetricFiles/vol_faces.npy"
        matrix_path = "../Dataset/VolumetricFiles/Matrices_volumetric.pkl"

    faces = np.load(faces_path).astype('int')
    config.update({'faces_path': faces_path, 'matrix_path': matrix_path})
    
    # Load mesh matrices and set up device
    with open(matrix_path, "rb") as f:
        class Mesh:
            def __init__(self, v, f):
                self.v = v
                self.f = f
        dic = pkl.load(f)

    gpu = "cuda:" + str(config["cuda_device"])
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    config['device'] = device
    config['finished'] = False

    M = dic["M"]
    A = dic["A"]
    D = dic["D"]
    U = dic["U"]

    D_t = [scipy_to_torch_sparse(d).to(device).float() for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device).float() for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device).float() for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]
    
    config['n_nodes'] = num_nodes
    config['filters'] = [3, 16, 32, 32, 64, 64]   
    
    if config['full']:
        config['h'] = 210
        config['w'] = 210
        config['slices'] = 16
        
        all_transforms_train = transforms.Compose([
                                    AlignMeshWithSaxImage(),
                                    RandomScalingBoth(),
                                    Rotate(config['rotate']),
                                    AugColor(0.0),
                                    ToTorchTensors()
                                ])
        
        all_transforms_val = transforms.Compose([
                                    AlignMeshWithSaxImage(),
                                    PadArraysToSquareShape(),
                                    ToTorchTensors()
                                ])
    else:
        config['h'] = 100
        config['w'] = 100
        config['slices'] = 16
        
        all_transforms_train = transforms.Compose([
                                    AlignMeshWithSaxImage(),
                                    RandomScalingBoth(),
                                    Rotate(config['rotate']),
                                    CropSax(),
                                    AugColor(0.0),
                                    ToTorchTensors()
                                ])
        
        all_transforms_val = transforms.Compose([
                                    AlignMeshWithSaxImage(),
                                    CropArraysToSquareShape(),
                                    ToTorchTensors()
                                ])
        
    if config['surface']:
        mesh_type = "Surface"
    else:
        mesh_type = "Volumetric"
    
    # The training dataset file automatically performs k-fold cross validation with 5 folds
    train_dataset = CardiacImageMeshDataset(part_file, "../Dataset/Subjects", mode = "Training", mesh_type = mesh_type,
                                            transform = all_transforms_train)        
    val_dataset = CardiacImageMeshDataset(part_file, "../Dataset/Subjects", mode = "Validation", mesh_type = mesh_type,
                                          transform = all_transforms_val)

    # Set up skip connections
    skip_connections = [True] * config['n_skips'] + [False] * (4 - config['n_skips']) if config['do_skip'] else [False] * 4

    # Initialize and load model
    
    models = []
    names = []
    configs = []
    
    config1 = config.copy()
    config1['name'] = 'WDS_1_WL_0.01_3D_64_2D_16_KL_1e-3_no_color'
    config1['latents3D'] = 64
    config1['latents2D'] = 16
    config1['kld_weight'] = 1e-3
    config1['w_ds'] = 1
    config1['w_laplacian'] = 0.01
    
    models.append(HybridGNet3D(config1, D_t, U_t, A_t, skip_connections).float())
    names.append(config1['name'])
    configs.append(config1)    
    
    config2 = config.copy()
    config2['name'] = 'WDS_1_WL_0.01_3D_64_2D_16_KL_1e-4_no_color'
    config2['latents3D'] = 64
    config2['latents2D'] = 16
    config2['kld_weight'] = 1e-4
    config2['w_ds'] = 1
    config2['w_laplacian'] = 0.01
    
    models.append(HybridGNet3D(config2, D_t, U_t, A_t, skip_connections).float())
    names.append(config2['name'])
    configs.append(config2)
    
    config3 = config.copy()
    config3['name'] = 'WDS_1_WL_0.01_3D_64_2D_16_KL_1e-5_no_color'
    config3['latents3D'] = 64
    config3['latents2D'] = 16
    config3['kld_weight'] = 1e-5
    config3['w_ds'] = 1
    config3['w_laplacian'] = 0.01
    
    models.append(HybridGNet3D(config3, D_t, U_t, A_t, skip_connections).float())
    names.append(config3['name'])
    configs.append(config3)
    
    # Train the model
    trainer(train_dataset, val_dataset, models, names, configs, config)