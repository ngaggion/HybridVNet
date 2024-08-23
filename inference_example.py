import json
import os
import pickle as pkl
import numpy as np
import torch
from models.hybridGNet_3D import HybridGNet3D
from models.hybridGNet_3D_noLAX import HybridGNet3D as HybridGNet3D_noLAX
from models.utils import scipy_to_torch_sparse
from torchvision import transforms
import SimpleITK as sitk
import meshio 

# Import the PredictDataset and transformation classes
from utils.dataset_inference_w_LAX_example import PredictDataset, ToTorchTensors, PadArraysToSquareShape
from utils.inference_aux import get_mask_image

# Some auxiliars to avoid installing psbody-mesh during test
class PartialUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except:
            return FakeClass
class FakeClass:
    def __init__(self, *args, **kwargs):
        pass
def partial_load(file):
    return PartialUnpickler(file).load()

def configure_model(config):
    matrix_path = config['matrix_path']
 
    with open(matrix_path, "rb") as f:
        dic = partial_load(f)
    
    gpu = "cuda:" + str(config["cuda_device"])
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    config['device'] = device

    M, A, D, U = dic["M"], dic["A"], dic["D"], dic["U"]

    D_t = [scipy_to_torch_sparse(d).to(device).float() for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device).float() for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device).float() for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]
    
    config['n_nodes'] = num_nodes

    skip_connections = [True] * config['n_skips'] + [False] * (4 - config['n_skips']) if config['do_skip'] else [False] * 4

    if "noLAX" in config['name']:      
        model = HybridGNet3D_noLAX(config, D_t, U_t, A_t, skip_connections).float().to(device)
    else:
        model = HybridGNet3D(config, D_t, U_t, A_t, skip_connections).float().to(device)

    return model, M[0].f


def go_back(config, image, mesh_v, x0=0, y0=0):
    # Scales and translates mesh vertices back to original image space
    def get_both_paddings(desired, actual):
        pad = desired - actual
        v1, v2 = pad // 2, pad // 2
        if v1 + v2 < pad:
            v2 += 1
        return v1, v2

    # Get the origin of the image
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = image.GetDirection()

    # Calculate the pixel size in each dimension
    pixel_size = np.array([spacing[0], spacing[1], spacing[2]])
    
    outh, outw = config['h'], config['w']
    
    image = sitk.GetArrayFromImage(image)
    original_h, original_w, z = min(image.shape[1], outh), min(image.shape[2],outw), min(image.shape[0], 16)

    if config['full']:
        # We have to remove the padding           
        x_pad = get_both_paddings(outw, original_w)[0]
        y_pad = get_both_paddings(outh, original_h)[0]
        z_pad = get_both_paddings(16, z)[0]

        # Transform to pixel coordinates
        mesh_v[:, 0] = mesh_v[:, 0] * outw - x_pad
        mesh_v[:, 1] = mesh_v[:, 1] * outh - y_pad
        mesh_v[:, 2] = mesh_v[:, 2] * 16 - z_pad
        mesh_v[:, 2] = z - mesh_v[:, 2] # Invert z axis

        # Transform to physical coordinates
        mesh_v[:, 0] *= pixel_size[0]
        mesh_v[:, 1] *= pixel_size[1]
        mesh_v[:, 2] *= pixel_size[2]

        # Apply direction matrix
        direction_matrix = np.array(direction).reshape(3, 3)
        mesh_v = np.dot(mesh_v, direction_matrix.T)

        # Add origin
        mesh_v[:, 0] += origin[0]
        mesh_v[:, 1] += origin[1]
        mesh_v[:, 2] += origin[2]

    return mesh_v

def save_nii(filename, data, origin, spacing, direction):
    dataITK = sitk.GetImageFromArray(data) 
    dataITK.SetSpacing(spacing)
    dataITK.SetOrigin(origin)
    dataITK.SetDirection(direction)
    sitk.WriteImage(dataITK, filename)


def predict_meshes(config, model, faces, predict_dataset, meshes_path):
    model.eval()
    device = config['device']

    subpart_list = np.loadtxt("../HybridVNet_weights/SurfaceFiles/subparts_fhm.txt", dtype=str)

    with torch.no_grad():
        for t in range(len(predict_dataset)):
            print('\r', t + 1, 'of', len(predict_dataset), end='')

            sample = predict_dataset[t]
            image = sample['Sax_Array'].to(device)
            sax_image = sample['SAX']
            
            subject, time = predict_dataset.dataframe.iloc[t][['subject', 'time']]
            
            subj_time_path = os.path.join(meshes_path, str(subject), str(time))

            if "noLAX" in config['name']:
                output, _ = model(image.unsqueeze(0))
            else:
                lax2ch = sample['Lax2CH_Array'].to(device)
                lax3ch = sample['Lax3CH_Array'].to(device)
                lax4ch = sample['Lax4CH_Array'].to(device)
                output, _ = model(image.unsqueeze(0), lax2ch.unsqueeze(0), lax3ch.unsqueeze(0), lax4ch.unsqueeze(0))
            
            mesh_vertices = go_back(config, sax_image, output.squeeze(0).cpu().numpy())
            
            # Create a meshio Mesh object
            mesh = meshio.Mesh(
                points=mesh_vertices,
                cells=[("triangle", faces)]
            )

            # Save as STL
            path = meshes_path + str(subject) + "/" 
            os.makedirs(path, exist_ok=True)
            path += str(time).replace('.nii.gz',".stl")
            
            meshio.write(path, mesh)

            # Save segmentation mask for SAX Images
            mask_seg = get_mask_image(mesh_vertices, faces, subpart_list, sax_image).transpose(2, 0, 1)
            save_nii(path.replace(".stl", ".nii.gz"), mask_seg, sax_image.GetOrigin(), sax_image.GetSpacing(), sax_image.GetDirection())


if __name__ == "__main__":
    class Mesh:
        def __init__(self, v, f):
            self.v = v
            self.f = f

    output_dir = "../Abdul/Predictions"
    os.makedirs(output_dir, exist_ok=True)

    model_path = "../HybridVNet_weights/FullSAX_2CH_3CH_4CH"
    #model_path = "../HybridVNet_weights/FullSAX_noLAX"

    config = json.load(open(os.path.join(model_path, "config.json")))
    
    model, faces = configure_model(config)
    model.load_state_dict(torch.load(os.path.join(model_path, "final.pt"), map_location=config['device']))

    transform = transforms.Compose([
        PadArraysToSquareShape(),
        ToTorchTensors()
    ])
    
    print("Predicting meshes for model", config['name'])

    predict_dataset = PredictDataset("../DATA/Abdul/dataset_abdul_test/images", transform=transform)
    
    predict_meshes(config, model, faces, predict_dataset, "../DATA/Abdul/dataset_abdul_test/predictions/mesh")