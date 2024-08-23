import json
import os
import torch
from models.hybridGNet_3D import HybridGNet3D
from models.hybridGNet_3D_noLAX import HybridGNet3D as HybridGNet3D_noLAX
from torchvision import transforms
import SimpleITK as sitk
import meshio 

from utils.dataset_inference_w_LAX_example import PredictDataset, ToTorchTensors, PadArraysToSquareShape
from utils.inference_aux import configure_model, go_back, save_nii, get_mask_image

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
            path = meshes_path + "/" + str(subject) + "/" 
            os.makedirs(path, exist_ok=True)
            path += str(time).replace('.nii.gz',".stl")
            
            meshio.write(path, mesh)

            # Save segmentation mask for SAX Images
            mask_seg = get_mask_image(mesh_vertices, faces, subpart_list, sax_image).transpose(2, 0, 1)
            save_nii(path.replace(".stl", ".nii.gz"), mask_seg, sax_image.GetOrigin(), sax_image.GetSpacing(), sax_image.GetDirection())

if __name__ == "__main__":
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