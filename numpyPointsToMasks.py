import os 
from utils.file_utils import load_folder
import pandas as pd
import numpy as np
from trimesh import Trimesh
import SimpleITK as sitk
from utils.SaxImage import SAXImage

VOXEL_SIZE = 1.0

def extract_subpart(points, faces, ids, subpartID):
    # points: points
    # faces: faces
    # ids: IDs to extract
    # subpartID: list of IDs for every point

    ids = [ids] if not isinstance(ids, list) else ids
    
    n_points = len(points)
    new_points = []
    point_ids = []

    for i in range(points.shape[0]):
        for id_ in ids:
            if id_ in subpartID[i]:
                new_points.append(points[i])
                point_ids.append(i)
                break

    points = np.array(new_points)

    point_ids_set = set(point_ids)
    triangles = [
        tuple(triangle)
        for triangle in faces
        if all([pp in point_ids_set for pp in triangle])
    ]

    id_mapping = {x: i for i, x in enumerate(point_ids)}
    triangles = np.array(
        [tuple([id_mapping[x] for x in triangle]) for triangle in triangles]
    )
    
    return points, triangles


def get_grid(label, faces, subs):
    full_mesh = get_tmesh(label.copy(), faces, ["RV", "LV", "MVP", "AVP", "TVP", "PVP"], subs)

    grilla = np.zeros(full_mesh.shape, dtype=np.uint8)
    grilla.shape

    rv_wall = get_tmesh(label.copy(), faces, ["RV"], subs)
    lv_Endo = get_tmesh(label.copy(), faces, ["LV", "MVP", "AVP"], subs)
    lv_myo = get_tmesh(label.copy(), faces, ["LV"], subs)

    grilla[full_mesh.matrix] = 100
    grilla = assign_voxels(grilla, rv_wall, 150, full_mesh)
    grilla = assign_voxels(grilla, lv_Endo, 50, full_mesh)
    grilla = assign_voxels(grilla, lv_myo, 250, full_mesh)

    return full_mesh, grilla


def assign_voxels(grid, mesh, value, full):
        """
        Internal method. Don't use it directly.
        """
        # get the _grid and indices where voxels are true
        xs, ys, zs = np.meshgrid(range(mesh.shape[0]), range(mesh.shape[1]), range(mesh.shape[2]), indexing='ij')
        xs = xs[mesh.matrix]
        ys = ys[mesh.matrix]
        zs = zs[mesh.matrix]

        # convert those indices into point coordinates
        mesh_pts = mesh.indices_to_points(np.stack([xs, ys, zs]).transpose())

        # using main voxel_grid to get the indices inside rvlv
        fhm_idx = full.points_to_indices(mesh_pts)

        # assign value
        grid[fhm_idx[:, 0], fhm_idx[:, 1], fhm_idx[:, 2]] = value

        return grid


def get_tmesh(v, f, subpartID, subpart_ids):
    v, f = extract_subpart(v.copy(), f, subpartID, subpart_ids)
    return Trimesh(vertices=v, faces=f).voxelized(VOXEL_SIZE).fill()


def get_mask_values(fhm, grilla, pos, outbound=0):
    # get valid indices
    indices = fhm.points_to_indices(pos)

    j = (0 <= indices[:, 0]) & (indices[:, 0] < grilla.shape[0]) & \
        (0 <= indices[:, 1]) & (indices[:, 1] < grilla.shape[1]) & \
        (0 <= indices[:, 2]) & (indices[:, 2] < grilla.shape[2])
        
    values = outbound * np.ones(pos.shape[0], dtype=np.uint8)
    values[j] = grilla[indices[j, 0], indices[j, 1], indices[j, 2]]

    return values


def get_mask_image(mesh, faces, subs, sax):
    """
    Generate mask image for each image slice 
    Mask values are defined in MaskedFHM class (see there)
    Returns H x W x S mask image.
    """
    
    full, grid = get_grid(mesh, faces, subs)

    # output mask
    mask_img = np.zeros((sax.height, sax.width, sax.num_slices), dtype=np.uint8)

    # get slice index for z direction
    slice_idx = sax.slice_gap * range(sax.num_slices) + sax.origin[2]

    # create image grid
    xr = np.arange(0, mask_img.shape[1], dtype=float) * sax.spacing[0] + sax.origin[0]
    yr = np.arange(0, mask_img.shape[0], dtype=float) * sax.spacing[1] + sax.origin[1]
    xs, ys = np.meshgrid(xr, yr)
    
    for i, si in enumerate(slice_idx):
        # get position for each slice_idx
        pos = np.stack([xs.flatten(), ys.flatten(), si * np.ones(len(xs.flatten()))], axis=1)

        # get_mask_values
        values = get_mask_values(full, grid, pos)

        # assign value
        mask_img[:, :, i] = values.reshape(sax.height, sax.width)

    return mask_img


def save_nii(filename, data,spacing):
    dataITK = sitk.GetImageFromArray(data) 
    dataITK.SetSpacing(spacing)   
    sitk.WriteImage(dataITK, filename)
    
from utils.segmentationMetrics import HD, MCD
from medpy.metric import dc
import json

if __name__ == "__main__":
    input = "../Predictions/Surface/"
    overwrite = True
    evaluate = True
    
    faces = np.load("../Dataset/Meshes/DownsampledMeshes_files/faces_fhm_numpy.npy")
    subs = np.loadtxt("../Dataset/Meshes/DownsampledMeshes_files/subparts_fhm.txt", dtype=str)

    models = load_folder(input)
    
    for model_path in models:
        config = json.load(open(os.path.join(model_path.replace('Predictions/Surface', 'Code/weights'), "config.json")))
        
        if config['finished'] and os.path.isfile(os.path.join(model_path.replace('Predictions/Surface', 'Code/weights'), "segmented_mask.txt")):
            continue
        
        print("Segmenting model", model_path.split("/")[-1])
        
        if evaluate:
            dataframe = pd.DataFrame(columns=["ID",  
                                "LV Myo - DC", "LV Myo - HD", "LV Myo - MCD",
                                "LV Endo - DC", "LV Endo - HD", "LV Endo - MCD",
                                "RV Endo - DC", "RV Endo - HD", "RV Endo - MCD"])
    
        i = 0
                
        out_path = os.path.join(model_path, "Masks")

        if not os.path.exists(out_path) or overwrite:
            os.makedirs(out_path, exist_ok=True)
            
        mesh_path = os.path.join(model_path, "Meshes")
        
        subjects = load_folder(mesh_path)
        j = 0
        
        for subject in subjects:
            print('\r', 'Subject', j + 1, 'of', len(subjects), end='')
            j+=1
            subject_id = subject.split("/")[-1]
            
            timesteps = load_folder(subject)
            for time in timesteps:
                time_id = time.split("/")[-1]
                
                image_path = "../Dataset/Images/SAX_VTK/" + subject_id + "/image_SAX_" + time_id[-3:] + ".vtk"
                image = SAXImage(image_path)
                
                mesh = np.load(os.path.join(time, "mesh.npy"))
                mask_seg = get_mask_image(mesh, faces, subs, image).transpose(2,0,1)
                
                save_folder = os.path.join(out_path, subject_id, time_id)
                if not os.path.exists(save_folder) or overwrite:
                    os.makedirs(save_folder, exist_ok=True)
                    
                save_path = os.path.join(save_folder, "mask.nii.gz")
                save_nii(save_path, mask_seg, [image.spacing[0],image.spacing[1],float(image.slice_gap)])
                
                if evaluate:
                    gt_path = os.path.join("../Dataset/LV_RV_Masks", subject_id, time_id, "mask.nii.gz")
                    gt = sitk.ReadImage(gt_path)
                    gt = sitk.GetArrayFromImage(gt)
                    
                    dice_myo = dc(gt == 250, mask_seg == 250)
                    hausdorff_myo = HD(gt == 250, mask_seg == 250)
                    assd_value_myo = MCD(gt == 250, mask_seg == 250)
                    
                    dice_Endo = dc(gt == 50, mask_seg == 50)
                    hausdorff_Endo = HD(gt == 50, mask_seg == 50)
                    assd_value_Endo = MCD(gt == 50, mask_seg == 50)

                    dice_rv_Endo = dc(gt == 100, mask_seg == 100)
                    hausdorff_rv_Endo = HD(gt == 100, mask_seg == 100)
                    assd_value_rv_Endo = MCD(gt == 100, mask_seg == 100)

                    dataframe.loc[i] = [id, 
                        dice_myo, hausdorff_myo, assd_value_myo, 
                        dice_Endo, hausdorff_Endo, assd_value_Endo, 
                        dice_rv_Endo, hausdorff_rv_Endo, assd_value_rv_Endo]
                    
                    i = i + 1
        
        print("")
        if evaluate:
            dataframe.to_csv(os.path.join(model_path, "metrics.csv"), index=False)
        
        if config['finished']:
            # create a segmented.txt file to indicate that the model has been segmented
            with open(os.path.join(model_path.replace('Predictions/Surface', 'Code/weights'), "segmented_mask.txt"), "w") as f:
                f.write("True")
