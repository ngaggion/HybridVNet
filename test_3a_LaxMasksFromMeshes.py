import numpy as np
from trimesh import Trimesh
import pandas as pd
import SimpleITK as sitk
import os
from utils.SaxImage import SAXImage
from utils.LaxImage import LAXImage

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


def to_matrix_vector(transform):
    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[0:ndimin, 0:ndimout]
    vector = transform[0:ndimin, ndimout]
    return matrix, vector

def get_mask_image_2CH(masked_fhm, grilla, lax, pid, Origin):
    # output mask
    mask_img = 0 * np.ones((lax.height, lax.width, lax.num_slices), dtype=np.uint8)

    # get slice index for z direction
    slice_idx = [0]

    # create image grid
    xr = np.arange(0, mask_img.shape[1], dtype=float)
    yr = np.arange(0, mask_img.shape[0], dtype=float)
    xs, ys = np.meshgrid(xr, yr)

    df = pd.read_csv('./csv/params_2CH.csv', index_col=False)
    matrix = df[(df["case_id"] == int(pid))].to_dict('list')
   
    M, b = to_matrix_vector(np.fromstring(str(matrix['trans'][0])[1:-1], dtype=float, sep=',').reshape(
                                      4, 4))

    for i, si in enumerate(slice_idx):
        pos = np.stack([xs.flatten(), ys.flatten(), si * np.ones(len(xs.flatten()))], axis=1)

        mapped_pos = []
        for index in range(len(pos)):
            
            p = pos[index]
            p_ = np.array([p[0], p[1], p[2]])
            new_p = np.dot(M, p_) + b
            new_p_ = np.array([new_p[0] * lax.spacing[0] + Origin[0], new_p[1] * lax.spacing[1] + Origin[1], new_p[2] * 10 + Origin[2]])
            mapped_pos.append(new_p_)
            
        pos = np.array(mapped_pos)
        values = get_mask_values(masked_fhm, grilla, pos)
        mask_img[:, :, i] = values.reshape(lax.height, lax.width)

    return mask_img


def get_mask_image_4CH(masked_fhm, grilla, lax, pid, Origin):
    # output mask
    mask_img = 0 * np.ones((lax.height, lax.width, lax.num_slices), dtype=np.uint8)

    # get slice index for z direction
    slice_idx = [0]

    # create image grid
    xr = np.arange(0, mask_img.shape[1], dtype=float)
    yr = np.arange(0, mask_img.shape[0], dtype=float)
    xs, ys = np.meshgrid(xr, yr)
   
    df = pd.read_csv('./csv/params_4CH.csv', index_col=False)
    matrix = df[(df["case_id"] == int(pid))].to_dict('list')
   
    M, b = to_matrix_vector(np.fromstring(str(matrix['trans'][0])[1:-1], dtype=float, sep=',').reshape(
                                      4, 4))

    for i, si in enumerate(slice_idx):
        
        pos = np.stack([xs.flatten(), ys.flatten(), si * np.ones(len(xs.flatten()))], axis=1)

        mapped_pos = []
        for index in range(len(pos)):
            
            p = pos[index]
            p_ = np.array([p[0], p[1], p[2]])
            new_p = np.dot(M, p_) + b
            new_p_ = np.array([new_p[0] * lax.spacing[0] + Origin[0], new_p[1] * lax.spacing[1] + Origin[1], new_p[2] * 10 + Origin[2]])
            mapped_pos.append(new_p_)
        pos = np.array(mapped_pos)
        values = get_mask_values(masked_fhm, grilla, pos)
        mask_img[:, :, i] = values.reshape(lax.height, lax.width)

    return mask_img


def get_grid(label, faces, subs):
    full_mesh = get_tmesh(label.copy(), faces, ["RA", "LA"], subs)

    VX_BG = 0
    VX_LA = 50
    VX_RA = 100

    grilla = np.zeros(full_mesh.shape, dtype=np.uint8)
    grilla.shape

    ra = get_tmesh(label.copy(), faces, ["RA"], subs)
    la = get_tmesh(label.copy(), faces, ["LA"], subs)

    grilla[full_mesh.matrix] = 10
    grilla = assign_voxels(grilla, la, 50, full_mesh)
    grilla = assign_voxels(grilla, ra, 100, full_mesh)

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

def save_nii(filename, data,spacing):
    dataITK = sitk.GetImageFromArray(data) 
    dataITK.SetSpacing(spacing)   
    sitk.WriteImage(dataITK, filename)
    
from utils.segmentationMetrics import HD, MCD
from medpy.metric import dc
import json
from functools import partial
from multiprocessing import Pool

def process_subject(subject, out_path, faces, subs, evaluate, overwrite):
    subject_id = subject.split("/")[-1]
    
    timesteps = load_folder(subject)
    for time in timesteps:
        time_id = time.split("/")[-1]
         
        save_folder = os.path.join(out_path, subject_id, time_id)
        if not os.path.exists(save_folder) or overwrite:
            os.makedirs(save_folder, exist_ok=True)
                    
        subject = "../Dataset/Subjects/" + subject_id + "/image/" + time_id

        SAX_PATH = os.path.join(subject, "SAX")
        LAX_PATH = os.path.join(subject, "LAX")
        LAX_2CH_PATH = os.path.join(LAX_PATH, "2CH", '0001')
        LAX_4CH_PATH = os.path.join(LAX_PATH, "4CH", '0001')

        SaxImage = SAXImage(SAX_PATH)
        Lax2CH = LAXImage(LAX_2CH_PATH)
        Lax4CH = LAXImage(LAX_4CH_PATH)
        
        mesh_path = os.path.join(time, "mesh.npy")

        points = np.load(mesh_path)

        #direction_matrix = np.array(SaxImage.direction).reshape(3, 3)
        #inverse_direction_matrix = np.linalg.inv(direction_matrix)

        #points = np.dot((points - SaxImage.origin), inverse_direction_matrix.T) + SaxImage.origin

        faces = np.load("../Dataset/SurfaceFiles/faces_fhm_numpy.npy")
        subs = np.loadtxt("../Dataset/SurfaceFiles/subparts_fhm.txt", dtype=str)

        fmesh, grilla = get_grid(points, faces, subs)
        pid = subject_id
        
        img2ch = get_mask_image_2CH(fmesh, grilla, Lax2CH, pid, SaxImage.origin)
        img4ch = get_mask_image_4CH(fmesh, grilla, Lax4CH, pid, SaxImage.origin)

        save_nii(save_folder + "/2CH.nii.gz", img2ch.transpose(2, 0, 1), [Lax2CH.spacing[0], Lax2CH.spacing[1], float(Lax2CH.slice_gap)])
        save_nii(save_folder + "/4CH.nii.gz", img4ch.transpose(2, 0, 1), [Lax4CH.spacing[0], Lax4CH.spacing[1], float(Lax4CH.slice_gap)])
        
    return
        
from utils.file_utils import load_folder

if __name__ == "__main__":
    input = "../Predictions/Surface/"
    overwrite = True
    evaluate = True
    
    faces = np.load("../Dataset/SurfaceFiles/faces_fhm_numpy.npy")
    subs = np.loadtxt("../Dataset/SurfaceFiles/subparts_fhm.txt", dtype=str)

    models = load_folder(input)
    
    for model_path in models:
        try:
            config = json.load(open(os.path.join(model_path.replace('Predictions/Surface', 'Code/weights'), "config.json")))
        except:
            continue
        
        if config['finished'] and os.path.isfile(os.path.join(model_path.replace('Predictions/Surface', 'Code/weights'), "segmented_lax.txt")):
            continue
        
        print("Segmenting model", model_path.split("/")[-1])
            
        i = 0
                
        out_path = os.path.join(model_path, "LaxMasks")

        if not os.path.exists(out_path) or overwrite:
            os.makedirs(out_path, exist_ok=True)
            
        mesh_path = os.path.join(model_path, "Meshes")
        subjects = load_folder(mesh_path)
        
        # Create a partial function with the common arguments
        func = partial(process_subject, out_path=out_path, faces=faces, subs=subs, evaluate=evaluate, overwrite=overwrite)
        
        with Pool(4) as p:
            p.map(func, subjects)
            
        print("")

        if config['finished']:
            # create a segmented.txt file to indicate that the model has been segmented
            with open(os.path.join(model_path.replace('Predictions/Surface', 'Code/weights'), "segmented_lax.txt"), "w") as f:
                f.write("True")
