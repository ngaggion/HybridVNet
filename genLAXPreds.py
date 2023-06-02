import numpy as np
from trimesh import Trimesh
import pandas as pd
import SimpleITK as sitk
import os

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

    df = pd.read_csv('files/params/params_2CH.csv', index_col=False)
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
   
    df = pd.read_csv('files/params/params_4CH.csv', index_col=False)
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

def get_both_paddings(desired, actual):
    pad = (desired - actual)
    
    v1 = int(pad/2)
    v2 = int(pad/2) 
    if (v1 + v2) < pad:
        v2 += 1
    
    return (v1, v2)

def go_back(image, mesh_v, x0, y0):
    h, w, z = image.height, image.width, image.num_slices
    dz = get_both_paddings(16, z)

    mesh_v[:,0] = (mesh_v[:,0] * 100 + x0) * image.spacing[0] + image.origin[0] 
    mesh_v[:,1] = (mesh_v[:,1] * 100 + y0) * image.spacing[1] + image.origin[1] 
    mesh_v[:,2] = (mesh_v[:,2] * 16 - dz[0]) * image.slice_gap + image.origin[2] 

    return mesh_v

import cv2 

def MCD(seg_A, seg_B):
    table_md = []
    seg_A = seg_A.transpose(2,1,0)
    seg_B = seg_B.transpose(2,1,0)
    seg_A[seg_A>0] = 1
    seg_B[seg_B>0] = 1
    
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))
            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

     
            md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * 1.8
            table_md += [md]

    mean_md = np.mean(table_md) if table_md else None
    return mean_md


def HD(seg_A, seg_B):
    
    table_hd = []
    seg_A = seg_A.transpose(2,1,0)
    seg_B = seg_B.transpose(2,1,0)
    seg_A[seg_A>0] = 1
    seg_B[seg_B>0] = 1
    
    
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))

            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * 1.8
            table_hd += [hd]

    mean_hd = np.mean(table_hd) if table_hd else None
    return mean_hd

from utils.LaxImage import LAXImage
from utils.SaxImage import SAXImage
import json
from utils.file_utils import load_folder
from medpy.metric import dc

if __name__ == "__main__":
    input = "../Predictions/Surface/"
    overwrite = True
    evaluate = True
    
    faces = np.load("../Dataset/Meshes/DownsampledMeshes_files/faces_fhm_numpy.npy")
    subs = np.loadtxt("../Dataset/Meshes/DownsampledMeshes_files/subparts_fhm.txt", dtype=str)
    csv = pd.read_csv("files/train_test_splits/test_surface_splits.csv")


    models = load_folder(input)
    
    for model_path in models:
        config = json.load(open(os.path.join(model_path.replace('Predictions/Surface', 'Code/weights'), "config.json")))
        
        if config['finished'] and os.path.isfile(os.path.join(model_path.replace('Predictions/Surface', 'Code/weights'), "segmented_lax_mask.txt")):
            continue
        
        print("Segmenting model", model_path.split("/")[-1])
        
        if evaluate:
            dataframe = pd.DataFrame(columns=["ID", 
                                "LA 2CH - DC", "LA 2CH - HD", "LA 2CH - MCD",
                                "LA 4CH - DC", "LA 4CH - HD", "LA 4CH - MCD",
                                "RA 4CH - DC", "RA 4CH - HD", "RA 4CH - MCD"])
    
        i = 0
                
        out_path = os.path.join(model_path, "LaxMasks")

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

                pid = subject_id 

                image_path = "../Dataset/Images/SAX_VTK/" + subject_id + "/image_SAX_" + time_id[-3:] + ".vtk"
                image = SAXImage(image_path)
                points = np.load(os.path.join(time, "mesh.npy"))

                lax_file = "../Dataset/Images/LAX/" + subject_id + '/' + time_id

                LAX_2CH = lax_file + '/2CH/0001'
                LAX_4CH = lax_file + '/4CH/0001'
                
                save_folder = os.path.join(out_path, subject_id, time_id)
                if not os.path.exists(save_folder) or overwrite:
                    os.makedirs(save_folder, exist_ok=True)

                lax_image1 = LAXImage(LAX_2CH)
                lax_image2 = LAXImage(LAX_4CH)

                fmesh, grilla = get_grid(points, faces, subs)
                
                img2ch = get_mask_image_2CH(fmesh, grilla, lax_image1, pid, image.origin).transpose(2, 0, 1)
                img4ch = get_mask_image_4CH(fmesh, grilla, lax_image2, pid, image.origin).transpose(2, 0, 1)

                save_nii(os.path.join(save_folder, "2CH.nii.gz"), img2ch, [lax_image1.spacing[0], lax_image1.spacing[1], float(lax_image1.slice_gap)])
                save_nii(os.path.join(save_folder, "4CH.nii.gz"), img4ch, [lax_image2.spacing[0], lax_image2.spacing[1], float(lax_image2.slice_gap)])

                if evaluate:
                    gt_2CH = os.path.join("../Dataset/LAX_Masks", subject_id, time_id, "2CH.nii.gz")
                    gt_4CH = os.path.join("../Dataset/LAX_Masks", subject_id, time_id, "4CH.nii.gz")
                    
                    gt_2CH = sitk.ReadImage(gt_2CH)
                    gt_2CH = sitk.GetArrayFromImage(gt_2CH)
                    gt_4CH = sitk.ReadImage(gt_4CH)
                    gt_4CH = sitk.GetArrayFromImage(gt_4CH)
                    
                    dice_la2ch = dc(gt_2CH == 50, img2ch == 50)
                    hausdorff_la2ch = HD(gt_2CH == 50, img2ch == 50)
                    assd_value_la2ch = MCD(gt_2CH == 50, img2ch == 50)
                    
                    dice_la4ch = dc(gt_4CH == 50, img4ch == 50)
                    hausdorff_la4ch = HD(gt_4CH == 50, img4ch == 50)
                    assd_value_la4ch = MCD(gt_4CH == 50, img4ch == 50)

                    dice_ra4ch = dc(gt_4CH == 100, img4ch == 100)
                    hausdorff_ra4ch = HD(gt_4CH == 100, img4ch == 100)
                    assd_value_ra4ch = MCD(gt_4CH == 100, img4ch == 100)

                    dataframe.loc[i] = [id, 
                        dice_la2ch, hausdorff_la2ch, assd_value_la2ch, 
                        dice_la4ch, hausdorff_la4ch, assd_value_la4ch, 
                        dice_ra4ch, hausdorff_ra4ch, assd_value_ra4ch]
                                                
                    i = i + 1
        
        print("")
        if evaluate:
            dataframe.to_csv(os.path.join(model_path, "lax_metrics.csv"), index=False)
        
        if config['finished']:
            # create a segmented.txt file to indicate that the model has been segmented
            with open(os.path.join(model_path.replace('Predictions/Surface', 'Code/weights'), "segmented_lax_mask.txt"), "w") as f:
                f.write("True")