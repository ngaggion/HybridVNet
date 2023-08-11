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

from utils.LaxImage import LAXImage
from utils.SaxImage import SAXImage
import cv2

faces = np.load("../Dataset/Meshes/DownsampledMeshes_files/faces_fhm_numpy.npy")
subs = np.loadtxt("../Dataset/Meshes/DownsampledMeshes_files/subparts_fhm.txt", dtype=str)

csv = pd.read_csv("files/train_test_splits/test_surface_splits.csv")
all_pairs = csv.values.tolist()

save_path = "../Dataset/LAX_Masks"

for pair in all_pairs:
    pid = str(pair[0])
    time = str(pair[1])
    meshfile = pair[2]
    imgfile = pair[3]
    lax_file = pair[4]

    image = SAXImage(imgfile)
    points = np.load(meshfile)

    LAX_2CH = lax_file + '/2CH/0001'
    LAX_4CH = lax_file + '/4CH/0001'
    
    save = os.path.join(save_path, pid, time)
    os.makedirs(save, exist_ok=True)

    lax_image1 = LAXImage(LAX_2CH)
    lax_image2 = LAXImage(LAX_4CH)

    fmesh, grilla = get_grid(points, faces, subs)

    img2ch = get_mask_image_2CH(fmesh, grilla, lax_image1, pid, image.origin)
    img4ch = get_mask_image_4CH(fmesh, grilla, lax_image2, pid, image.origin)
    
    #lax2_array = lax_image1.pixel_array
    #lax4_array = lax_image2.pixel_array
    
    #cv2.imwrite(os.path.join(save, "2CH.png"), lax2_array)
    #cv2.imwrite(os.path.join(save, "4CH.png"), lax4_array)

    save_nii(os.path.join(save, "2CH.nii.gz"), img2ch.transpose(2, 0, 1), [lax_image1.spacing[0], lax_image1.spacing[1], float(lax_image1.slice_gap)])
    save_nii(os.path.join(save, "4CH.nii.gz"), img4ch.transpose(2, 0, 1), [lax_image2.spacing[0], lax_image2.spacing[1], float(lax_image2.slice_gap)])

    #break