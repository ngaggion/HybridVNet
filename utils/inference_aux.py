import os 
from utils.file_utils import load_folder
import pandas as pd
import numpy as np
from trimesh import Trimesh
import SimpleITK as sitk

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
    width, height, num_slices = sax.GetSize()
    slice_gap = sax.GetSpacing()[2]
    spacing = sax.GetSpacing()
    origin = sax.GetOrigin()
    
    mesh_v = mesh - origin
    direction_matrix = np.array(sax.GetDirection()).reshape(3, 3)
    inverse_direction_matrix = np.linalg.inv(direction_matrix)
    mesh_v = np.dot(mesh_v, inverse_direction_matrix.T)
    mesh = mesh_v + origin
    
    full, grid = get_grid(mesh, faces, subs)

    mask_img = np.zeros((height, width, num_slices), dtype=np.uint8)

    slice_idx = slice_gap * np.arange(0, num_slices, dtype=float) + origin[2]
    xr = np.arange(0, mask_img.shape[1], dtype=float) * spacing[0] + origin[0]
    yr = np.arange(0, mask_img.shape[0], dtype=float) * spacing[1] + origin[1]

    xs, ys = np.meshgrid(xr, yr)
    
    for i, si in enumerate(slice_idx):
        pos = np.stack([xs.flatten(), ys.flatten(), si * np.ones(len(xs.flatten()))], axis=1)
        values = get_mask_values(full, grid, pos)
        mask_img[:, :, i] = values.reshape(height, width)

    return mask_img