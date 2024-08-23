import os 
import pickle as pkl
import numpy as np
import torch
from models.utils import scipy_to_torch_sparse
import SimpleITK as sitk
from trimesh import Trimesh

VOXEL_SIZE = 1.0

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
    def get_both_paddings(desired, actual):
        pad = desired - actual
        v1, v2 = pad // 2, pad // 2
        if v1 + v2 < pad:
            v2 += 1
        return v1, v2

    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = image.GetDirection()

    pixel_size = np.array([spacing[0], spacing[1], spacing[2]])
    
    outh, outw = config['h'], config['w']
    
    image = sitk.GetArrayFromImage(image)
    original_h, original_w, z = min(image.shape[1], outh), min(image.shape[2],outw), min(image.shape[0], 16)

    if config['full']:
        x_pad = get_both_paddings(outw, original_w)[0]
        y_pad = get_both_paddings(outh, original_h)[0]
        z_pad = get_both_paddings(16, z)[0]

        mesh_v[:, 0] = mesh_v[:, 0] * outw - x_pad
        mesh_v[:, 1] = mesh_v[:, 1] * outh - y_pad
        mesh_v[:, 2] = mesh_v[:, 2] * 16 - z_pad
        mesh_v[:, 2] = z - mesh_v[:, 2] # Invert z axis

        mesh_v[:, 0] *= pixel_size[0]
        mesh_v[:, 1] *= pixel_size[1]
        mesh_v[:, 2] *= pixel_size[2]

        direction_matrix = np.array(direction).reshape(3, 3)
        mesh_v = np.dot(mesh_v, direction_matrix.T)

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

def extract_subpart(points, faces, ids, subpartID):
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
    xs, ys, zs = np.meshgrid(range(mesh.shape[0]), range(mesh.shape[1]), range(mesh.shape[2]), indexing='ij')
    xs = xs[mesh.matrix]
    ys = ys[mesh.matrix]
    zs = zs[mesh.matrix]

    mesh_pts = mesh.indices_to_points(np.stack([xs, ys, zs]).transpose())

    fhm_idx = full.points_to_indices(mesh_pts)

    grid[fhm_idx[:, 0], fhm_idx[:, 1], fhm_idx[:, 2]] = value

    return grid

def get_tmesh(v, f, subpartID, subpart_ids):
    v, f = extract_subpart(v.copy(), f, subpartID, subpart_ids)
    return Trimesh(vertices=v, faces=f).voxelized(VOXEL_SIZE).fill()

def get_mask_values(fhm, grilla, pos, outbound=0):
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