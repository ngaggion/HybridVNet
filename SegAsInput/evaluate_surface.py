import torch
import numpy as np
from chamfer import chamfer_distance
from scipy.spatial.distance import directed_hausdorff

def compute_chamfer(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer Distance between two point clouds using PyTorch3D
    
    Args:
        x: (B, N, 3) tensor of points
        y: (B, M, 3) tensor of points
        
    Returns:
        chamfer_dist: mean chamfer distance
    """
    # PyTorch3D implementation returns loss and normals distance
    # We only want the distance component
    dist, _ = chamfer_distance(x, y, point_reduction='mean', batch_reduction=None)
    return dist

def compute_hausdorff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute bidirectional Hausdorff Distance between two point clouds
    
    Args:
        x: (B, N, 3) tensor of points
        y: (B, M, 3) tensor of points
        
    Returns:
        hausdorff_dist: bidirectional Hausdorff distance
    """
    dist = max(directed_hausdorff(x.squeeze().numpy(), y.squeeze().numpy())[0],
               directed_hausdorff(y.squeeze().numpy(), x.squeeze().numpy())[0])
    return dist


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

import pandas as pd

if __name__ == "__main__":
    subs = np.loadtxt("/home/ngaggion/DATA/HybridGNet3D/Dataset/SurfaceFiles/subparts_fhm.txt", dtype=str)
    faces = np.load("/home/ngaggion/DATA/HybridGNet3D/Dataset/SurfaceFiles/faces_fhm_numpy.npy")

    split = "/home/ngaggion/DATA/HybridGNet3D/BaselineChen/ChenSplits/test_split.csv"
    split = pd.read_csv(split)

    results = []

    for i in range(len(split)):
        subject = split["subject"][i]
        time = split["time"][i]

        points1 = np.load("/home/ngaggion/DATA/HybridGNet3D/BaselineChen/Surface/seg_noLAX_surf/Meshes/" + str(subject) +"/"+ time+ "/mesh.npy")
        points2 = np.load("/home/ngaggion/DATA/HybridGNet3D/Backup/Dataset/Meshes/DownsampledMeshes/" + str(subject) +"/"+ time+ "/fhm.npy")
        
        points1, faces1 = extract_subpart(points1, faces, ["LV", "RV"], subs)
        points2, faces2 = extract_subpart(points2, faces, ["LV", "RV"], subs)
        
        points1 = torch.tensor(points1).unsqueeze(0).float()
        points2 = torch.tensor(points2).unsqueeze(0).float()

        cd = compute_chamfer(points1, points2).item()
        hd = compute_hausdorff(points1, points2)

        print(f"Subject: {subject}, Time: {time}, CD: {cd:.4f}, HD: {hd:.4f}")

        results.append([subject, time, cd, hd])
    
    results = pd.DataFrame(results, columns=["subject", "time", "CD", "HD"])
    results.to_csv("results_surface.csv", index=False)