import meshio
import numpy as np
import os 

if __name__ == "__main__":
    
    pred_path = "../Predictions/Surface/surface_roi_ds_1_lap_0.01/Meshes/3758998/time024/mesh.npy"
    gt_path = "../Dataset/Meshes/DownsampledMeshes/3758998/time024/fhm.npy"
    image_path = "../Dataset/Images/SAX_VTK/3758998/image_SAX_024.vtk"
    
    faces_path = "../Dataset/Meshes/DownsampledMeshes_files/faces_fhm_numpy.npy"
    
    pred = np.load(pred_path)
    gt = np.load(gt_path)
    faces = np.load(faces_path).astype(np.int64)
    
    outpath = "../ExtrasForViews"
    
    # copy the image to the outpath
    os.system("cp " + image_path + " " + outpath)
    
    mesh = meshio.Mesh(pred, [("triangle", faces)])
    gt_mesh = meshio.Mesh(gt, [("triangle", faces)])
    
    # save mesh
    meshio.write_points_cells(os.path.join(outpath, "pred.vtk"), mesh.points, mesh.cells)
    meshio.write_points_cells(os.path.join(outpath, "gt.vtk"), gt_mesh.points, gt_mesh.cells)