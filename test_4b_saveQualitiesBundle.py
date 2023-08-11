import os 
from utils.file_utils import load_folder
import numpy as np    
       
if __name__ == "__main__":
    input_path = "../Predictions/Volumetric/" 
    
    models = load_folder(input_path)
    
    for model_path in models:
        print(model_path)
        i = 0
            
        mesh_path = os.path.join(model_path, "VTK_Meshes")
        subjects = load_folder(mesh_path)
        
        scaled_jacobian_matrix = []
        aspect_ratio_matrix = []
        
        for subject in subjects:
            timesteps = load_folder(subject)
            for time in timesteps:
                scaled_jacobian = np.load(os.path.join(time, "scaled_jacobian.npy"))
                aspect_ratio = np.load(os.path.join(time, "aspect_ratio.npy"))
                
                scaled_jacobian_matrix.append(scaled_jacobian.reshape(-1, 1))
                aspect_ratio_matrix.append(aspect_ratio.reshape(-1, 1))
        
        scaled_jacobian_matrix = np.array(scaled_jacobian_matrix)
        aspect_ratio_matrix = np.array(aspect_ratio_matrix)
        
        # Subjects and times are in the first dimension, cells in the second
        mean_scaled_jacobian_subjects = np.mean(scaled_jacobian_matrix, axis=1)
        mean_scaled_jacobian_tetra = np.mean(scaled_jacobian_matrix, axis=0)
        
        mean_aspect_ratio_subjects = np.mean(aspect_ratio_matrix, axis=1)
        mean_aspect_ratio_tetra = np.mean(aspect_ratio_matrix, axis=0)
        
        np.save(os.path.join(model_path, "mean_scaled_jacobian_subjects.npy"), mean_scaled_jacobian_subjects)
        np.save(os.path.join(model_path, "mean_scaled_jacobian_tetra.npy"), mean_scaled_jacobian_tetra)
        np.save(os.path.join(model_path, "mean_aspect_ratio_subjects.npy"), mean_aspect_ratio_subjects)
        np.save(os.path.join(model_path, "mean_aspect_ratio_tetra.npy"), mean_aspect_ratio_tetra)
        
        # delete everything
        del scaled_jacobian_matrix, aspect_ratio_matrix
        del mean_scaled_jacobian_subjects, mean_scaled_jacobian_tetra, mean_aspect_ratio_subjects, mean_aspect_ratio_tetra
    
