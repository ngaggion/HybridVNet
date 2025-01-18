import os
from utils.file_utils import load_folder
import numpy as np
from multiprocessing import Pool, cpu_count

def process_subject(args):
    """Process all timesteps for a single subject"""
    subject, metric = args
    subject_data = []
    timesteps = load_folder(subject)
    for time in timesteps:
        metric_data = np.load(os.path.join(time, f"{metric}.npy"))
        subject_data.append(metric_data.reshape(-1, 1))
    return np.array(subject_data)

if __name__ == "__main__":
    input_path = "../Predictions/Volumetric/"
    models = load_folder(input_path)

    n_cores = cpu_count()  # Get number of CPU cores
    #metrics = ['scaled_jacobian', 'aspect_ratio', 'shape', 'skew', 'mean_ratio', 'distortion']
    metrics = ['skew', 'mean_ratio', 'distortion']
    
    for model_path in models:
        print(model_path)
        mesh_path = os.path.join(model_path, "VTK_Meshes")
        subjects = load_folder(mesh_path)
        
        for metric in metrics:
            print(f"Processing {metric}")
            
            # Prepare arguments for parallel processing
            args = [(subject, metric) for subject in subjects]
            
            # Process subjects in parallel
            with Pool(processes=n_cores) as pool:
                results = pool.map(process_subject, args)
            
            # Combine results
            metric_matrix = np.vstack(results)
            
            # Calculate means
            mean_metric_subjects = np.mean(metric_matrix, axis=1)
            mean_metric_tetra = np.mean(metric_matrix, axis=0)
            
            # Save results
            np.save(os.path.join(model_path, f"mean_{metric}_subjects.npy"), mean_metric_subjects)
            np.save(os.path.join(model_path, f"mean_{metric}_tetra.npy"), mean_metric_tetra)
            
            # Clean up
            del metric_matrix, mean_metric_subjects, mean_metric_tetra, results