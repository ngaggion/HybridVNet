import pandas as pd 
from utils.SaxImage import SAXImage
import SimpleITK as sitk
import os
import numpy as np

train_splits = "../Dataset/train_split.csv"
test_splits = "../Dataset/test_split.csv"

train_splits = pd.read_csv(train_splits)
test_splits = pd.read_csv(test_splits)

# Get only the subject and time columns

train_splits = train_splits[['subject', 'time']]
test_splits = test_splits[['subject', 'time']]

splits = [train_splits, test_splits]

for split in splits:
    for index, row in split.iterrows():
        subject = str(row['subject'])
        time = row['time']
        print(row['subject'], row['time'])
        
        path = "../Dataset/Subjects/" + subject + "/image/" + time 
        
        SAX_PATH = os.path.join(path, "SAX")
        SaxImage = SAXImage(SAX_PATH)
        img1 = SaxImage.SaxImage
        
        img2_path = "../Backup/Dataset/Images/SAX_VTK/" + subject + "/image_SAX_" + time[4:] + ".vtk"
        img2 = sitk.ReadImage(img2_path)
            
        mesh_path = "../Backup/Dataset/Meshes/VolumetricMeshes/" + subject + "/" + time + "/fhm_vol.npy"
        point_set_modified = np.load(mesh_path)
        
        # Get the image properties of the image with modified metadata
        direction_modified = np.array(img2.GetDirection()).reshape((3, 3))
        origin_modified = np.array(img2.GetOrigin())

        # Get the image properties of the image with original metadata
        direction_original = np.array(img1.GetDirection()).reshape((3, 3))
        origin_original = np.array(img1.GetOrigin())

        # Compute the transformation from modified to original physical space
        direction_transform = np.linalg.inv(direction_modified) @ direction_original

        # Apply the transformation to the point set to obtain the points in the original physical space
        point_set_original = np.dot(point_set_modified - origin_modified, direction_transform.T) + origin_original

        # Save the point set
        
        outpath = path = "../Dataset/Subjects/" + subject + "/mesh/" + time + "/"
        
        try:
            os.makedirs(outpath)
        except:
            pass
        
        np.save(outpath + "volumetric.npy", point_set_original)    