from utils.SaxImage_OLD import SAXImage
import SimpleITK as sitk
import os
import numpy as np
import cv2
from skimage import transform
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import pandas as pd


class CardiacImageMeshDataset(Dataset):
    def __init__(self, file, dataset_path, mode = None, mesh_type = 'surface', val_fold = 0, K = 10, transform=None):
        csv = pd.read_csv(file)

        subjects = csv['subject'].unique()
        np.random.seed(12)
        np.random.shuffle(subjects)

        if mode == 'Training':
            self.subjects = subjects[:int(len(subjects)*0.9)]
        elif mode == 'Validation':
            self.subjects = subjects[int(len(subjects)*0.9):]
        else:
            self.subjects = subjects

        self.dataframe = csv[csv['subject'].isin(self.subjects)]
        self.transform = transform
        self.mesh_type = mesh_type
        self.dataset_path = dataset_path
        
        print("Mode: ", mode)
        print("Total subjects:", len(self.subjects))
        print("Total pairs of images with annotations:", len(self.dataframe))

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datapoint = self.dataframe.iloc[idx]
        subject = datapoint['subject']
        time = datapoint['time']
        
        SAX_PATH = os.path.join("../Backup/Dataset/Images/SAX_VTK", str(subject), "image_SAX_%s.vtk" % time[-3:])

        SaxImage = SAXImage(SAX_PATH)
        SaxImage_array = SaxImage.pixel_array
        SaxImage_array = (SaxImage_array - np.min(SaxImage_array)) / (np.max(SaxImage_array) - np.min(SaxImage_array))

        if self.mesh_type == 'Surface':
            mesh_path = os.path.join("../Backup/Dataset/Meshes/DownsampledMeshes/", str(subject), time, "fhm.npy")
            mesh = np.load(mesh_path)
        elif self.mesh_type == 'Volumetric':
            mesh_path = os.path.join("../Backup/Dataset/Meshes/VolumetricMeshes/", str(subject), time, "fhm_vol.npy")
            mesh = np.load(mesh_path)
        else:
            raise ValueError("Mesh type not supported")

        sample = {'SAX': SaxImage, 'Mesh': mesh, 'Sax_Array': SaxImage_array}
        
        if self.transform:
            sample = self.transform(sample)
        
        sax_shape = sample["Sax_Array"].shape
        
        if sax_shape[0] == 0 or sax_shape[1] == 0 or sax_shape[2] == 0 or sax_shape[3] == 0:
            return self.__getitem__(idx)          
        
        return sample
    

class AlignMeshWithSaxImage(object):
    """
    Aligns the mesh with the SAX image.
    """

    def __call__(self, sample):
        sax_image = sample['SAX']
        mesh = sample['Mesh']
        
        # Get the origin of the image
        origin = np.array(sax_image.origin)
        
        # Calculate the pixel size in each dimension
        pixel_size = np.array([sax_image.spacing[0], sax_image.spacing[1], sax_image.slice_gap])

        # Convert the physical points to voxel indices by subtracting the origin and multiplying with the inverse direction matrix
        voxel_indices = mesh - origin

        # Convert the voxel indices to image space by dividing by the pixel size
        image_space_points = voxel_indices / pixel_size
        
        sample['Mesh'] = image_space_points
        
        return sample
    

def _get_both_paddings(desired, actual):
        pad = (desired - actual)
        
        v1 = int(pad / 2)
        v2 = int(pad / 2) 
        if (v1 + v2) < pad:
            v2 += 1
        
        return (v1, v2)
    
    
class PadArraysToSquareShape(object):
    """
    Zero pads SAX image arrays to fixed square shape.
    SAX_IMAGE_SHAPE = (210, 210, 16)
    """

    def __call__(self, sample):
        SAX_IMAGE_SHAPE = (210, 210, 16)
        
        sax_array = sample['Sax_Array']
        
        sax_h_paddings = _get_both_paddings(SAX_IMAGE_SHAPE[0], sax_array.shape[0])
        sax_w_paddings = _get_both_paddings(SAX_IMAGE_SHAPE[1], sax_array.shape[1])
        sax_z_paddings = _get_both_paddings(SAX_IMAGE_SHAPE[2], sax_array.shape[2])
        
        sax_array = np.pad(sax_array, (sax_h_paddings, sax_w_paddings, sax_z_paddings), 'constant', constant_values=0)
        
        mesh = sample['Mesh']
        mesh[:, 0] += sax_w_paddings[0]
        mesh[:, 1] += sax_h_paddings[0]
        mesh[:, 2] += sax_z_paddings[0]
        sample['Mesh'] = mesh
        
        sample['Sax_Array'] = sax_array
        
        return sample
    
    
class CropArraysToSquareShape(object):
    """
    Zero pads SAX image arrays to fixed square shape.
    SAX_IMAGE_SHAPE = (100, 100, 16)
    """

    def __call__(self, sample):
        SAX_IMAGE_SHAPE = (100, 100, 16)
        
        sax_array = sample['Sax_Array']
        mesh = sample['Mesh']      
        
        x, y, _ = mesh.mean(axis=0)

        x = int(x)
        y = int(y)

        x0 = x - 50
        x0 = max(0, x0)
        y0 = y - 50
        y0 = max(0, y0)

        x1 = x0 + 100
        if x1 > sax_array.shape[1]:
            x1 = sax_array.shape[1]
            x0 = x1 - 100 
                
        y1 = y0 + 100
        if y1 > sax_array.shape[0]:
            y1 = sax_array.shape[0]
            y0 = y1 - 100

        mesh[:,0] -= x0
        mesh[:,1] -= y0

        sax_array = sax_array[y0:y1, x0:x1, :]
        
        sax_z_paddings = _get_both_paddings(SAX_IMAGE_SHAPE[2], sax_array.shape[2])
        sax_array = np.pad(sax_array, ((0,0), (0,0), sax_z_paddings), 'constant', constant_values=0)
        mesh[:, 2] += sax_z_paddings[0]
        
        sample['Mesh'] = mesh
        sample['Sax_Array'] = sax_array
        
        return sample
    
    
def pad_or_crop_image_and_mesh(sax_array, mesh, new_sax_h, new_sax_w, SAX_IMAGE_SHAPE):
    # Estimates new mesh limits
    min_h = np.round(np.min(mesh[:, 1])).astype(int)
    max_h = np.round(np.max(mesh[:, 1])).astype(int)
    mesh_height = max_h - min_h

    min_w = np.round(np.min(mesh[:, 0])).astype(int)
    max_w = np.round(np.max(mesh[:, 0])).astype(int)
    mesh_width = max_w - min_w  
    
    # Adjust height
    if new_sax_h < SAX_IMAGE_SHAPE[0]:
        pad = SAX_IMAGE_SHAPE[0] - new_sax_h
        pad_h_1 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
        pad_h_2 = pad - pad_h_1
        mesh[:, 1] += pad_h_1
        sax_array = np.pad(sax_array, ((pad_h_1, pad_h_2), (0, 0), (0, 0)), 'constant', constant_values=0)
        
    elif new_sax_h > SAX_IMAGE_SHAPE[0]:        
        # Crop range
        crop_range = SAX_IMAGE_SHAPE[0] - mesh_height
        max_left_limit = new_sax_h - SAX_IMAGE_SHAPE[0] + 1
        
        if crop_range > 0 and min_h > 0:
            left_limit = np.random.randint(0, min(crop_range, min_h))
            left_limit = min(left_limit, max_left_limit)
        else:
            left_limit = min_h
            
        right_limit = left_limit + SAX_IMAGE_SHAPE[0]
            
        sax_array = sax_array[left_limit:right_limit, :, :]
        mesh[:, 1] -= left_limit

    # Adjust width
    if new_sax_w < SAX_IMAGE_SHAPE[1]:
        pad = SAX_IMAGE_SHAPE[1] - new_sax_w
        pad_w_1 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
        pad_w_2 = pad - pad_w_1
        mesh[:, 0] += pad_w_1
        sax_array = np.pad(sax_array, ((0, 0), (pad_w_1, pad_w_2), (0, 0)), 'constant', constant_values=0)
        
    elif new_sax_w > SAX_IMAGE_SHAPE[1]:
        # Crop range
        crop_range = SAX_IMAGE_SHAPE[1] - mesh_width
        max_left_limit = new_sax_w - SAX_IMAGE_SHAPE[1] + 1
        
        if crop_range > 0 and min_w > 0:
            left_limit = np.random.randint(0, min(crop_range, min_w))
            left_limit = min(left_limit, max_left_limit)
        else:
            left_limit = min_w
            
        right_limit = left_limit + SAX_IMAGE_SHAPE[1]
            
        sax_array = sax_array[:, left_limit:right_limit, :]
        mesh[:, 0] -= left_limit
        
    # Always pad the z axis to the desired shape
    padding = _get_both_paddings(SAX_IMAGE_SHAPE[2], sax_array.shape[2])
    sax_array = np.pad(sax_array, ((0, 0), (0, 0), padding), 'constant', constant_values=0)
    mesh[:, 2] += padding[0]
        
    return sax_array, mesh

class RandomScaling(object):
    """
    Scales the Short-axis and long-axis images accordingly to physical space. 
    Then crops or pads the images to the out shapes:
    SAX_IMAGE_SHAPE = (210, 210, 16)
    """

    def __call__(self, sample):
        SAX_IMAGE_SHAPE = (210, 210, 16)
        
        sax_array = sample['Sax_Array']
        mesh = sample['Mesh']
              
        resize_h_factor = np.random.uniform(0.90, 1.20)
        resize_w_factor = np.random.uniform(0.90, 1.20)

        mesh[:, 1] *= resize_h_factor
        mesh[:, 0] *= resize_w_factor
                        
        sax_h, sax_w, sax_z = sax_array.shape
        new_sax_h = np.round(sax_h * resize_h_factor).astype(int)
        new_sax_w = np.round(sax_w * resize_w_factor).astype(int)
        
        sax_array = transform.resize(sax_array, (new_sax_w, new_sax_h))
               
        sax_array, mesh = pad_or_crop_image_and_mesh(sax_array, mesh, new_sax_h, new_sax_w, SAX_IMAGE_SHAPE)
                
        sample['Sax_Array'] = sax_array
        sample['Mesh'] = mesh
        
        return sample

class CropSax(object):
    def __call__(self, sample):
        SAX_IMAGE_SHAPE = (100, 100, 16)
        
        sax_array = sample['Sax_Array']
        mesh = sample['Mesh']
        
        h, w = sax_array.shape[:2]
        
        sax_array, mesh = pad_or_crop_image_and_mesh(sax_array, mesh, h, w, SAX_IMAGE_SHAPE)
        
        sample['Sax_Array'] = sax_array
        sample['Mesh'] = mesh
        
        return sample

class AugColor(object):
    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        # Build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # Apply gamma correction using the lookup table
        return np.float32(cv2.LUT(image.astype('uint8'), table))

    def __init__(self, gamma_factor):
        self.gamma_factor = gamma_factor

    def __call__(self, sample):
        sax_image = sample['Sax_Array']
        
        # Gamma
        gamma = np.random.uniform(1 - self.gamma_factor, 1 + self.gamma_factor / 2)

        for j in range(sax_image.shape[2]):
            sax_image[:, :, j] = self.adjust_gamma(sax_image[:, :, j] * 255, gamma) / 255
        sax_image = sax_image + np.random.normal(0, 1 / 128, sax_image.shape)
        
        sample['Sax_Array'] = sax_image
        
        return sample
    
    
class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        sax_image = sample['Sax_Array']
        mesh = sample['Mesh']
        
        angle = np.random.uniform(-self.angle, self.angle)
        sax_image = transform.rotate(sax_image, angle)
        
        center = (sax_image.shape[0] / 2, sax_image.shape[1] / 2)
        
        mesh[:, :2] -= center
        
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        mesh[:, :2] = np.dot(mesh[:, :2], R)        
        mesh[:, :2] += center
        
        sample['Sax_Array'] = sax_image
        sample['Mesh'] = mesh
        
        return sample


class ToTorchTensors(object):
    def __call__(self, sample):
        sax_image = sample['Sax_Array']
        mesh = sample['Mesh']
        
        mesh[:, 0] /= sax_image.shape[1]
        mesh[:, 1] /= sax_image.shape[0]
        mesh[:, 2] /= sax_image.shape[2]
        
        sax_image_tensor = torch.from_numpy(sax_image.transpose(2, 0, 1)).unsqueeze(0).float()
        mesh_tensor = torch.from_numpy(mesh).float()
        
        return {
            'Sax_Array': sax_image_tensor,
            'Mesh': mesh_tensor
        }

class ToTorchTensorsTest(object):
    # The difference is that it returns also x0, y0 positions for the cropping and the ITK images
    # Cannot be used into a dataloader
    
    def __call__(self, sample):
        sax_image = sample['Sax_Array']
        mesh = sample['Mesh']
        
        mesh[:, 0] /= sax_image.shape[1]
        mesh[:, 1] /= sax_image.shape[0]
        mesh[:, 2] /= sax_image.shape[2]
        
        sax_image_tensor = torch.from_numpy(sax_image.transpose(2, 0, 1)).unsqueeze(0).float()
        mesh_tensor = torch.from_numpy(mesh).float()
        
        sample['Sax_Array'] = sax_image_tensor
        sample['Mesh'] = mesh_tensor
        
        return sample
    
    
class CropSax(object):
    def __call__(self, sample):
        SAX_IMAGE_SHAPE = (100, 100, 16)
        
        sax_array = sample['Sax_Array']
        mesh = sample['Mesh']
        
        h, w = sax_array.shape[:2]
        
        sax_array, mesh = pad_or_crop_image_and_mesh(sax_array, mesh, h, w, SAX_IMAGE_SHAPE)
        
        sample['Sax_Array'] = sax_array
        sample['Mesh'] = mesh
        
        return sample
        