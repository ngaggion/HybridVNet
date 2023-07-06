from utils.SaxImage_VTK import SAXImage
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
    min_h = np.min(mesh[:, 1])
    max_h = np.max(mesh[:, 1])
    mesh_height = max_h - min_h

    min_w = np.min(mesh[:, 0])
    max_w = np.max(mesh[:, 0])
    mesh_width = max_w - min_w  
    
    if mesh_height > SAX_IMAGE_SHAPE[0]:
        print("Height bigger than ROI")
    elif mesh_width > SAX_IMAGE_SHAPE[1]:
        print("Width bigger than ROI")

    # Adjust height
    if new_sax_h < SAX_IMAGE_SHAPE[0]:
        padding = _get_both_paddings(SAX_IMAGE_SHAPE[0], new_sax_h)
        pad_h_1 = np.random.randint(0, padding[0] + padding[1])
        pad_h_2 = padding[0] + padding[1] - pad_h_1
        mesh[:, 1] += pad_h_1
        sax_array = np.pad(sax_array, ((pad_h_1, pad_h_2), (0, 0), (0, 0)), 'constant', constant_values=0)
        
    elif new_sax_h > SAX_IMAGE_SHAPE[0]:        
        crop_amount = new_sax_h - SAX_IMAGE_SHAPE[0]
        
        # Ensure we don't crop the mesh from the left
        crop_left_limit = min(crop_amount, int(min_h))
        
        # Ensure we don't crop the mesh from the right
        mesh_limit = max_h - SAX_IMAGE_SHAPE[0]
        mesh_limit = int(round(max(mesh_limit, 0)))

        try:
            random_crop_left = np.random.randint(mesh_limit, crop_left_limit)
        except:
            if mesh_limit == crop_left_limit:
                random_crop_left = mesh_limit
            elif mesh_limit > crop_left_limit:
                random_crop_left = crop_left_limit
                
        right_limit = random_crop_left + SAX_IMAGE_SHAPE[0]
        
        if right_limit >= new_sax_h:
            right_limit = new_sax_h
            random_crop_left = new_sax_h - SAX_IMAGE_SHAPE[0]
            
        sax_array = sax_array[random_crop_left:right_limit, :, :]
        mesh[:, 1] -= random_crop_left 

    # Adjust width
    if new_sax_w < SAX_IMAGE_SHAPE[1]:
        padding = _get_both_paddings(SAX_IMAGE_SHAPE[1], new_sax_w)
        pad_w_1 = np.random.randint(0, padding[0] + padding[1])
        pad_w_2 = padding[0] + padding[1] - pad_w_1
        mesh[:, 0] += pad_w_1
        sax_array = np.pad(sax_array, ((0, 0), (pad_w_1, pad_w_2), (0, 0)), 'constant', constant_values=0)
        
    elif new_sax_w > SAX_IMAGE_SHAPE[1]:
        crop_amount = new_sax_w - SAX_IMAGE_SHAPE[1]
        crop_left_limit = min(crop_amount, int(min_w))
        # Ensure we don't crop the mesh from the right
        mesh_limit = max_w - SAX_IMAGE_SHAPE[1]
        mesh_limit = int(round(max(mesh_limit, 0)))

        try:
            random_crop_left = np.random.randint(mesh_limit, crop_left_limit)
        except:
            if mesh_limit == crop_left_limit:
                random_crop_left = mesh_limit
            elif mesh_limit > crop_left_limit:
                random_crop_left = crop_left_limit
        
        right_limit = random_crop_left + SAX_IMAGE_SHAPE[1]
        
        if right_limit >= new_sax_w:
            right_limit = new_sax_w
            random_crop_left = new_sax_w - SAX_IMAGE_SHAPE[1]
            
        sax_array = sax_array[:, random_crop_left:right_limit, :]
        mesh[:, 0] -= random_crop_left
        
    # Always pad the z axis to the desired shape
    padding = _get_both_paddings(SAX_IMAGE_SHAPE[2], sax_array.shape[2])
    sax_array = np.pad(sax_array, ((0, 0), (0, 0), padding), 'constant', constant_values=0)
    mesh[:, 2] += padding[0]
        
    return sax_array, mesh


def pad_or_crop_image(image, output_shape):
    input_shape = image.shape[:2]
    target_height, target_width = output_shape

    if input_shape == output_shape:
        return image

    # First it padds the image to the desired shape
    
    pad_height = max(target_height - input_shape[0], 0)
    pad_width = max(target_width - input_shape[1], 0)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    
    # Then croppes the image to the desired shape if needed
    
    input_shape = padded_image.shape[:2]
    crop_height = max(input_shape[0] - target_height, 0)
    crop_width = max(input_shape[1] - target_width, 0)

    crop_top = crop_height // 2
    crop_bottom = crop_height - crop_top
    crop_left = crop_width // 2
    crop_right = crop_width - crop_left
    
    cropped_image = padded_image[crop_top:input_shape[0] + pad_top - crop_bottom, crop_left:input_shape[1] + pad_left - crop_right]

    return cropped_image


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
              
        resize_h_factor = np.random.uniform(0.70, 1.30)
        resize_w_factor = np.random.uniform(0.70, 1.30)
                
        sax_h, sax_w, sax_z = sax_array.shape
        new_sax_h = int(round(sax_h * resize_h_factor, 0))
        new_sax_w = int(round(sax_w * resize_w_factor, 0))
        
        # The sax_array is resized to the new shape
        sax_array = cv2.resize(sax_array, (new_sax_w, new_sax_h))
        
        # The real scaling factor is calculated due to the rounding of the new shape
        resize_h_factor = new_sax_h / sax_h
        resize_w_factor = new_sax_w / sax_w
        
        mesh[:, 1] *= resize_h_factor
        mesh[:, 0] *= resize_w_factor
        
        sax_array, mesh = pad_or_crop_image_and_mesh(sax_array, mesh, new_sax_h, new_sax_w, SAX_IMAGE_SHAPE)
                
        sample['Sax_Array'] = sax_array
        sample['Mesh'] = mesh
        
        return sample

class RandomCrop(object):
    """
    Scales the Short-axis and long-axis images accordingly to physical space. 
    Then crops or pads the images to the out shapes:
    SAX_IMAGE_SHAPE = (100, 100, 16)
    """

    def __call__(self, sample):
        SAX_IMAGE_SHAPE = (100, 100, 16)
        
        sax_array = sample['Sax_Array']
        mesh = sample['Mesh']
              
        resize_h_factor = np.random.uniform(0.70, 1.30)
        resize_w_factor = np.random.uniform(0.70, 1.30)
                
        sax_h, sax_w, sax_z = sax_array.shape
        new_sax_h = int(round(sax_h * resize_h_factor, 0))
        new_sax_w = int(round(sax_w * resize_w_factor, 0))
        
        # The sax_array is resized to the new shape
        sax_array = cv2.resize(sax_array, (new_sax_w, new_sax_h))
        
        # The real scaling factor is calculated due to the rounding of the new shape
        resize_h_factor = new_sax_h / sax_h
        resize_w_factor = new_sax_w / sax_w
        
        mesh[:, 1] *= resize_h_factor
        mesh[:, 0] *= resize_w_factor
        
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
        
        # columns x and y are inverted
        mesh[:, :2] = np.dot(mesh[:, :2], R)        
        mesh[:, :2] += center
        
        sample['Sax_Array'] = sax_image
        sample['Mesh'] = mesh
        
        return sample


class ToTorchTensors(object):
    def __call__(self, sample):
        sax_image = sample['Sax_Array']
        mesh = sample['Mesh']
        
        mesh[:, 0] /= sax_image.shape[0]
        mesh[:, 1] /= sax_image.shape[1]
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
        
        mesh[:, 0] /= sax_image.shape[0]
        mesh[:, 1] /= sax_image.shape[1]
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
        