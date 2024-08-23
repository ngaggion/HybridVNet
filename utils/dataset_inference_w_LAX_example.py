import SimpleITK as sitk
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

def get_both_paddings(desired, actual):
    pad = (desired - actual)
    v1 = int(pad / 2)
    v2 = int(pad / 2) 
    if (v1 + v2) < pad:
        v2 += 1
    return (v1, v2)

class PredictDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.subjects = os.listdir(os.path.join(dataset_path, 'SAX'))
        self.dataset_path = dataset_path
        self.transform = transform

        self.dataframe = pd.DataFrame(columns=['subject', 'time'])
        for subject in self.subjects:
            frames = os.listdir(os.path.join(dataset_path, 'SAX', subject))
            frames = [frame for frame in frames if frame.endswith('.nii.gz')]
            for frame in frames:
                self.dataframe = pd.concat([self.dataframe, pd.DataFrame({'subject': [subject], 'time': [frame]})], ignore_index=True)

        print("Total subjects:", len(self.subjects))
        print("Total pairs of images:", len(self.dataframe))

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datapoint = self.dataframe.iloc[idx]
        subject = datapoint['subject']
        time = datapoint['time']
        
        SAX_PATH = os.path.join(self.dataset_path, "SAX", subject, time)
        LAX_2CH_PATH = os.path.join(self.dataset_path, "CH2", subject, time).replace("frame", "ch2_frame")
        LAX_3CH_PATH = os.path.join(self.dataset_path, "CH3", subject, time).replace("frame", "ch3_frame")
        LAX_4CH_PATH = os.path.join(self.dataset_path, "CH4", subject, time).replace("frame", "ch4_frame")
        
        SaxImage = sitk.ReadImage(SAX_PATH)
        SaxImage_array = sitk.GetArrayFromImage(SaxImage)
        SaxImage_array = (SaxImage_array - np.min(SaxImage_array)) / (np.max(SaxImage_array) - np.min(SaxImage_array))
        # flip z and transpose
        SaxImage_array = SaxImage_array[::-1, :, :]
        SaxImage_array = np.transpose(SaxImage_array, (1, 2, 0))
        
        def load_lax_image(path):
            try:
                img = sitk.ReadImage(path)
                arr = sitk.GetArrayFromImage(img)
                return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            except:
                return np.zeros((1, 224, 224))  # Assuming this is the expected shape

        Lax2CH_array = load_lax_image(LAX_2CH_PATH)
        Lax3CH_array = load_lax_image(LAX_3CH_PATH)
        Lax4CH_array = load_lax_image(LAX_4CH_PATH)

        sample = {
            'SAX': SaxImage,
            'Sax_Array': SaxImage_array,
            'Lax2CH_Array': Lax2CH_array,
            'Lax3CH_Array': Lax3CH_array,
            'Lax4CH_Array': Lax4CH_array
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class PadArraysToSquareShape(object):
    """
    Zero pads SAX and LAX image arrays to fixed square shape.
    SAX_IMAGE_SHAPE = (210, 210, 16)
    LAX_IMAGE_SHAPE = (224, 224, 1)
    """

    def __call__(self, sample):
        SAX_IMAGE_SHAPE = (210, 210, 16)
        LAX_IMAGE_SHAPE = (224, 224, 1)
        
        sax_array = sample['Sax_Array']
        lax2ch_array = sample['Lax2CH_Array']
        lax3ch_array = sample['Lax3CH_Array']
        lax4ch_array = sample['Lax4CH_Array']
        
        # if one size is bigger than the desired size, we crop it
        if sax_array.shape[0] > SAX_IMAGE_SHAPE[0]:
            sax_array = sax_array[:SAX_IMAGE_SHAPE[0], :, :]
        if sax_array.shape[1] > SAX_IMAGE_SHAPE[1]:
            sax_array = sax_array[:, :SAX_IMAGE_SHAPE[1], :]
        if sax_array.shape[2] > SAX_IMAGE_SHAPE[2]:
            sax_array = sax_array[:, :, :SAX_IMAGE_SHAPE[2]]

        sax_h_paddings = get_both_paddings(SAX_IMAGE_SHAPE[0], sax_array.shape[0])
        sax_w_paddings = get_both_paddings(SAX_IMAGE_SHAPE[1], sax_array.shape[1])
        sax_z_paddings = get_both_paddings(SAX_IMAGE_SHAPE[2], sax_array.shape[2])
        
        sax_array = np.pad(sax_array, (sax_h_paddings, sax_w_paddings, sax_z_paddings), 'constant', constant_values=0)
        
        def pad_lax(lax_array):
            lax_h_paddings = get_both_paddings(LAX_IMAGE_SHAPE[0], lax_array.shape[1])
            lax_w_paddings = get_both_paddings(LAX_IMAGE_SHAPE[1], lax_array.shape[2])
            return np.pad(lax_array, ((0, 0), lax_h_paddings, lax_w_paddings), 'constant', constant_values=0)
        
        sample['Sax_Array'] = sax_array
        sample['Lax2CH_Array'] = pad_lax(lax2ch_array)
        sample['Lax3CH_Array'] = pad_lax(lax3ch_array)
        sample['Lax4CH_Array'] = pad_lax(lax4ch_array)
        
        return sample

class ToTorchTensors(object):
    def __call__(self, sample):
        sax_image = sample['Sax_Array']
        lax2ch_array = sample['Lax2CH_Array']
        lax3ch_array = sample['Lax3CH_Array']
        lax4ch_array = sample['Lax4CH_Array']        
        
        sax_image_tensor = torch.from_numpy(sax_image).unsqueeze(0).float()
        lax2ch_tensor = torch.from_numpy(lax2ch_array).float()
        lax3ch_tensor = torch.from_numpy(lax3ch_array).float()
        lax4ch_tensor = torch.from_numpy(lax4ch_array).float()

        sax_image_tensor = sax_image_tensor.permute(0, 3, 1, 2)
        
        sample['Sax_Array'] = sax_image_tensor
        sample['Lax2CH_Array'] = lax2ch_tensor
        sample['Lax3CH_Array'] = lax3ch_tensor
        sample['Lax4CH_Array'] = lax4ch_tensor

        return sample