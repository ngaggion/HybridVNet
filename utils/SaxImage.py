import SimpleITK as sitk
import os
import warnings
import numpy as np

# Ignore the warning message globally
warnings.filterwarnings("ignore", message="Non uniform sampling or missing slices detected")

class SAXImage:
    def __init__(self, SAX_PATH):
        # Check if the SAX_PATH is a valid directory
        if not os.path.isdir(SAX_PATH):
            raise ValueError(f"Directory {SAX_PATH} does not exist.")
        
        # The SAX_PATH contains a series of images, one for each slice.
        self._SAX_PATH = SAX_PATH
        
        SaxImage = sitk.ImageSeriesReader()
        
        # Get the file names for the SAX images
        sax_names = sorted([os.path.join(SAX_PATH, f) for f in os.listdir(SAX_PATH) if f != "nlineVF"])[::-1]
        
        SaxImage.SetFileNames(sax_names)
        SaxImage = SaxImage.Execute()
    
        self.SaxImage = SaxImage
        
    @property
    def width(self):
        return self.SaxImage.GetSize()[0]
    
    @property
    def height(self):
        return self.SaxImage.GetSize()[1]
    
    @property
    def num_slices(self):
        return self.SaxImage.GetSize()[2]
    
    @property
    def spacing(self):
        return self.SaxImage.GetSpacing()
    
    @property
    def origin(self):
        return self.SaxImage.GetOrigin()
    
    @property
    def slice_gap(self):
        return self.spacing[2]
    
    @property
    def direction(self):
        return self.SaxImage.GetDirection()
    
    def pixel_array(self):
        # Convert the image to a  numpy array
        return np.float32(sitk.GetArrayFromImage(self.SaxImage).transpose(1, 2, 0))
        
    def __repr__(self):
        return f"Image dimension (WxH): {self.width} x {self.height}\n" \
               f"Number of slices: {self.num_slices}\n" \
               f"Spacing: {self.spacing}\n" \
               f"Slice gap: {self.slice_gap}\n" \
               f"Origin: {self.origin}"
