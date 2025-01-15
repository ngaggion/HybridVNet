import vtk
import os
import logging
import numpy as np

logger = logging.getLogger("pycardiox")

class SAXImage:
    """
    Read SAX VTK image using vtk.
    Note that vtk will read the image into a volume of dimension: x:width x y:height x z:slice
    """
    def __init__(self, filename):
        if not os.path.isfile(filename):
            logger.error(f"File {filename} does not exist.")
            raise ValueError(f"File {filename} does not exist.")
            
        self._filename = filename
        
        # read vtk file
        self.imgvtk_reader = vtk.vtkStructuredPointsReader()
        self.imgvtk_reader.SetFileName(self._filename)
        self.imgvtk_reader.Update()
        
        # Get the output data
        self.img_data = self.imgvtk_reader.GetOutput()
        self.origin = self.img_data.GetOrigin()
        
    @property
    def width(self):
        return self.img_data.GetDimensions()[0]
        
    @property
    def height(self):
        return self.img_data.GetDimensions()[1]
        
    @property
    def num_slices(self):
        return self.img_data.GetDimensions()[2]
        
    @property
    def range(self):
        return self.img_data.GetScalarRange()
        
    @property
    def spacing(self):
        return self.img_data.GetSpacing()[:2]
        
    @property
    def slice_gap(self):
        return self.img_data.GetSpacing()[2]
        
    @property
    def pixel_array(self):
        """
        Get pixel data as numpy array and swap axes from W x H x S into H x W x S
        """
        dims = self.img_data.GetDimensions()
        scalars = self.img_data.GetPointData().GetScalars()
        
        # Get scalar type and numpy equivalent
        vtk_data_type = scalars.GetDataType()
        if vtk_data_type in [3, 4]:  # float type
            np_dtype = np.float32
        else:  # assume int16 for other types
            np_dtype = np.int16
            
        # Create numpy array and copy data
        n_points = dims[0] * dims[1] * dims[2]
        array = np.zeros(n_points, dtype=np_dtype)
        for i in range(n_points):
            array[i] = scalars.GetTuple1(i)
            
        # Reshape to volume dimensions and transpose
        vol_array = array.reshape([dims[2], dims[1], dims[0]])
        return np.transpose(vol_array, (1, 2, 0))
        
    def get_image_slice(self, sl):
        """Get a single slice from the volume"""
        return self.pixel_array[:, :, sl]
        
    def get_image_plane(self, sl):
        """Returns (point, normal_vector) tuple"""
        return np.add(self.origin, [0, 0, sl * self.slice_gap]), [0, 0, 1]
        
    def __repr__(self):
        return (f"Image dimension (WxH): {self.width} x {self.height}\n"
                f"Number of slices: {self.num_slices}\n"
                f"Scalar range: {self.range}\n"
                f"Spacing: {self.spacing}\n"
                f"Slice gap: {self.slice_gap}\n"
                f"Origin: {self.origin}")