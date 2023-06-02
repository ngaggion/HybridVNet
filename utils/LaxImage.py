import os
import numpy as np
import SimpleITK as sitk

class LAXImage:
    def __init__(self, filename):
        if not os.path.isfile(filename):
            raise ValueError(f"File {filename} does not exist.")
        self._filename = filename

        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(filename)

        itkimage = file_reader.Execute()

        self.itkimage = itkimage

        self.origin = self.itkimage.GetOrigin()

    @property
    def width(self):
        return self.itkimage.GetSize()[0]

    @property
    def height(self):
        return self.itkimage.GetSize()[1]

    @property
    def num_slices(self):
        return self.itkimage.GetSize()[2]

    @property
    def range(self):
        return self.itkimage.scalarRange()

    @property
    def spacing(self):
        return self.itkimage.GetSpacing()[:2]

    @property
    def slice_gap(self):
        return self.itkimage.GetSpacing()[-1]

    def pixel_array(self):
        # Convert the image to a  numpy array
        return np.float32(sitk.GetArrayFromImage(self.itkimage).transpose(1, 2, 0))

    def get_image_slice(self, sl):
        return self.pixel_array[:, :, sl]

    def get_image_plane(self, sl):
        """
        Returns (point, normal_vector) tuple
        """
        return self.origin + [0, 0, sl * self.slice_gap], [0, 0, 1]

    def __repr__(self):
        return f"Image dimension (WxH): {self.width} x {self.height}\n" \
               f"Number of slices: {self.num_slices}\n" \
               f"Spacing: {self.spacing}\n" \
               f"Slice gap: {self.slice_gap}\n" \
               f"Origin: {self.origin}"

