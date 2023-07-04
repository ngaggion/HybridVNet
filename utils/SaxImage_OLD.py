import vedo
import vtk
import os
import logging
import numpy as np

"""
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

logger = logging.getLogger("pycardiox")


class SAXImage:
    """
    Read SAX VTK image using vedo/vtk.
    Note that vtk will read the image into a volume of dimension: x:width x y:height x z:slice
    """

    def __init__(self, filename):
        if not os.path.isfile(filename):
            logger.error(f"File {filename} does not exist.")
            raise ValueError(f"File {filename} does not exist.")
        self._filename = filename

        # read
        imgvtk_reader = vtk.vtkStructuredPointsReader()
        imgvtk_reader.SetFileName(self._filename)
        imgvtk_reader.Update()

        # origin for the volume != vtk image
        img_data = imgvtk_reader.GetOutput()
        self.origin = img_data.GetOrigin()

        self.vol = vedo.Volume(imgvtk_reader)

    @property
    def width(self):
        return self.vol.dimensions()[0]

    @property
    def height(self):
        return self.vol.dimensions()[1]

    @property
    def num_slices(self):
        return self.vol.dimensions()[2]

    @property
    def range(self):
        return self.vol.scalarRange()

    @property
    def spacing(self):
        return self.vol.spacing()[:2]

    @property
    def slice_gap(self):
        return self.vol.spacing()[-1]

    @property
    def pixel_array(self):
        """
        Swap axis from W x H x S into H x W x S
        """
        return np.swapaxes(self.vol.getDataArray(), 0, 1)

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
               f"Scalar range: {self.range}\n" \
               f"Spacing: {self.spacing}\n" \
               f"Slice gap: {self.slice_gap}\n" \
               f"Origin: {self.origin}"
