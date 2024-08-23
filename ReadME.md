# HybridVNet: Multi-view Hybrid Graph Convolutional Network for Volume-to-mesh Reconstruction in Cardiovascular MRI

This repository contains the implementation of HybridVNet, a novel architecture for generating high-quality surface and volumetric meshes directly from Cardiovascular Magnetic Resonance (CMR) images, as described in our paper "Multi-view Hybrid Graph Convolutional Network for Volume-to-mesh Reconstruction in Cardiovascular MRI".

## Features

- Direct volume-to-mesh reconstruction from CMR images
- Multiview integration of Short Axis (SAX) and Long Axis (LAX) CMR views
- Generation of both surface and tetrahedral meshes
- Novel differentiable regularization term for tetrahedral meshes

## Dependencies

Main dependencies include:
- PyTorch
- PyTorch Geometric
- PyTorch3D (only required for training, for the surface mesh losses)
- psbody (only required to generate downsampling and upsampling matrices)
- NumPy
- Matplotlib
- SimpleITK

Install instructions soon.

## Usage

Download the Docker image from the Docker Hub repository by running:

```bash
docker pull ngaggion/hybridvnet:latest
```

Then, run the Docker container with the following command:

```bash
MOUNT="YOUR_LOCAL_INFORMATION_PATH"

docker run -it --gpus all \
    -v $MOUNT:/DATA/ \
    hybrivnet:latest
```

It's recommended to always pull from the repo when starting the docker.

```bash
git pull
```

After using the container, it's recommended to restrict access to the X server with the following command:

```bash
xhost -local:docker
```

### Docker Usage Notes

To enable GPU support within the Docker container, it's required to install the nvidia-docker2 package. **Please note that we are using CUDA 11.8, given your GPU restrictions you may want to build your own image.** In this case, you'll **only** need to modify the first line of the Dockerfile using any official pytorch >= 2.0.0 docker image that works with your hardware and build it from scratch.

For Ubuntu-based distributions please follow these steps:

1. **Add the GPG key:**

    ```bash
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    ```

2. **Add the repository:**

    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/nvidia-docker/$distribution/$(arch)/" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    ```

3. **Update your package list:**

    ```bash
    sudo apt-get update
    ```

4. **Install NVIDIA Docker2:**

    ```bash
    sudo apt-get install -y nvidia-docker2
    ```

## Citation

If you use this code in your research, please cite our paper:

```
@article{gaggion2023multi,
  title={Multi-view Hybrid Graph Convolutional Network for Volume-to-mesh Reconstruction in Cardiovascular MRI},
  author={Gaggion, Nicol{\'a}s and Matheson, Benjamin A and Xia, Yan and Bonazzola, Rodrigo and Ravikumar, Nishant and Taylor, Zeike A and Milone, Diego H and Frangi, Alejandro F and Ferrante, Enzo},
  journal={arXiv preprint arXiv:2311.13706},
  year={2023}
}
```

## Contributing

We welcome contributions to improve HybridVNet. Please feel free to submit issues and pull requests.
