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
