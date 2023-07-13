#!/bin/bash

python trainer.py --name ROI_WDS_1_WL_0.01_3D_32_2D_8_KL_1e-5 --w_ds 1 --w_laplacian 0.01 --rotate 30 --latents3D 32 --latents2D 8
python trainer_noLAX.py --name ROI_WDS_1_WL_0.01_3D_32_2D_8_KL_1e-5_noLAX --w_ds 1 --w_laplacian 0.01 --rotate 30 --latents3D 32 --latents2D 8
