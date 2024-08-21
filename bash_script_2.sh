#!/bin/bash

python trainer.py --name FM_ROI --w_ds 1 --w_laplacian 0.01 --rotate 30 --latents3D 32 --latents2D 8
python trainer_noLAX.py --name FM_ROI_noLAX --w_ds 1 --w_laplacian 0.01 --rotate 30 --latents3D 32 --latents2D 8
python trainer.py --name FM_FULL --full --w_ds 1 --w_laplacian 0.01 --rotate 30 --latents3D 32 --latents2D 8
python trainer_noLAX.py --name FM_FULL_noLAX --full --w_ds 1 --w_laplacian 0.01 --rotate 30 --latents3D 32 --latents2D 8
