#!/bin/bash

# python trainer.py --name VOL_ROI_DS_1_REG_0 --volumetric --epochs 600 --w_ds 1 --w_shape 0
python trainer.py --name VOL_ROI_DS_1_REG_0.001 --volumetric --epochs 600 --w_ds 1 --w_shape 0.001
python trainer.py --name VOL_ROI_DS_1_REG_0.0001 --volumetric --epochs 600 --w_ds 1 --w_shape 0.0001
python trainer.py --name VOL_ROI_DS_1_REG_0.01 --volumetric --epochs 600 --w_ds 1 --w_shape 0.01